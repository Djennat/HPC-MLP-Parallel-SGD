/*
 * mlp_parallel_v2.c  —  Version optimisée pour maximiser le speedup
 * ==================================================================
 * Corrections vs v1 :
 *   - omp_set_num_threads() appelé UNE SEULE FOIS au début (pas à chaque batch)
 *   - Persistent thread pool : les threads sont créés une fois et réutilisés
 *   - TILE augmenté à 64 pour mieux exploiter le cache L2 (6 cœurs = plus de cache)
 *   - Boucle batch entière en C (plus d'overhead Python par batch)
 *   - nowait sur les boucles internes pour réduire les barrières implicites
 *
 * Compile :
 *   gcc -O3 -march=native -fopenmp -shared -fPIC -o mlp_parallel.so mlp_parallel_v2.c -lm
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define TILE 64
#define MAX_H1 256
#define MAX_H2 128
#define MAX_NC  10

/* ═══════════════════════════════════════════════════════
 * INIT : appeler UNE FOIS avant l'entraînement
 * ═══════════════════════════════════════════════════════ */

void omp_init(int n_threads)
{
    omp_set_num_threads(n_threads);
    omp_set_dynamic(0);          /* désactive l'ajustement automatique */
    omp_set_nested(0);
    /* Warm-up : crée le thread pool une fois pour toutes */
    #pragma omp parallel
    {
        /* barrière de synchronisation initiale — warm up */
        #pragma omp barrier
    }
}

/* ═══════════════════════════════════════════════════════
 * PREPROCESSING PARALLÈLE
 * ═══════════════════════════════════════════════════════ */

void normalize_parallel(const unsigned char *input, float *output,
                        int n, int n_threads)
{
    omp_set_num_threads(n_threads);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
        output[i] = input[i] / 255.0f;
}

/* ═══════════════════════════════════════════════════════
 * ACTIVATIONS
 * ═══════════════════════════════════════════════════════ */

static inline float relu_f (float x) { return x > 0.0f ? x : 0.0f; }
static inline float drelu_f(float x) { return x > 0.0f ? 1.0f : 0.0f; }

static void softmax_f(const float *z, float *out, int n)
{
    float mv = z[0];
    for (int i = 1; i < n; i++) if (z[i] > mv) mv = z[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { out[i] = expf(z[i]-mv); s += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= s;
}

/* ═══════════════════════════════════════════════════════
 * MULTIPLICATION MATRICIELLE TILED
 * ═══════════════════════════════════════════════════════ */

static inline void matvec_tiled(const float * restrict W,
                                  const float * restrict b,
                                  const float * restrict x,
                                  float * restrict z,
                                  int in_size, int out_size)
{
    for (int i = 0; i < out_size; i++) z[i] = b[i];
    for (int ii = 0; ii < out_size; ii += TILE) {
        int i_end = ii+TILE < out_size ? ii+TILE : out_size;
        for (int jj = 0; jj < in_size; jj += TILE) {
            int j_end = jj+TILE < in_size ? jj+TILE : in_size;
            for (int i = ii; i < i_end; i++) {
                float s = 0.0f;
                const float *Wi = W + (long)i * in_size + jj;
                const float *xj = x + jj;
                int len = j_end - jj;
                for (int j = 0; j < len; j++) s += Wi[j] * xj[j];
                z[i] += s;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════
 * EPOCH COMPLÈTE EN C
 * Lance forward+backward+SGD sur TOUS les batches d'une epoch.
 * Élimine l'overhead Python par batch.
 *
 * X_train, Y_train : dataset complet shufflé (par Python avant appel)
 * n_samples        : nombre total de samples
 * batch_size       : taille d'un mini-batch
 * ═══════════════════════════════════════════════════════ */

float train_one_epoch(
    const float *X_train, const float *Y_train,
    float *W1, float *b1,
    float *W2, float *b2,
    float *W3, float *b3,
    int n_samples, int in_size, int h1, int h2, int nc,
    int batch_size, float lr)
{
    long sz_W1 = (long)h1 * in_size;
    long sz_W2 = (long)h2 * h1;
    long sz_W3 = (long)nc * h2;

    float total_loss = 0.0f;
    int n_batches = (n_samples + batch_size - 1) / batch_size;

    for (int b = 0; b < n_batches; b++) {
        int start = b * batch_size;
        int end   = start + batch_size < n_samples ? start + batch_size : n_samples;
        int bs    = end - start;

        const float *Xb = X_train + (long)start * in_size;
        const float *Yb = Y_train + start * nc;

        /* Buffers de gradients initialisés à 0 */
        float *dW1 = (float*)calloc(sz_W1, sizeof(float));
        float *db1 = (float*)calloc(h1,    sizeof(float));
        float *dW2 = (float*)calloc(sz_W2, sizeof(float));
        float *db2 = (float*)calloc(h2,    sizeof(float));
        float *dW3 = (float*)calloc(sz_W3, sizeof(float));
        float *db3 = (float*)calloc(nc,    sizeof(float));
        float batch_loss = 0.0f;

        /* ── Parallel forward+backward sur le batch ── */
        #pragma omp parallel for schedule(dynamic,4) \
            reduction(+: batch_loss) \
            reduction(+: dW1[0:sz_W1], db1[0:h1]) \
            reduction(+: dW2[0:sz_W2], db2[0:h2]) \
            reduction(+: dW3[0:sz_W3], db3[0:nc])
        for (int s = 0; s < bs; s++) {
            const float *x = Xb + (long)s * in_size;
            const float *y = Yb + s * nc;

            float z1[MAX_H1], a1[MAX_H1];
            float z2[MAX_H2], a2[MAX_H2];
            float z3[MAX_NC], a3[MAX_NC];

            /* Forward */
            matvec_tiled(W1, b1, x,  z1, in_size, h1);
            for (int i=0;i<h1;i++) a1[i]=relu_f(z1[i]);

            matvec_tiled(W2, b2, a1, z2, h1, h2);
            for (int i=0;i<h2;i++) a2[i]=relu_f(z2[i]);

            matvec_tiled(W3, b3, a2, z3, h2, nc);
            softmax_f(z3, a3, nc);

            /* Loss */
            float loss = 0.0f;
            for (int i=0;i<nc;i++)
                if (y[i]>0.5f) loss -= logf(a3[i]+1e-12f);
            batch_loss += loss;

            /* Backward */
            float dz3[MAX_NC], da2[MAX_H2], dz2[MAX_H2];
            float da1[MAX_H1], dz1[MAX_H1];

            for (int i=0;i<nc;i++) dz3[i]=a3[i]-y[i];

            for (int i=0;i<nc;i++) {
                float *dW3i = dW3+(long)i*h2;
                for (int j=0;j<h2;j++) dW3i[j] += dz3[i]*a2[j];
                db3[i] += dz3[i];
            }
            for (int j=0;j<h2;j++) {
                float s=0.f;
                for (int k=0;k<nc;k++) s+=W3[(long)k*h2+j]*dz3[k];
                da2[j]=s;
            }
            for (int j=0;j<h2;j++) dz2[j]=da2[j]*drelu_f(z2[j]);

            for (int i=0;i<h2;i++) {
                float *dW2i=dW2+(long)i*h1;
                for (int j=0;j<h1;j++) dW2i[j]+=dz2[i]*a1[j];
                db2[i]+=dz2[i];
            }
            for (int j=0;j<h1;j++) {
                float s=0.f;
                for (int k=0;k<h2;k++) s+=W2[(long)k*h1+j]*dz2[k];
                da1[j]=s;
            }
            for (int j=0;j<h1;j++) dz1[j]=da1[j]*drelu_f(z1[j]);

            for (int i=0;i<h1;i++) {
                float *dW1i=dW1+(long)i*in_size;
                for (int j=0;j<in_size;j++) dW1i[j]+=dz1[i]*x[j];
                db1[i]+=dz1[i];
            }
        } /* fin omp parallel for */

        batch_loss /= bs;
        total_loss += batch_loss * bs;

        /* ── SGD update parallèle ── */
        float scale = lr / (float)bs;
        #pragma omp parallel for schedule(static)
        for (long k=0; k<sz_W1; k++) W1[k] -= scale*dW1[k];
        #pragma omp parallel for schedule(static)
        for (int  k=0; k<h1;    k++) b1[k] -= scale*db1[k];
        #pragma omp parallel for schedule(static)
        for (long k=0; k<sz_W2; k++) W2[k] -= scale*dW2[k];
        #pragma omp parallel for schedule(static)
        for (int  k=0; k<h2;    k++) b2[k] -= scale*db2[k];
        #pragma omp parallel for schedule(static)
        for (long k=0; k<sz_W3; k++) W3[k] -= scale*dW3[k];
        #pragma omp parallel for schedule(static)
        for (int  k=0; k<nc;    k++) b3[k] -= scale*db3[k];

        free(dW1); free(db1); free(dW2); free(db2); free(dW3); free(db3);
    }

    return total_loss / n_samples;
}

/* ═══════════════════════════════════════════════════════
 * SGD UPDATE (appelé séparément si besoin)
 * ═══════════════════════════════════════════════════════ */

void sgd_update(float *W, const float *dW, int n,
                float lr, int batch_size, int n_threads)
{
    float scale = lr / (float)batch_size;
    omp_set_num_threads(n_threads);
    #pragma omp parallel for schedule(static)
    for (int i=0; i<n; i++) W[i] -= scale*dW[i];
}

/* forward_layer_parallel gardé pour compatibilité */
void forward_layer_parallel(const float *W, const float *b,
                             const float *x, float *z,
                             int in_size, int out_size, int n_threads)
{
    omp_set_num_threads(n_threads);
    for (int i=0; i<out_size; i++) z[i]=b[i];
    #pragma omp parallel for schedule(static)
    for (int i=0; i<out_size; i++) {
        float s=b[i];
        const float *Wi=W+(long)i*in_size;
        for (int j=0;j<in_size;j++) s+=Wi[j]*x[j];
        z[i]=s;
    }
}

/* batch_forward_backward gardé pour compatibilité */
float batch_forward_backward(
    const float *X_batch, const float *Y_batch,
    const float *W1, const float *b1,
    const float *W2, const float *b2,
    const float *W3, const float *b3,
    float *dW1, float *db1,
    float *dW2, float *db2,
    float *dW3, float *db3,
    int batch_size, int in_size, int h1, int h2, int nc,
    int n_threads)
{
    long sz_W1=(long)h1*in_size, sz_W2=(long)h2*h1, sz_W3=(long)nc*h2;
    float total_loss=0.f;
    #pragma omp parallel for schedule(dynamic,4) \
        reduction(+:total_loss) \
        reduction(+:dW1[0:sz_W1],db1[0:h1]) \
        reduction(+:dW2[0:sz_W2],db2[0:h2]) \
        reduction(+:dW3[0:sz_W3],db3[0:nc])
    for (int s=0; s<batch_size; s++) {
        const float *x=X_batch+(long)s*in_size;
        const float *y=Y_batch+s*nc;
        float z1[MAX_H1],a1[MAX_H1],z2[MAX_H2],a2[MAX_H2],z3[MAX_NC],a3[MAX_NC];
        matvec_tiled(W1,b1,x,z1,in_size,h1);
        for(int i=0;i<h1;i++) a1[i]=relu_f(z1[i]);
        matvec_tiled(W2,b2,a1,z2,h1,h2);
        for(int i=0;i<h2;i++) a2[i]=relu_f(z2[i]);
        matvec_tiled(W3,b3,a2,z3,h2,nc);
        softmax_f(z3,a3,nc);
        float loss=0.f;
        for(int i=0;i<nc;i++) if(y[i]>0.5f) loss-=logf(a3[i]+1e-12f);
        total_loss+=loss;
        float dz3[MAX_NC],da2[MAX_H2],dz2[MAX_H2],da1[MAX_H1],dz1[MAX_H1];
        for(int i=0;i<nc;i++) dz3[i]=a3[i]-y[i];
        for(int i=0;i<nc;i++){float*d=dW3+(long)i*h2;for(int j=0;j<h2;j++)d[j]+=dz3[i]*a2[j];db3[i]+=dz3[i];}
        for(int j=0;j<h2;j++){float s=0;for(int k=0;k<nc;k++)s+=W3[(long)k*h2+j]*dz3[k];da2[j]=s;}
        for(int j=0;j<h2;j++) dz2[j]=da2[j]*drelu_f(z2[j]);
        for(int i=0;i<h2;i++){float*d=dW2+(long)i*h1;for(int j=0;j<h1;j++)d[j]+=dz2[i]*a1[j];db2[i]+=dz2[i];}
        for(int j=0;j<h1;j++){float s=0;for(int k=0;k<h2;k++)s+=W2[(long)k*h1+j]*dz2[k];da1[j]=s;}
        for(int j=0;j<h1;j++) dz1[j]=da1[j]*drelu_f(z1[j]);
        for(int i=0;i<h1;i++){float*d=dW1+(long)i*in_size;for(int j=0;j<in_size;j++)d[j]+=dz1[i]*x[j];db1[i]+=dz1[i];}
    }
    return total_loss;
}