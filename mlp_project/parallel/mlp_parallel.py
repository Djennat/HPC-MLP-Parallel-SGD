"""
mlp_parallel_v2.py
==================
Wrapper Python optimisé — appelle train_one_epoch() directement en C
pour éliminer l'overhead Python par batch.
"""

import os, sys, ctypes, time, json, math, random
import numpy as np

LIB_PATH = os.path.join(os.path.dirname(__file__), 'mlp_parallel_v2.so')
if not os.path.exists(LIB_PATH):
    print("[ERREUR] mlp_parallel_v2.so introuvable.")
    print("Compile avec :")
    print("  gcc -O3 -march=native -fopenmp -shared -fPIC -o mlp_parallel_v2.so mlp_parallel_v2.c -lm")
    sys.exit(1)

lib = ctypes.CDLL(LIB_PATH)
FP  = ctypes.POINTER(ctypes.c_float)
UFP = ctypes.POINTER(ctypes.c_ubyte)

# Signatures
lib.omp_init.restype  = None
lib.omp_init.argtypes = [ctypes.c_int]

lib.normalize_parallel.restype  = None
lib.normalize_parallel.argtypes = [UFP, FP, ctypes.c_int, ctypes.c_int]

lib.train_one_epoch.restype  = ctypes.c_float
lib.train_one_epoch.argtypes = [
    FP, FP,               # X_train, Y_train
    FP, FP, FP, FP, FP, FP,  # W1,b1,W2,b2,W3,b3
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_float  # batch_size, lr
]

def ptr(a):
    return a.ctypes.data_as(FP)

def uptr(a):
    return a.ctypes.data_as(UFP)

# ── Preprocessing parallèle ──
def normalize_parallel(X_uint8, n_threads):
    n     = X_uint8.size
    X_in  = np.ascontiguousarray(X_uint8.flatten(), dtype=np.uint8)
    X_out = np.zeros(n, dtype=np.float32)
    lib.normalize_parallel(uptr(X_in), ptr(X_out), ctypes.c_int(n), ctypes.c_int(n_threads))
    return X_out.reshape(X_uint8.shape)

# ── One-hot ──
def one_hot_np(labels, nc=10):
    Y = np.zeros((len(labels), nc), dtype=np.float32)
    Y[np.arange(len(labels)), labels] = 1.0
    return Y

# ── Shuffle ──
def shuffle_np(X, Y, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return np.ascontiguousarray(X[idx]), np.ascontiguousarray(Y[idx])

# ── Xavier init ──
def xavier(fi, fo, seed):
    rng = np.random.default_rng(seed)
    lim = math.sqrt(6.0/(fi+fo))
    return rng.uniform(-lim, lim, (fo, fi)).astype(np.float32)

# ── Evaluation NumPy ──
def evaluate(X, y, p):
    def softmax(Z):
        E = np.exp(Z - Z.max(axis=1,keepdims=True))
        return E/E.sum(axis=1,keepdims=True)
    a1 = np.maximum(0, X @ p['W1'].T + p['b1'])
    a2 = np.maximum(0, a1 @ p['W2'].T + p['b2'])
    a3 = softmax(a2 @ p['W3'].T + p['b3'])
    return float(np.mean(np.argmax(a3,1)==y))

# ── Entraînement ──
def train(X_train, y_train, X_test, y_test,
          h1=256, h2=128, nc=10,
          epochs=15, batch_size=64, lr=0.01, n_threads=4):

    in_size = X_train.shape[1]
    print(f"\nArchitecture : {in_size}→{h1}→{h2}→{nc}")
    print(f"Epochs:{epochs} | Batch:{batch_size} | LR:{lr} | Threads:{n_threads}\n")

    # Init thread pool UNE SEULE FOIS
    lib.omp_init(ctypes.c_int(n_threads))

    p = {
        'W1': xavier(in_size, h1, 42), 'b1': np.zeros((1,h1), dtype=np.float32),
        'W2': xavier(h1, h2, 43),      'b2': np.zeros((1,h2), dtype=np.float32),
        'W3': xavier(h2, nc, 44),      'b3': np.zeros((1,nc), dtype=np.float32),
    }

    Y_onehot = one_hot_np(y_train, nc)
    results  = []
    t_total  = time.time()

    for epoch in range(1, epochs+1):
        t0 = time.time()

        # Shuffle côté Python (from scratch via np seed fixe)
        X_s, Y_s = shuffle_np(X_train, Y_onehot, seed=epoch)

        # Passer les matrices W en row-major (transposées)
        W1f = np.ascontiguousarray(p['W1'])   # (h1, in_size)
        W2f = np.ascontiguousarray(p['W2'])   # (h2, h1)
        W3f = np.ascontiguousarray(p['W3'])   # (nc, h2)
        b1f = np.ascontiguousarray(p['b1'].flatten())
        b2f = np.ascontiguousarray(p['b2'].flatten())
        b3f = np.ascontiguousarray(p['b3'].flatten())

        # ── Appel C : epoch entière ──
        avg_loss = lib.train_one_epoch(
            ptr(X_s), ptr(Y_s),
            ptr(W1f), ptr(b1f),
            ptr(W2f), ptr(b2f),
            ptr(W3f), ptr(b3f),
            ctypes.c_int(len(X_s)), ctypes.c_int(in_size),
            ctypes.c_int(h1), ctypes.c_int(h2), ctypes.c_int(nc),
            ctypes.c_int(batch_size), ctypes.c_float(lr)
        )

        # Récupérer les poids mis à jour
        p['W1'] = W1f; p['b1'] = b1f.reshape(1,-1)
        p['W2'] = W2f; p['b2'] = b2f.reshape(1,-1)
        p['W3'] = W3f; p['b3'] = b3f.reshape(1,-1)

        epoch_time = time.time() - t0
        acc = evaluate(X_test, y_test, p)

        results.append({'epoch':epoch,'loss':round(float(avg_loss),6),
                        'accuracy':round(acc,6),'time_s':round(epoch_time,3)})
        print(f"  Epoch {epoch:2d}/{epochs} | Loss:{avg_loss:.4f} | "
              f"Acc:{acc*100:.2f}% | Time:{epoch_time:.2f}s | T:{n_threads}")

    total_time = time.time() - t_total
    final_acc  = evaluate(X_test, y_test, p)
    print(f"\n{'='*55}")
    print(f"  Threads : {n_threads}  |  Temps : {total_time:.2f}s  |  Acc : {final_acc*100:.2f}%")
    print(f"{'='*55}")
    return results, total_time, final_acc


def main():
    DATA_DIR    = os.path.join(os.path.dirname(__file__),'..','data')
    RESULTS_DIR = os.path.join(os.path.dirname(__file__),'..','results')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Chargement des données ...")
    X_raw  = np.load(os.path.join(DATA_DIR,'X_train.npy'))
    y_train= np.load(os.path.join(DATA_DIR,'y_train.npy')).astype(np.int64)
    Xt_raw = np.load(os.path.join(DATA_DIR,'X_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR,'y_test.npy')).astype(np.int64)

    print("\nPreprocessing parallèle ...")
    t0 = time.time()
    X_train = normalize_parallel(X_raw,  n_threads=6)
    X_test  = normalize_parallel(Xt_raw, n_threads=6)
    print(f"  Terminé en {time.time()-t0:.3f}s")

    thread_configs = [1, 2, 4, 6, 8]
    all_results = []

    for n_threads in thread_configs:
        print(f"\n{'─'*55}")
        print(f"  Test avec {n_threads} thread(s)")
        print(f"{'─'*55}")
        results, train_time, final_acc = train(
            X_train, y_train, X_test, y_test,
            epochs=15, batch_size=64, lr=0.01, n_threads=n_threads
        )
        all_results.append({
            'n_threads': n_threads,
            'training_time_s': round(train_time,4),
            'final_accuracy': round(final_acc,6),
            'per_epoch': results
        })

    # Sauvegarde
    out = os.path.join(RESULTS_DIR,'parallel_results_v2.json')
    with open(out,'w') as f:
        json.dump(all_results, f, indent=2)

    # Tableau speedup
    t1 = next(r['training_time_s'] for r in all_results if r['n_threads']==1)
    print(f"\n{'='*60}")
    print(f"  {'Threads':>8} {'Temps(s)':>12} {'Speedup':>10} {'Efficacité':>12} {'Accuracy':>10}")
    print(f"  {'-'*56}")
    for r in all_results:
        sp  = t1 / r['training_time_s']
        eff = sp  / r['n_threads'] * 100
        print(f"  {r['n_threads']:>8} {r['training_time_s']:>12.2f} "
              f"{sp:>9.2f}x {eff:>11.1f}% {r['final_accuracy']*100:>9.2f}%")
    print(f"{'='*60}")
    print(f"\nRésultats → {out}")

if __name__ == "__main__":
    main()