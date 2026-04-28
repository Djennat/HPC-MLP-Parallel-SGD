"""
MLP Serial Implementation — Mini-Batch SGD on Fashion MNIST
============================================================
Cahier des charges respecté :
  - Preprocessing FROM SCRATCH sans NumPy (normalize, shuffle, one_hot, make_batches)
  - Forward pass, backprop, SGD : NumPy autorisé (calculs matriciels ≠ preprocessing)
  - Version série pure : aucune construction parallèle
  - Mesure du temps d'exécution
"""

import os, time, random, json
import numpy as np

# ════════════════════════════════════════════════════════
# 1. PREPROCESSING FROM SCRATCH  (interdit NumPy ici)
# ════════════════════════════════════════════════════════

def normalize(X_raw):
    """
    Normalise pixels [0,255] → [0.0,1.0] FROM SCRATCH.
    X_raw : liste Python de listes d'entiers (n × 784)
    Retourne : np.ndarray float32 prêt pour le calcul
    """
    n       = len(X_raw)
    n_feat  = len(X_raw[0])
    result  = [[pixel / 255.0 for pixel in row] for row in X_raw]
    return np.array(result, dtype=np.float32)


def one_hot(labels_list, num_classes=10):
    """
    Encode labels en one-hot FROM SCRATCH.
    labels_list : liste d'entiers
    Retourne    : np.ndarray float32 (n × num_classes)
    """
    n      = len(labels_list)
    result = []
    for label in labels_list:
        vec       = [0.0] * num_classes
        vec[label] = 1.0
        result.append(vec)
    return np.array(result, dtype=np.float32)


def shuffle_data(X, Y, seed=42):
    """
    Fisher-Yates shuffle FROM SCRATCH (sans np.random).
    X : np.ndarray, Y : np.ndarray
    """
    random.seed(seed)
    n       = len(X)
    indices = list(range(n))
    for i in range(n - 1, 0, -1):
        j            = random.randint(0, i)
        indices[i], indices[j] = indices[j], indices[i]
    return X[indices], Y[indices]


def make_batches(X, Y, batch_size):
    """
    Découpe en mini-batches FROM SCRATCH.
    Retourne une liste de tuples (X_batch, Y_batch).
    """
    n       = len(X)
    batches = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batches.append((X[start:end], Y[start:end]))
    return batches


# ════════════════════════════════════════════════════════
# 2. FONCTIONS D'ACTIVATION  (NumPy autorisé ici)
# ════════════════════════════════════════════════════════

def relu(Z):
    return np.maximum(0.0, Z)

def relu_deriv(Z):
    return (Z > 0.0).astype(np.float32)

def softmax(Z):
    Z_s = Z - Z.max(axis=1, keepdims=True)
    E   = np.exp(Z_s)
    return E / E.sum(axis=1, keepdims=True)


# ════════════════════════════════════════════════════════
# 3. INITIALISATION DES POIDS  (Xavier)
# ════════════════════════════════════════════════════════

def init_params(input_size=784, h1=256, h2=128, nc=10, seed=42):
    rng   = np.random.default_rng(seed)
    def xavier(fan_in, fan_out):
        lim = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-lim, lim, (fan_in, fan_out)).astype(np.float32)

    return {
        'W1': xavier(input_size, h1),   # (784, 256)
        'b1': np.zeros((1, h1),  dtype=np.float32),
        'W2': xavier(h1, h2),            # (256, 128)
        'b2': np.zeros((1, h2),  dtype=np.float32),
        'W3': xavier(h2, nc),            # (128, 10)
        'b3': np.zeros((1, nc),  dtype=np.float32),
    }


# ════════════════════════════════════════════════════════
# 4. FORWARD PASS
# ════════════════════════════════════════════════════════

def forward(X, p):
    """
    X : (batch, 784)
    Retourne cache dict avec toutes les activations.
    """
    Z1 = X  @ p['W1'] + p['b1']  ;  A1 = relu(Z1)
    Z2 = A1 @ p['W2'] + p['b2']  ;  A2 = relu(Z2)
    Z3 = A2 @ p['W3'] + p['b3']  ;  A3 = softmax(Z3)
    return {'X': X, 'Z1': Z1, 'A1': A1,
                     'Z2': Z2, 'A2': A2,
                     'Z3': Z3, 'A3': A3}


# ════════════════════════════════════════════════════════
# 5. CROSS-ENTROPY LOSS
# ════════════════════════════════════════════════════════

def cross_entropy(A3, Y):
    n    = len(A3)
    loss = -np.sum(Y * np.log(A3 + 1e-12)) / n
    return float(loss)


# ════════════════════════════════════════════════════════
# 6. BACKPROPAGATION
# ════════════════════════════════════════════════════════

def backward(cache, Y, p):
    n   = len(Y)
    A3  = cache['A3']
    A2  = cache['A2'];  Z2 = cache['Z2']
    A1  = cache['A1'];  Z1 = cache['Z1']
    X   = cache['X']

    dZ3 = (A3 - Y) / n                          # (batch, 10)

    dW3 = A2.T @ dZ3                             # (128, 10)
    db3 = dZ3.sum(axis=0, keepdims=True)         # (1, 10)

    dA2 = dZ3 @ p['W3'].T                        # (batch, 128)
    dZ2 = dA2 * relu_deriv(Z2)

    dW2 = A1.T @ dZ2                             # (256, 128)
    db2 = dZ2.sum(axis=0, keepdims=True)

    dA1 = dZ2 @ p['W2'].T                        # (batch, 256)
    dZ1 = dA1 * relu_deriv(Z1)

    dW1 = X.T @ dZ1                              # (784, 256)
    db1 = dZ1.sum(axis=0, keepdims=True)

    return {'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2,
            'dW3': dW3, 'db3': db3}


# ════════════════════════════════════════════════════════
# 7. MISE À JOUR SGD
# ════════════════════════════════════════════════════════

def sgd_update(p, grads, lr):
    for key in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
        p[key] -= lr * grads['d' + key]


# ════════════════════════════════════════════════════════
# 8. ÉVALUATION
# ════════════════════════════════════════════════════════

def evaluate(X, y_labels, p):
    cache = forward(X, p)
    preds = np.argmax(cache['A3'], axis=1)
    return float(np.mean(preds == y_labels))


# ════════════════════════════════════════════════════════
# 9. BOUCLE D'ENTRAÎNEMENT
# ════════════════════════════════════════════════════════

def train(X_train, y_train, X_test, y_test,
          h1=256, h2=128, epochs=15, batch_size=64, lr=0.01):

    nc = 10
    print(f"\nArchitecture : {X_train.shape[1]} → {h1} → {h2} → {nc}")
    print(f"Epochs: {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    p        = init_params(X_train.shape[1], h1, h2, nc, seed=42)
    Y_onehot = one_hot(y_train.tolist(), nc)     # ← from scratch
    results  = []
    t_total  = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()

        # Shuffle from scratch
        X_s, Y_s = shuffle_data(X_train, Y_onehot, seed=epoch)
        batches   = make_batches(X_s, Y_s, batch_size)  # ← from scratch

        total_loss = 0.0
        for Xb, Yb in batches:
            cache      = forward(Xb, p)
            total_loss += cross_entropy(cache['A3'], Yb) * len(Xb)
            grads      = backward(cache, Yb, p)
            sgd_update(p, grads, lr)

        epoch_time = time.time() - t_epoch
        avg_loss   = total_loss / len(X_train)
        acc        = evaluate(X_test, y_test, p)

        results.append({'epoch': epoch, 'loss': round(avg_loss, 6),
                        'accuracy': round(acc, 6), 'time_s': round(epoch_time, 3)})
        print(f"  Epoch {epoch:2d}/{epochs}  |  Loss: {avg_loss:.4f}  |  "
              f"Acc: {acc*100:.2f}%  |  Time: {epoch_time:.1f}s")

    total_time = time.time() - t_total
    final_acc  = evaluate(X_test, y_test, p)

    print(f"\n{'='*55}")
    print(f"  Temps total  : {total_time:.2f}s")
    print(f"  Accuracy     : {final_acc*100:.2f}%")
    print(f"{'='*55}")
    return p, results, total_time, final_acc


# ════════════════════════════════════════════════════════
# 10. MAIN
# ════════════════════════════════════════════════════════

def main():
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    print("Chargement des données ...")
    X_raw   = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy')).astype(np.int64)
    Xt_raw  = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_test  = np.load(os.path.join(DATA_DIR, 'y_test.npy')).astype(np.int64)

    # ── Preprocessing FROM SCRATCH ──
    print("Preprocessing (normalisation from scratch) ...")
    t0      = time.time()
    X_train = normalize(X_raw.tolist())    # boucles Python pures
    X_test  = normalize(Xt_raw.tolist())
    preproc = time.time() - t0
    print(f"  Terminé en {preproc:.2f}s\n")

    # ── Entraînement ──
    p, results, train_time, final_acc = train(
        X_train, y_train, X_test, y_test,
        h1=256, h2=128, epochs=15, batch_size=64, lr=0.01
    )

    # ── Sauvegarde ──
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {
        'implementation'      : 'serial',
        'preprocessing_time_s': round(preproc, 4),
        'training_time_s'     : round(train_time, 4),
        'final_accuracy'      : round(final_acc, 6),
        'epochs'              : len(results),
        'per_epoch'           : results
    }
    out = os.path.join(RESULTS_DIR, 'serial_results.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Résultats → {out}")

if __name__ == "__main__":
    main()