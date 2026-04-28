# MLP + Parallel Mini-Batch SGD — Fashion MNIST
## HPC Mini Project 2025–2026

---

## Prérequis (GitHub Codespace)

```bash
# Vérifier que GCC et OpenMP sont disponibles
gcc --version
echo '#include <omp.h>' | gcc -fopenmp -x c - -o /dev/null && echo "OpenMP OK"

# Installer les dépendances Python
pip install numpy matplotlib
```

---

## Structure du projet

```
mlp_project/
├── data/
│   ├── download.py          ← Téléchargement de Fashion MNIST
│   ├── X_train.npy          ← Généré après download
│   ├── y_train.npy
│   ├── X_test.npy
│   └── y_test.npy
├── serial/
│   └── mlp_serial.py        ← Implémentation série (baseline)
├── parallel/
│   ├── mlp_parallel.c       ← Noyau C/OpenMP
│   ├── mlp_parallel.so      ← Généré après compilation
│   └── mlp_parallel.py      ← Wrapper Python (ctypes)
├── results/
│   ├── plot_results.py      ← Génération des graphiques
│   ├── serial_results.json  ← Généré après run série
│   └── parallel_results.json← Généré après run parallèle
├── bonus/                   ← GPU (optionnel +15%)
└── README.md
```

---

## Étape 1 — Télécharger les données

```bash
cd data/
python download.py
cd ..
```

---

## Étape 2 — Exécuter la version série (baseline)

```bash
cd serial/
python mlp_serial.py
cd ..
```

---

## Étape 3 — Compiler la librairie C/OpenMP

```bash
cd parallel/
gcc -O2 -fopenmp -shared -fPIC -o mlp_parallel.so mlp_parallel.c -lm
echo "Compilation OK"
cd ..
```

---

## Étape 4 — Exécuter la version parallèle

```bash
cd parallel/
python mlp_parallel.py
cd ..
```

> Le script teste automatiquement 1, 2, 4 et 8 threads.

---

## Étape 5 — Générer les graphiques

```bash
cd results/
python plot_results.py
cd ..
```

---

## Paramètres principaux

| Paramètre      | Valeur par défaut |
|----------------|-------------------|
| Architecture   | 784 → 256 → 128 → 10 |
| Epochs         | 5 (ajustable)     |
| Batch size     | 64                |
| Learning rate  | 0.01              |
| Activation     | ReLU (hidden), Softmax (output) |
| Init poids     | Xavier uniform    |
| Threads testés | 1, 2, 4, 8        |

---

## Notes importantes

- Le **preprocessing** (normalisation) est implémenté **from scratch** sans NumPy dans `serial/mlp_serial.py`, et via le noyau C/OpenMP dans `parallel/mlp_parallel.c`.
- L'utilisation de NumPy est limitée au chargement des fichiers `.npy` et à l'évaluation de l'accuracy.
- Les versions série et parallèle utilisent **la même initialisation** (même seed Xavier) pour permettre une comparaison directe des résultats.