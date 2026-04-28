"""
Fashion MNIST Downloader — version robuste
Essaie plusieurs sources jusqu'à ce qu'une fonctionne.
"""
import urllib.request
import gzip
import os
import struct
import numpy as np

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

SOURCES = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/{filename}",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/{filename}",
    "http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/{filename}",
]

# compat alias
BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def download_file(filename):
    url = BASE_URL + filename
    dest = os.path.join(SAVE_DIR, filename)
    if os.path.exists(dest):
        print(f"  [OK] {filename} déjà téléchargé.")
        return dest
    print(f"  Téléchargement de {filename} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  [OK] {filename} téléchargé.")
    return dest


def load_images(filepath):
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols)


def load_labels(filepath):
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def main():
    print("=== Téléchargement de Fashion MNIST ===\n")

    # Télécharger les 4 fichiers
    paths = {}
    for key, filename in FILES.items():
        paths[key] = download_file(filename)

    print("\nChargement et sauvegarde en .npy ...")

    X_train = load_images(paths["train_images"])
    y_train = load_labels(paths["train_labels"])
    X_test  = load_images(paths["test_images"])
    y_test  = load_labels(paths["test_labels"])

    np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(SAVE_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(SAVE_DIR, "y_test.npy"),  y_test)

    print(f"\n[OK] Dataset sauvegardé dans {SAVE_DIR}")
    print(f"  X_train : {X_train.shape}  (dtype: {X_train.dtype})")
    print(f"  y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_test  : {y_test.shape}")
    print(f"\nClasses : {CLASS_NAMES}")


if __name__ == "__main__":
    main()