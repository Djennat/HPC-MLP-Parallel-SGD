"""
plot_results.py
===============
Génère les graphiques pour le rapport PDF :
  1. Speedup vs nombre de threads
  2. Efficacité parallèle
  3. Accuracy série vs parallèle
  4. Temps d'entraînement par epoch
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR     = RESULTS_DIR


def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"[WARN] {filename} introuvable — skip")
        return None
    with open(path) as f:
        return json.load(f)


def plot_all():
    serial   = load_json('serial_results.json')
    parallel = load_json('parallel_results.json')

    if parallel is None:
        print("parallel_results.json manquant — arrêt.")
        return

    threads    = [r['n_threads']        for r in parallel]
    times      = [r['training_time_s']  for r in parallel]
    accuracies = [r['final_accuracy']   for r in parallel]
    t1         = times[0]  # temps avec 1 thread

    speedups    = [t1 / t for t in times]
    efficiencies = [s / n for s, n in zip(speedups, threads)]
    ideal        = threads  # speedup idéal = n_threads

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # ── Graphique 1 : Speedup ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(threads, ideal,    'k--', linewidth=1, label='Idéal (linéaire)', alpha=0.5)
    ax1.plot(threads, speedups, 'o-',  color='#2196F3', linewidth=2,
             markersize=8, label='Speedup mesuré')
    for x, y in zip(threads, speedups):
        ax1.annotate(f'{y:.2f}x', (x, y), textcoords="offset points",
                     xytext=(6, 4), fontsize=9)
    ax1.set_xlabel('Nombre de threads')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Speedup parallèle (OpenMP)')
    ax1.legend(fontsize=9)
    ax1.set_xticks(threads)
    ax1.grid(True, alpha=0.3)

    # ── Graphique 2 : Efficacité ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(threads, [1.0]*len(threads), 'k--', linewidth=1, alpha=0.5, label='Idéal (100%)')
    ax2.plot(threads, efficiencies, 's-', color='#4CAF50', linewidth=2,
             markersize=8, label='Efficacité mesurée')
    for x, y in zip(threads, efficiencies):
        ax2.annotate(f'{y*100:.0f}%', (x, y), textcoords="offset points",
                     xytext=(6, 4), fontsize=9)
    ax2.set_xlabel('Nombre de threads')
    ax2.set_ylabel('Efficacité (Speedup / n_threads)')
    ax2.set_title('Efficacité parallèle')
    ax2.legend(fontsize=9)
    ax2.set_xticks(threads)
    ax2.set_ylim(0, 1.2)
    ax2.grid(True, alpha=0.3)

    # ── Graphique 3 : Accuracy comparison ──
    ax3 = fig.add_subplot(gs[1, 0])
    labels   = ['Série'] + [f'{n}T' for n in threads]
    accs_all = []
    if serial:
        accs_all.append(serial['final_accuracy'] * 100)
    else:
        accs_all.append(0)
    accs_all += [a * 100 for a in accuracies]

    colors = ['#FF9800'] + ['#2196F3'] * len(threads)
    bars   = ax3.bar(labels, accs_all, color=colors, edgecolor='white', width=0.6)
    for bar, val in zip(bars, accs_all):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
    ax3.set_xlabel('Implémentation')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Accuracy finale : Série vs Parallèle')
    ax3.set_ylim(min(accs_all) - 2, max(accs_all) + 3)
    ax3.grid(True, alpha=0.3, axis='y')

    # ── Graphique 4 : Temps par epoch ──
    ax4 = fig.add_subplot(gs[1, 1])
    for r in parallel:
        ep_times = [e['time_s'] for e in r['per_epoch']]
        epochs   = [e['epoch']  for e in r['per_epoch']]
        ax4.plot(epochs, ep_times, 'o-', linewidth=1.5, markersize=5,
                 label=f"{r['n_threads']} thread(s)")
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Temps (s)')
    ax4.set_title('Temps par epoch selon le nombre de threads')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('MLP + Parallel Mini-Batch SGD — Fashion MNIST\n'
                 'Analyse des performances OpenMP',
                 fontsize=13, fontweight='bold', y=1.01)

    out_path = os.path.join(OUT_DIR, 'performance_analysis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Graphique sauvegardé : {out_path}")

    # ── Tableau texte ──
    print("\n" + "="*65)
    print(f"  {'Config':>12}  {'Temps (s)':>12}  {'Speedup':>10}  {'Efficacité':>12}  {'Accuracy':>10}")
    print(f"  {'-'*61}")
    if serial:
        print(f"  {'Série':>12}  {serial['training_time_s']:>12.2f}  "
              f"{'1.00x':>10}  {'100%':>12}  {serial['final_accuracy']*100:>9.2f}%")
    for r, sp, eff in zip(parallel, speedups, efficiencies):
        print(f"  {str(r['n_threads'])+' thread(s)':>12}  {r['training_time_s']:>12.2f}  "
              f"{sp:>9.2f}x  {eff*100:>11.1f}%  {r['final_accuracy']*100:>9.2f}%")
    print("="*65)


if __name__ == "__main__":
    plot_all()