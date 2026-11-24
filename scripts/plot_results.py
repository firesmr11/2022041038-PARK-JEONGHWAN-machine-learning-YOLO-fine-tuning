"""Rebuild report figures from the checked-in measurements (no model training)."""
import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=Path('results'))
    a = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a.results.joinpath('figures').mkdir(parents=True, exist_ok=True)
    with (a.results/'test_comparison.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {'Colab T4': '#2269A8', 'Local RTX 4060': '#CA681A'}
    for env in ['Colab T4', 'Local RTX 4060']:
        for size in [196, 414]:
            items = sorted([r for r in rows if r['environment'] == env and int(r['train_images']) == size], key=lambda r:int(r['epochs']))
            ax.plot([int(r['epochs']) for r in items], [float(r['mAP50_95_percent']) for r in items],
                    marker='o', color=colors[env], linestyle='-' if size == 196 else '--', label=f'{env}, train={size}')
    ax.set(xlabel='Training budget (epochs)', ylabel='Test mAP50-95 (%)', xticks=[25,50,75], ylim=(94.5,100),
           title='YOLOv8n: fixed test set (19 images, 26 objects)')
    ax.grid(axis='y', alpha=.25)
    ax.legend(fontsize=9, loc='lower right')
    fig.tight_layout()
    fig.savefig(a.results/'figures/test_map_comparison.png', dpi=200)
    plt.close(fig)
    with (a.results/'local_training_times.csv').open(encoding='utf-8-sig') as f:
        times = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(8,4.5))
    for size, color in [(196,'#2269A8'),(414,'#CA681A')]:
        items = [r for r in times if int(r['train_images'])==size]
        ax.plot([int(r['epochs']) for r in items], [float(r['results_csv_final_time_seconds']) for r in items], marker='o',label=f'train={size}',color=color)
    ax.set(xlabel='Training budget (epochs)', ylabel='Training time (s)', xticks=[25,50,75], ylim=(0,None),
           title='Local RTX 4060: final time recorded in results.csv')
    ax.grid(axis='y',alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.results/'figures/local_training_time.png',dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(11,7.5))
    for key, color in [('local_196_75','#CA681A'),('colab_414_50','#2269A8')]:
        path = a.results/'history'/f'{key}_checkpoint_history.csv'
        with path.open(encoding='utf-8-sig') as f:
            history = list(csv.DictReader(f))
        if not history: continue
        x = [int(float(r['epoch'])) for r in history]
        for ax, column, title in zip(axes.flat,
              ['train/box_loss','train/cls_loss','val/box_loss','metrics/mAP50-95(B)'],
              ['Training box loss','Training classification loss','Validation box loss','Validation mAP50-95']):
            if column in history[0]:
                ax.plot(x,[float(r[column]) for r in history],label=key,color=color)
            ax.set(title=title,xlabel='Epoch recorded in checkpoint')
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Recovered training histories stored in the selected checkpoints',fontsize=12)
    fig.tight_layout()
    fig.savefig(a.results/'figures/checkpoint_history.png',dpi=200)
    plt.close(fig)
    print('Saved figures to',a.results/'figures')


if __name__ == '__main__':
    main()
