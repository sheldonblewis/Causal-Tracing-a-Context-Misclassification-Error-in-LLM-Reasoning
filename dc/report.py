import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

O = Path('outputs/v2')
F = Path('figures')
plt.rcParams.update({'figure.dpi': 140, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'legend.frameon': False})
CC = {'ambiguous': '#c0715f', 'decimal': '#c9a46c', 'padded': '#5b8f6e'}
SEG = {'all positions': '#1f3a5f', 'shorter number': '#c0715f', 'longer number': '#c9a46c', 'final token': '#8fb39a'}


def report():
    F.mkdir(exist_ok=True)
    base = pd.read_csv(O / 'baseline.csv')
    acc = base.groupby(['kind', 'condition'])['correct'].mean().unstack()[['ambiguous', 'decimal', 'padded']]
    ld = base.groupby(['kind', 'condition'])['logit_diff'].mean().unstack()[['ambiguous', 'decimal', 'padded']]
    lines = ['| pairs | ambiguous | "treat as decimals" | zero-padded |', '|---|---|---|---|']
    for k in ('trap', 'control'):
        n = int((base.kind == k).sum() / 3)
        lines.append(f'| {k} (n={n}) | ' + ' | '.join(f'{acc.loc[k, c]:.0%} (ld {ld.loc[k, c]:+.2f})' for c in acc.columns) + ' |')
    lines.append(f"\nmean probability mass on the two letter tokens: {base.p_letters.mean():.2f}")

    fig, ax = plt.subplots(figsize=(5.5, 3))
    x = np.arange(2)
    for j, c in enumerate(acc.columns):
        ax.bar(x + (j - 1) * 0.26, acc.loc[['trap', 'control'], c], 0.26, color=CC[c], label=c)
    ax.set_xticks(x, ['trap (7.11 vs 7.8)', 'control (7.81 vs 7.2)'])
    ax.axhline(0.5, color='k', lw=0.6, ls='--')
    ax.set_ylim(0, 1)
    ax.set_ylabel('accuracy')
    ax.legend(fontsize=7)
    ax.set_title('phi-3-mini: accuracy by pair type and prompt')
    fig.tight_layout()
    fig.savefig(F / 'accuracy.png')
    plt.close(fig)

    lens = pd.DataFrame(json.loads((O / 'logit_lens.json').read_text()))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    for ax, kind in zip(axes, ('trap', 'control')):
        for c in CC:
            m = np.array(lens[(lens.kind == kind) & (lens.condition == c)]['lens'].tolist())
            ax.plot(np.arange(m.shape[1]), m.mean(0), color=CC[c], lw=1.6, label=c)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_title(f'logit lens, {kind} pairs')
        ax.set_xlabel('layer')
    axes[0].set_ylabel('logit(correct) - logit(wrong)')
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(F / 'logit_lens.png')
    plt.close(fig)

    if (O / 'patching.csv').exists():
        p = pd.read_csv(O / 'patching.csv')
        g = p.groupby(['segment', 'layer'])['effect']
        mean, sem = g.mean().unstack(0), g.sem().unstack(0)
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        for s in SEG:
            if s in mean:
                ax.plot(mean.index, mean[s], color=SEG[s], lw=1.7, label=s)
                ax.fill_between(mean.index, mean[s] - sem[s], mean[s] + sem[s], color=SEG[s], alpha=0.15)
        ax.axhline(0, color='k', lw=0.6)
        ax.axhline(1, color='k', lw=0.6, ls='--')
        ax.set_xlabel('layer patched (residual stream after it)')
        ax.set_ylabel('share of the padding fix restored')
        ax.set_title(f'patching padded -> unpadded, {p.pair.nunique()} trap pairs')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(F / 'patching.png')
        plt.close(fig)
        peak = mean.idxmax()
        lines += ['', '| segment patched | peak layer | peak effect | effect at last layer |', '|---|---|---|---|']
        for s in SEG:
            if s in mean:
                lines.append(f'| {s} | {peak[s]} | {mean[s].max():.2f} | {mean[s].iloc[-1]:.2f} |')
    (O / 'table.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
