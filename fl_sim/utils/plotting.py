import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils import get_best_runs
import numpy as np
import os

sns.set(
    style="whitegrid", font_scale=1.2, context="talk",
    palette=sns.color_palette("bright"), color_codes=False)
# consider using TrueType fonts if submitting to a conference
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['figure.figsize'] = (8, 6)

PLOT_PATH = '../plots/'
MARKERS = ['o', 'v', 's', 'P', 'p', '*', 'H', 'X', 'D',
           'o', 'v', 's', 'P', 'p', '*', 'H', 'X', 'D']


def plot(exps, kind, log_scale=True, legend=None, file=None,
         x_label='comm. rounds', y_label=None, last=True, best=True,
         min_value=0., title=None, bottom=None):
    fig, ax = plt.subplots()

    for i, exp in enumerate(exps):
        label = 'none' if legend is None else legend[i]
        runs = get_best_runs(exp, last=last, best=best)
        plot_mean_std(ax, runs, kind, i, label, min_value)

    if log_scale:
        ax.set_yscale('log')
    if legend is not None:
        ax.legend()

    ax.set_xlabel(x_label)
    if y_label is None:
        ax.set_ylabel(kind)
    else:
        ax.set_ylabel(y_label)

    if title is not None:
        ax.set_title(title)

    ax.set_ylim(bottom=bottom)

    fig.tight_layout()
    if file is not None:
        os.makedirs(PLOT_PATH, exist_ok=True)
        plt.savefig(PLOT_PATH + file + '.pdf')

    plt.show()


def plot_mean_std(ax, runs, metric_name, i, label, min_value=0.):
    max_len = np.max([len(run[metric_name]) for run in runs])
    runs = [run for run in runs if len(run[metric_name]) == max_len]
    quant = np.array([run[metric_name] for run in runs])
    axis = 0

    mean = np.nanmean(quant, axis=axis) - min_value
    std = np.nanstd(quant, axis=axis)

    print(f'Output value: {mean[-1]} +- {std[-1]}')

    # x = np.arange(1, len(mean) + 1)
    preffix = metric_name.split('_')[0]
    x = np.array(runs[0][preffix + '_round'])
    ax.plot(x, mean, marker=MARKERS[i], markersize=12, label=label,
            markevery=np.max([len(x), 10]) // 10)
    ax.fill_between(x, mean + std, mean - std, alpha=0.4)
