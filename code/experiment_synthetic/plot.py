# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import math
import sys

from matplotlib.patches import Patch


def parse_title(title):
    result = ""
    fields = title.split("_")

    if fields[1].split("=")[1] == "1":
        result += "P"
    else:
        result += "F"

    if fields[2].split("=")[1] == "1":
        result += "E"
    else:
        result += "O"

    if fields[3].split("=")[1] == "1":
        result += "S"
    else:
        result += "U"

    return result


def plot_bars(results, category, which, sep=1.1):
    models = list(results[list(results.keys())[0]].keys())

    if "SEM" in models:
        models.remove("SEM")
    models.sort()

    setups = list(results.keys())
    setups.sort()

    if which == "causal":
        hatch = None
        offset = 0
        idx = 0
    else:
        hatch = "//"
        offset = 4
        idx = 1

    counter = 1
    for s, setup in enumerate(setups):
        title = parse_title(setup)

        if category not in title:
            continue

        boxes = []
        boxes_means = []
        boxes_colors = []
        boxes_vars = []
        ax = plt.subplot(2, 4, counter + offset)
        counter += 1

        for m, model in enumerate(models):
            boxes.append(np.array(results[setup][model])[:, idx])
            boxes_means.append(
                np.mean(np.array(results[setup][model])[:, idx]))
            boxes_vars.append(np.std(np.array(results[setup][model])[:, idx]))
            boxes_colors.append("C" + str(m))

        plt.bar([0, 1, 2],
                boxes_means,
                yerr=np.array(boxes_vars),
                color=boxes_colors,
                hatch=hatch,
                alpha=0.7,
                log=True)

        if which == "causal":
            plt.xticks([1], [title])
        else:
            ax.xaxis.set_ticks_position('top')
            plt.xticks([1], [""])

        if (counter + offset) == 2 or (counter + offset) == 6:
            if which == "causal":
                plt.ylabel("causal error")
            else:
                plt.ylabel("non-causal error")

        if title == "PES" and which != "causal":
            legends = []
            for m, model in enumerate(models):
                legends.append(
                    Patch(facecolor="C" + str(m), alpha=0.7, label=model))
            plt.legend(handles=legends, loc="lower center")

        if title == "POU" and which != "causal":
            plt.minorticks_off()
            ax.set_yticks([0.1, 0.01])


def plot_experiment(all_solutions, category, fname):
    plt.rcParams["font.family"] = "serif"
    plt.rc('text', usetex=True)
    plt.rc('font', size=10)

    results = {}

    for line in all_solutions:
        words = line.split(" ")
        setup = str(words[0])
        model = str(words[1])
        err_causal = float(words[-2])
        err_noncausal = float(words[-1])

        if setup not in results:
            results[setup] = {}

        if model not in results[setup]:
            results[setup][model] = []

        results[setup][model].append([err_causal, err_noncausal])

    plt.figure(figsize=(7, 2))
    plot_bars(results, category, "causal")
    plot_bars(results, category, "noncausal")
    plt.tight_layout(0, 0, 0.5)

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        fname = "synthetic_results.pt"
    else:
        fname = sys.argv[1]
    lines = torch.load(fname)
    plot_experiment(lines, "F", "results_f.pdf")
    plot_experiment(lines, "P", "results_p.pdf")
