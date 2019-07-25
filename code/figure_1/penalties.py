# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np


def ls(x, y, reg=.1):
    return Ridge(alpha=reg, fit_intercept=False).fit(x, y).coef_


def sample(n=100000, e=1):
    x = np.random.randn(n, 1) * e
    y = x + np.random.randn(n, 1) * e
    z = y + np.random.randn(n, 1)
    return np.hstack((x, z)), y


def penalty_ls(x1, y1, x2, y2, t=1, reg=.1):
    phi = np.diag([1, t])
    w = np.array([1, 0]).reshape(1, 2)
    p1 = np.linalg.norm(ls(x1 @ phi, y1, reg) - w) 
    p2 = np.linalg.norm(ls(x2 @ phi, y2, reg) - w)
    return (p1 + p2) / 2


def penalty_g(x1, y1, x2, y2, t=1):
    phi = np.diag([1, t])
    w = np.array([1, 0]).reshape(2, 1)
    p1 = (phi.T @ x1.T @ x1 @ phi @ w - phi.T @ x1.T @ y1) / x1.shape[0]
    p2 = (phi.T @ x2.T @ x2 @ phi @ w - phi.T @ x2.T @ y2) / x2.shape[0]
    return np.linalg.norm(p1) ** 2 + np.linalg.norm(p2) ** 2


if __name__ == "__main__":
    x1, y1 = sample(e=1)
    x2, y2 = sample(e=2)

    plot_x = np.linspace(-1, 1, 100 + 1)
    plot_y_ls = []
    plot_y_ls_reg = []
    plot_y_1 = []

    for t in plot_x:
        plot_y_ls.append(penalty_ls(x1, y1, x2, y2, t))
        plot_y_ls_reg.append(penalty_ls(x1, y1, x2, y2, t, reg=1000))
        plot_y_1.append(penalty_g(x1, y1, x2, y2, t))

    plt.rcParams.update({'text.latex.preamble' : [r'\usepackage{amsmath, amsfonts}']})
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc('text', usetex=True)
    plt.rc('font', size=12)

    plt.figure(figsize=(8, 4))
    plt.plot(plot_x, plot_y_ls, lw=2, label=r'$\mathbb{D}_{\text{dist}}((1, 0), \Phi, e)$')
    plt.plot(plot_x, plot_y_ls_reg, ls="--", lw=2, label=r'$\mathbb{D}_{\text{dist}}$ (heavy regularization)')
    plt.plot(plot_x, plot_y_1, '.', lw=2, label=r'$\mathbb{D}_{\text{lin}}((1, 0), \Phi, e)$')
    plt.ylim(-1, 12)
    plt.xlabel(
        r'$c$, the weight of $\Phi$ on the input with varying correlation', labelpad=10)
    plt.ylabel(r'invariance penalty')
    plt.tight_layout(0, 0, 0)
    plt.legend(prop={'size': 11}, loc="upper right")
    plt.savefig("different_penalties.pdf")
