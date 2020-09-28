# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from sem import ChainEquationModel
from models import *

import argparse
import torch
import numpy


def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.3f}".format(vi) for vi in vlist) + "]"


def errors(w, w_hat):
    w = w.view(-1)
    w_hat = w_hat.view(-1)

    i_causal = torch.where(w != 0)[0].view(-1)
    i_noncausal = torch.where(w == 0)[0].view(-1)

    if len(i_causal):
        error_causal = (w[i_causal] - w_hat[i_causal]).pow(2).mean()
        error_causal = error_causal.item()
    else:
        error_causal = 0

    if len(i_noncausal):
        error_noncausal = (w[i_noncausal] - w_hat[i_noncausal]).pow(2).mean()
        error_noncausal = error_noncausal.item()
    else:
        error_noncausal = 0

    return error_causal, error_noncausal


def run_experiment(args):
    if args["seed"] >= 0:
        torch.manual_seed(args["seed"])
        numpy.random.seed(args["seed"])
        torch.set_num_threads(1)

    if args["setup_sem"] == "chain":
        setup_str = "chain_ones={}_hidden={}_hetero={}_scramble={}".format(
            args["setup_ones"],
            args["setup_hidden"],
            args["setup_hetero"],
            args["setup_scramble"])
    elif args["setup_sem"] == "icp":
        setup_str = "sem_icp"
    else:
        raise NotImplementedError

    all_methods = {
        "ERM": EmpiricalRiskMinimizer,
        "ICP": InvariantCausalPrediction,
        "IRM": InvariantRiskMinimization
    }

    if args["methods"] == "all":
        methods = all_methods
    else:
        methods = {m: all_methods[m] for m in args["methods"].split(',')}

    all_sems = []
    all_solutions = []
    all_environments = []

    for rep_i in range(args["n_reps"]):
        if args["setup_sem"] == "chain":
            sem = ChainEquationModel(args["dim"],
                                     ones=args["setup_ones"],
                                     hidden=args["setup_hidden"],
                                     scramble=args["setup_scramble"],
                                     hetero=args["setup_hetero"])

            env_list = [float(e) for e in args["env_list"].split(",")]
            environments = [sem(args["n_samples"], e) for e in env_list]
        else:
            raise NotImplementedError

        all_sems.append(sem)
        all_environments.append(environments)

    for sem, environments in zip(all_sems, all_environments):
        sem_solution, sem_scramble = sem.solution()

        solutions = [
            "{} SEM {} {:.5f} {:.5f}".format(setup_str,
                                             pretty(sem_solution), 0, 0)
        ]

        for method_name, method_constructor in methods.items():
            method = method_constructor(environments, args)

            method_solution = sem_scramble @ method.solution()

            err_causal, err_noncausal = errors(sem_solution, method_solution)

            solutions.append("{} {} {} {:.5f} {:.5f}".format(
                setup_str,
                method_name,
                pretty(method_solution),
                err_causal,
                err_noncausal))

        all_solutions += solutions

    return all_solutions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Invariant regression')
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--n_reps', type=int, default=10)
    parser.add_argument('--skip_reps', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)  # Negative is random
    parser.add_argument('--print_vectors', type=int, default=1)
    parser.add_argument('--n_iterations', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--methods', type=str, default="ERM,ICP,IRM")
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--env_list', type=str, default=".2,2.,5.")
    parser.add_argument('--setup_sem', type=str, default="chain")
    parser.add_argument('--setup_ones', type=int, default=1)
    parser.add_argument('--setup_hidden', type=int, default=0)
    parser.add_argument('--setup_hetero', type=int, default=0)
    parser.add_argument('--setup_scramble', type=int, default=0)
    args = dict(vars(parser.parse_args()))

    all_solutions = run_experiment(args)
    print("\n".join(all_solutions))
