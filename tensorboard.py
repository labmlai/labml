#!/usr/bin/env python

import argparse
import os

import lab.lab_utils as utils
from lab import colors
from lab.lab import Lab
from lab import logger


def main():
    lab = Lab(os.getcwd())
    parser = argparse.ArgumentParser(description='Run TensorBoard')
    parser.add_argument("-l",
                        action='store_true',
                        dest='list',
                        help='List all available experiments')
    parser.add_argument('-e',
                        required=False,
                        type=str,
                        nargs='+',
                        dest='experiments',
                        help='List of experiments')

    args = parser.parse_args()

    if args.list:
        utils.list_experiments(lab, logger)
    elif args.experiments:
        # List out the experiments.
        # This will fail if experiments are missing.
        runs = utils.get_last_trials(lab, args.experiments)
        utils.list_trials(runs, logger)

        # Invoke Tensorboard
        cmd = utils.get_tensorboard_cmd(lab, args.experiments)
        logger.log("Starting TensorBoard", color=colors.Style.bold)
        os.system(cmd)
    else:
        parser.print_usage()


if __name__ == "__main__":
    main()
