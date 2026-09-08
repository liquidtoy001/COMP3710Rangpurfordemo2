"""Download CIFAR-10 once, into a directory the GPU jobs then read from.

Run this on a *CPU* node, not a GPU node:

    srun --partition=cpu --time=00:20:00 --pty bash
    conda activate torch
    python prepare_data.py --data-dir $HOME/data
    exit

Roughly 170 MB. Every training job afterwards runs with ``download=False``, so
the A100 is never held idle waiting on the network.
"""

import argparse

from data import build_datasets, describe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="where to store CIFAR-10 (use $HOME/data on Rangpur so every job sees it)",
    )
    args = parser.parse_args()

    print(f"downloading CIFAR-10 into {args.data_dir} ...")
    build_datasets(args.data_dir, download=True)
    print("done.\n")
    describe(args.data_dir)


if __name__ == "__main__":
    main()
