#!/usr/bin/env python3
#  coding: utf-8

import argparse
from kaldiio import ReadHelper
import numpy as np
import os
from os.path import join
import sys

from hdf5_utils import write_hdf5

def get_parser():
    parser = argparse.ArgumentParser(
        description='Convet kaldi-style features to h5 files for WaveNet vocoder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scp_file', type=str, help='scp file')
    parser.add_argument('--out_dir', type=str, help='output directory')
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    os.makedirs(args.out_dir, exist_ok=True)
    with ReadHelper(f"scp:{args.scp_file}") as f:
        for utt_id, arr in f:
            out_path = join(args.out_dir, "{}-feats.h5".format(utt_id))
            write_hdf5(out_path, "/melspc", np.float32(arr))
    sys.exit(0)
