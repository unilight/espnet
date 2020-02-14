#!/usr/bin/env python3
# encoding: utf-8

import argparse
from io import open
import json
import logging
import sys

# I wonder if I can successfully import this line...
from espnet.utils.cli_utils import get_commandline_args


def get_parser():
    parser = argparse.ArgumentParser(
        description='Make json file for mel autoencoder training.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_json', type=str,
                        help='Json file')
    parser.add_argument('--output_json', type=str,
                        help='Json file')
    parser.add_argument('--num_utts', default=-1, type=int,
                        help='Number of utterances (take from head)')
    parser.add_argument('--verbose', '-V', default=1, type=int,
                        help='Verbose option')
    parser.add_argument('--out', '-O', type=str,
                        help='The output filename. '
                             'If omitted, then output to sys.stdout')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    with open(args.input_json, 'rb') as f:
        input_json = json.load(f)['utts']
    with open(args.output_json, 'rb') as f:
        output_json = json.load(f)['utts']

    count = 0
    data = {"utts": {}}
    for k, v in input_json.items():
        entry = {"input": input_json[k]['input'],
                 "output": output_json[k]['input'],
                 "utt2spk": input_json[k]['utt2spk']
                 }
        data["utts"][k] = entry
        count += 1
        if args.num_utts > 0 and count >= args.num_utts:
            break

    if args.out is None:
        out = sys.stdout
    else:
        out = open(args.out, 'w', encoding='utf-8')

    json.dump(data, out,
              indent=4, ensure_ascii=False,
              separators=(',', ': '),
              )
