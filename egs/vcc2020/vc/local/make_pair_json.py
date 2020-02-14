#!/usr/bin/env python3
# encoding: utf-8

import argparse
import codecs
from distutils.util import strtobool
from io import open
import json
import logging
import sys

# I wonder if I can successfully import this line...
from espnet.utils.cli_utils import get_commandline_args

def get_parser():
    parser = argparse.ArgumentParser(
        description='Merge source and target data.json files into one json file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src-json', type=str,
                        help='Json file for the source speaker')
    parser.add_argument('--trg-json', type=str,
                        help='Json file for the target speaker')
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

    with open(args.src_json, 'rb') as f:
        src_json = json.load(f)['utts']
    with open(args.trg_json, 'rb') as f:
        trg_json = json.load(f)['utts']

    # get source and target speaker
    _ = list(src_json.keys())[0].split('_')
    srcspk = _[0]
    _ = list(trg_json.keys())[0].split('_')
    trgspk = _[0]

    count = 0
    data = {"utts" : {} }
    for k, v in src_json.items():

        inp = src_json[k]['input']
        inp[0]['name'] = 'input1'
        out = src_json[k]['input']
        out[0]['name'] = 'target1'

        entry = {"input" : inp,
                 "output" : out,
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
