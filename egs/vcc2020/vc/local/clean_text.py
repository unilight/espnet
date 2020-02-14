#!/usr/bin/env python3

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import nltk
import os

from text.cleaners import custom_english_cleaners

try:
    # For phoneme conversion, use https://github.com/Kyubyong/g2p.
    from g2p_en import G2p
    f_g2p = G2p()
    f_g2p("")
except ImportError:
    raise ImportError("g2p_en is not installed. please run `. ./path.sh && pip install g2p_en`.")
except LookupError:
    # NOTE: we need to download dict in initial running
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download("punkt")


def g2p(text):
    """Convert grapheme to phoneme."""
    tokens = filter(lambda s: s != " ", f_g2p(text))
    return ' '.join(tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('transcription_dir', type=str, help='dir for the transcription text fileS')
    parser.add_argument('utt2spk', type=str, help='utt2spk file for the speaker')
    parser.add_argument("trans_type", type=str, default="kana",
                        choices=["char", "phn"],
                        help="Input transcription type")
    args = parser.parse_args()

    # clean every line in transcription file first
    transcription_dict = {}
    for transcription_file in os.listdir(args.transcription_dir):
        transcription_path = os.path.join(args.transcription_dir, transcription_file)
        with codecs.open(transcription_path, 'r', 'utf-8') as fid:
            for line in fid.readlines():
                segments = line.split(" ")
                id = transcription_file[0] + segments[0] # ex. E10001
                content = ' '.join(segments[1:])
                clean_content = custom_english_cleaners(content.rstrip())
                if args.trans_type == "phn":
                    text = clean_content.lower()
                    clean_content = g2p(text)

                transcription_dict[id] = clean_content
    
    # read the utt2spk file and actually write
    with codecs.open(args.utt2spk, 'r', 'utf-8') as fid:
        for line in fid.readlines():
            segments = line.split(" ")
            id = segments[0] # ex. E10001
            content = transcription_dict[id]

            print("%s %s" % (id, content))
