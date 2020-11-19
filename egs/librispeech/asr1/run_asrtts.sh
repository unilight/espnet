#!/bin/bash

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1      # verbose option
seed=1         # random seed number
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
asr_train_config=conf/tuning/train_pytorch_conformer_large.yaml
tts_train_config=conf/train_pytorch_transformer+spkemb.yaml
lm_config=conf/lm.yaml
decode_asr_config=conf/decode_ctc.yaml

# rnnlm related
lm_resume= # specify a snapshot file to resume LM training
lmtag=TFLM     # tag for managing LMs
lmexpdir=exp/train_rnnlm_pytorch_lm_transformer_cosine_batchsize32_lr1e-4_layer16_unigram5000_ngpu4

# decoding parameter
recog_model=model.loss.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=5                  # the number of ASR models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
lm_n_average=6               # the number of languge models to be averaged
use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.

datadir=/nas01/internal/wenchin-h/VC/Experiments/espnet/egs/librispeech/asr1/downloads

# base url for downloads.
data_url=www.openslr.org/resources/12

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram

# training related
asr_train=true
asr_decode=true
tts_train=true
tts_decode=true
asrtts_train=true
asrtts_decode=true
unpair=dualp
policy_gradient=true
use_rnnlm=false
rnnlm_loss=none

# exp tag
asrpt_tag=""
ttspt_tag=""
tag=""

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

pretrain_set=train_clean_100
train_set=train_clean_100
train_dev=dev
#recog_set="test_clean test_other dev_clean dev_other"
recog_set="test_clean test_other dev"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${datadir}/LibriSpeech/${part} data/${part//-/_}
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    utils/combine_data.sh --extra_files utt2num_frames data/${train_dev}_org data/dev_clean data/dev_other

    # remove utt having more than 3000 frames
    # remove utt having more than 400 characters
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_dev}_org data/${train_dev}

    # also remove for 100, 360 and 500
    for _set in train_clean_100 train_clean_360 train_other_500; do
        utils/copy_data_dir.sh data/${_set} data/${_set}_org
        remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${_set}_org data/${_set}
    done

    # compute global CMVN
    compute-cmvn-stats scp:data/train_clean_100/feats.scp data/train_clean_100/cmvn.ark

    # dump features for training
    for _set in train_clean_100 train_clean_360 train_other_500 dev test_clean test_other; do
        dump.sh --cmd "$train_cmd" --nj 80 --do_delta ${do_delta} \
            data/${_set}/feats.scp data/train_clean_100/cmvn.ark exp/dump_feats/train $dumpdir/$_set/delta${do_delta}
    done
fi

dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cut -f 2- -d" " data/${train_set}/text > data/lang_char/input.txt
    spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    for _set in train_clean_100 train_clean_360 train_other_500 dev test_clean test_other; do
        data2json.sh --feat $dumpdir/$_set/delta${do_delta}/feats.scp --bpecode ${bpemodel}.model \
            data/${_set} ${dict} > $dumpdir/$_set/delta${do_delta}/data_${bpemode}${nbpe}.json
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    nj=80

    echo "stage 3: x-vector extraction"
    # Make MFCCs and compute the energy-based VAD for each dataset
    mfccdir=mfcc
    vaddir=mfcc
    nnet_dir=exp/xvector_nnet_1a
    for name in train_clean_100 train_clean_360 train_other_500 dev test_clean test_other; do
        break
        if [ ! -s data/${name}_mfcc_16k/feats.scp ]; then
            utils/copy_data_dir.sh data/${name} data/${name}_mfcc_16k
            utils/data/resample_data_dir.sh 16000 data/${name}_mfcc_16k
            steps/make_mfcc.sh \
                --write-utt2num-frames true \
                --mfcc-config conf/mfcc.conf \
                --nj ${nj} --cmd "$train_cmd" \
                data/${name}_mfcc_16k exp/make_mfcc_16k ${mfccdir}
        fi
        utils/fix_data_dir.sh data/${name}_mfcc_16k
        sid/compute_vad_decision.sh --nj ${nj} --cmd "$train_cmd" \
            data/${name}_mfcc_16k exp/make_vad ${vaddir}
        utils/fix_data_dir.sh data/${name}_mfcc_16k
    done

    # Check pretrained model existence
    nnet_dir=exp/xvector_nnet_1a
    if [ ! -e ${nnet_dir} ]; then
        echo "X-vector model does not exist. Download pre-trained model."
        wget http://kaldi-asr.org/models/8/0008_sitw_v2_1a.tar.gz
        tar xvf 0008_sitw_v2_1a.tar.gz
        mv 0008_sitw_v2_1a/exp/xvector_nnet_1a exp
        rm -rf 0008_sitw_v2_1a.tar.gz 0008_sitw_v2_1a
    fi

    # Extract x-vector
    for name in train_clean_100 train_clean_360 train_other_500 dev test_clean test_other; do
        if [ ! -s ${nnet_dir}/xvectors_${name}/xvector.scp ]; then
            sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj ${nj} \
                ${nnet_dir} data/${name}_mfcc_16k \
                ${nnet_dir}/xvectors_${name}
        fi
    done

    # Update json
    for name in train_clean_100 train_clean_360 train_other_500 dev test_clean test_other; do
        # data_${bpemode}${nbpe}_tts.json = original + x-vector
        cp ${dumpdir}/$name/delta${do_delta}/data_${bpemode}${nbpe}.json ${dumpdir}/$name/delta${do_delta}/data_${bpemode}${nbpe}_tts.json
        local/update_json.sh ${dumpdir}/$name/delta${do_delta}/data_${bpemode}${nbpe}_tts.json ${nnet_dir}/xvectors_$name/xvector.scp
    done
fi

# You can skip this and remove --rnnlm option in the recognition (stage 5)
#if [ -z ${lmtag} ]; then
#    lmtag=$(basename ${lm_config%.*})
#fi
#lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
#lmexpdir=exp/${lmexpname}
#mkdir -p ${lmexpdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: LM Preparation"
    lmdatadir=data/local/lm_train_${bpemode}${nbpe}
    mkdir -p ${lmdatadir}
    # use external data
    if [ ! -e data/local/lm_train/librispeech-lm-norm.txt.gz ]; then
        wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P data/local/lm_train/
    fi
    cut -f 2- -d" " data/${train_set}/text | gzip -c > data/local/lm_train/${train_set}_text.gz
    # combine external text and transcriptions and shuffle them with seed 777
    zcat data/local/lm_train/librispeech-lm-norm.txt.gz data/local/lm_train/${train_set}_text.gz |\
        spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
    cut -f 2- -d" " data/${train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece \
        > ${lmdatadir}/valid.txt
    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --dict ${dict}
fi

if [ -z ${asrpt_tag} ]; then
    asrpt_expname=${pretrain_set}_${backend}_$(basename ${asr_train_config%.*})
    if ${do_delta}; then
        asrpt_expname=${asrpt_expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        asrpt_expname=${asrpt_expname}_$(basename ${preprocess_config%.*})
    fi
else
    asrpt_expname=${train_set}_${backend}_${asrpt_tag}
fi
asrpt_expdir=exp/${asrpt_expname}

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: ASR pretraining"

    if [ $asr_train == 'true' ]; then
        mkdir -p ${asrpt_expdir}

        ${cuda_cmd} --gpu ${ngpu} ${asrpt_expdir}/train.log \
            asr_train.py \
            --config ${asr_train_config} \
            --preprocess-conf ${preprocess_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --outdir ${asrpt_expdir}/results \
            --tensorboard-dir tensorboard/${asrpt_expname} \
            --debugmode ${debugmode} \
            --dict ${dict} \
            --debugdir ${asrpt_expdir} \
            --minibatches ${N} \
            --verbose ${verbose} \
            --resume ${resume} \
            --train-json ${dumpdir}/${pretrain_set}/delta${do_delta}/data_${bpemode}${nbpe}.json \
            --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
    fi
    if [ $asr_decode == 'true' ]; then
        echo "### ASR PR decoding"
        nj=32

        pids=() # initialize pids
        for rtask in ${recog_set}; do
        (
            decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_asr_config%.*})_${lmtag}
            feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
            # split data
            splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json
            #### use CPU for decoding
            # set batchsize 0 to disable batch decoding
            ${decode_cmd} JOB=1:${nj} ${asrpt_expdir}/${decode_dir}/log/decode.JOB.log \
                asr_recog.py \
                    --ngpu 0 \
                    --config $decode_asr_config \
                    --backend ${backend} \
                    --batchsize 0 \
                    --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
                    --result-label ${asrpt_expdir}/${decode_dir}/data.JOB.json \
                    --model ${asrpt_expdir}/results/${recog_model}  \
                    --rnnlm ${lmexpdir}/rnnlm.model.best \
                    --api v2
            score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${asrpt_expdir}/${decode_dir} ${dict}
        ) &
        pids+=($!) # store background pids
        done
        i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
        [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

        echo "Finished"
    fi
fi

if [ -z ${ttspt_tag} ]; then
    ttspt_expname=ttspt_${pretrain_set}_${backend}_$(basename ${tts_train_config%.*})
else
    ttspt_expname=ttspt_${pretrain_set}_${backend}_${ttspt_tag}
fi
ttspt_expdir=exp/${ttspt_expname}

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: TTS pretraining"

    mkdir -p ${ttspt_expdir}

    tr_json=${dumpdir}/train_clean_100/delta${do_delta}/data_${bpemode}${nbpe}_tts.json
    dt_json=${dumpdir}/dev/delta${do_delta}/data_${bpemode}${nbpe}_tts.json
    ${cuda_cmd} --gpu ${ngpu} ${ttspt_expdir}/train.log \
        tts_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --outdir ${ttspt_expdir}/results \
           --tensorboard-dir tensorboard/${ttspt_expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${tts_train_config}
fi

if [ -z ${tag} ]; then
    asrtts_expname=${pretrain_set}_${backend}_$(basename ${asr_train_config%.*})
else
    asrtts_expname=${pretrain_set}_${backend}_${tag}
fi
asrtts_expdir=exp/${asrtts_expname}
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: ASR-TTS model training"

    mkdir -p ${asrtts_expdir}

    # tr = paired; utr = unpaired
    tr_json=${dumpdir}/train_clean_100/delta${do_delta}/data_${bpemode}${nbpe}_tts.json
    utr_json=${dumpdir}/train_clean_360/delta${do_delta}/data_${bpemode}${nbpe}_tts.json
    dt_json=${dumpdir}/dev/delta${do_delta}/data_${bpemode}${nbpe}_tts.json

    ${cuda_cmd} --gpu ${ngpu} ${asrtts_expdir}/train.log \
        asrtts_train.py --ngpu ${ngpu} \
            --outdir ${asrtts_expdir}/results \
            --tensorboard-dir tensorboard/${asrtts_expname} \
            --verbose ${verbose} \
            --debugmode 1 \
            --dict ${dict} \
            --debugdir ${asrtts_expdir} \
            --train-json ${tr_json} \
            --valid-json ${dt_json} \
            --train-unpaired-json ${utr_json} \
            --config ${asr_train_config} \
            --preprocess-conf ${preprocess_config} \
            --report-cer --beam-size 2 --nbest 2 \
            --minlenratio 0.3 --maxlenratio 0.8 \
            --tts true --sampling multinomial \
            --num-save-attention 0 \
            --model-module espnet.nets.pytorch_backend.e2e_asr:E2E \
            --asr-init exp/${expname}/results/model.acc.best \
            --tts-init exp/${expname}_TTS_transformer/results/model.loss.best \
            --speech-only
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
        if ${use_lm_valbest_average}; then
            lang_model=rnnlm.val${lm_n_average}.avg.best
            opt="--log ${lmexpdir}/log"
        else
            lang_model=rnnlm.last${lm_n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${lmexpdir}/snapshot.ep.* \
            --out ${lmexpdir}/${lang_model} \
            --num ${lm_n_average}
    fi
    nj=32

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/${lang_model}

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "decoding asr-tts model"
    recog_model=model.acc.best
    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${asrtts_expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${asrtts_expdir}/${decode_dir}/data.JOB.json \
            --model ${asrtts_expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/${lang_model}

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${asrtts_expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
