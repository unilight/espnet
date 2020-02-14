#!/bin/bash

# Copyright 2018 Nagoya University (Takenori Yoshimura), Ryuichi Yamamoto
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1
stop_stage=100
ngpu=1       # number of gpus ("0" uses cpu, otherwise use gpu)
nj=32        # numebr of parallel jobs
dumpdir=dump # directory to dump full features
verbose=0    # verbose option (if set > 0, get more log)
N=0          # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=24000    # sampling frequency
fmax=7600   # maximum frequency
fmin=80     # minimum frequency
n_mels=80   # number of mel basis
n_fft=2048  # number of fft points
n_shift=300 # number of shift points
win_length=1200 # window length

# Input transcription type: char or phn
# Example
#  char: ミズヲマレーシアカラカワナクテワナラナイノデス。
#  phn: m i z u o m a r e e sh i a k a r a k a w a n a k U t e w a n a r a n a i n o d e s U
# NOTE: original transcription is provided by 漢字仮名交じり文. We convert the input to
# kana or phoneme using OpenJTalk's NLP frontend at the data prep. stage.
trans_type="phn"

# config files
train_config=conf/train_pytorch_transformer.yaml
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
n_average=1 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# pretrained model related
tts_train_config=
pretrained_model_path=
load_partial_pretrained_model="encoder"
params_to_train="encoder"
ae_train_config="conf/ae_R2R2.yaml"

# root directory of db
db_root=downloads

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="${trans_type}_train_no_dev"
train_dev="${trans_type}_dev"
eval_set="${trans_type}_eval"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/download.sh ${db_root}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/data_prep.sh ${db_root}/jsut_ver1.1/ data/${trans_type}_train ${trans_type}

    # Downsample to fs from 48k
    utils/data/resample_data_dir.sh $fs data/${trans_type}_train

    utils/validate_data_dir.sh --no-feats data/${trans_type}_train
fi

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}; mkdir -p ${feat_dt_dir}
feat_ev_dir=${dumpdir}/${eval_set}; mkdir -p ${feat_ev_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    # Generate the fbank features; by default 80-dimensional fbanks on each frame
    fbankdir=fbank
    make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
        --fs ${fs} \
        --fmax "${fmax}" \
        --fmin "${fmin}" \
        --n_fft ${n_fft} \
        --n_shift ${n_shift} \
        --win_length "${win_length}" \
        --n_mels ${n_mels} \
        data/${trans_type}_train \
        exp/make_fbank/train \
        ${fbankdir}

    # make a dev set
    utils/subset_data_dir.sh --first data/${trans_type}_train 500 data/${trans_type}_deveval
    utils/subset_data_dir.sh --first data/${trans_type}_deveval 250 data/${eval_set}
    utils/subset_data_dir.sh --last data/${trans_type}_deveval 250 data/${train_dev}
    n=$(( $(wc -l < data/${trans_type}_train/wav.scp) - 500 ))
    utils/subset_data_dir.sh --last data/${trans_type}_train ${n} data/${train_set}

    # compute statistics for global mean-variance normalization
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${trans_type}_train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${trans_type}_dev ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${trans_type}_eval ${feat_ev_dir}
fi

dict=data/lang_${trans_type}/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_${trans_type}/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 --trans_type ${trans_type} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --trans_type ${trans_type} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --trans_type ${trans_type} \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    data2json.sh --feat ${feat_ev_dir}/feats.scp --trans_type ${trans_type} \
         data/${eval_set} ${dict} > ${feat_ev_dir}/data.json
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Text-to-speech model training"
    tr_json=${feat_tr_dir}/data.json
    dt_json=${feat_dt_dir}/data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        tts_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --minibatches ${N} \
           --outdir ${expdir}/results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config}
fi

outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Decoding"

    pids=() # initialize pids
    for sets in ${train_dev} ${eval_set}; do
    (
        [ ! -e ${outdir}/${sets} ] && mkdir -p ${outdir}/${sets}
        cp ${dumpdir}/${sets}/data.json ${outdir}/${sets}
        splitjson.py --parts ${nj} ${outdir}/${sets}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${sets}/log/decode.JOB.log \
            tts_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${sets}/feats.JOB \
                --json ${outdir}/${sets}/split${nj}utt/data.JOB.json \
                --model ${expdir}/results/${model} \
                --config ${decode_config}
        # concatenate scp files
        for n in $(seq ${nj}); do
            cat "${outdir}/${sets}/feats.$n.scp" || exit 1;
        done > ${outdir}/${sets}/feats.scp
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Synthesis"
    pids=() # initialize pids
    for sets in ${train_dev} ${eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${sets} ] && mkdir -p ${outdir}_denorm/${sets}
        apply-cmvn --norm-vars=true --reverse=true data/${train_set}/cmvn.ark \
            scp:${outdir}/${sets}/feats.scp \
            ark,scp:${outdir}_denorm/${sets}/feats.ark,${outdir}_denorm/${sets}/feats.scp
        convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            --iters ${griffin_lim_iters} \
            ${outdir}_denorm/${sets} \
            ${outdir}_denorm/${sets}/log \
            ${outdir}_denorm/${sets}/wav
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Objective Evaluation for TTS"

    for name in ${dev_set} ${eval_set}; do
        local/ob_eval/evaluate_cer.sh --nj ${nj} \
            --do_delta false \
            --eval_gt ${eval_gt} \
            --eval_tts ${eval_tts} \
            --db_root ${db_root} \
            --backend pytorch \
            --wer ${wer} \
            --api v2 \
            ${asr_model} \
            ${outdir} \
            ${name} \
            ${spk}
    done

    echo "Finished."
fi


expdir=exp/${expname}
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: mel autoencoder training"

    # make pair json
    if [ ! -e ${feat_tr_dir}/ae_data.json ]; then
        echo "Making training json file for mel autoencoder"
        local/make_ae_json.py --json ${feat_tr_dir}/data.json -O ${feat_tr_dir}/ae_data.json
    fi
    if [ ! -e ${feat_dt_dir}/ae_data.json ]; then
        echo "Making development json file for mel autoencoder"
        local/make_ae_json.py --json ${feat_dt_dir}/data.json -O ${feat_dt_dir}/ae_data.json
    fi
    if [ ! -e ${feat_ev_dir}/ae_data.json ]; then
        echo "Making evaluation json file for mel autoencoder"
        local/make_ae_json.py --json ${feat_ev_dir}/data.json -O ${feat_ev_dir}/ae_data.json
        exit 1
    fi

    # check input arguments
    if [ -z ${tag} ]; then
        echo "Please specify exp tag."
        exit 1
    fi
    if [ -z ${pretrained_model_path} ]; then
        echo "Please specify pre-trained tts model path."
        exit 1
    fi
    if [ -z ${tts_train_config} ]; then
        echo "Please specify pre-trained tts model config."
        exit 1
    fi

    expname=${train_set}_${backend}_${tag}
    expdir=exp/${expname}
    train_config="$(change_yaml.py -a pretrained-model="${pretrained_model_path}" \
        -a load-partial-pretrained-model="${load_partial_pretrained_model}" \
        -a params-to-train="${params_to_train}" \
        -d model-module \
        -o "conf/$(basename "${tts_train_config}" .yaml).ae.yaml" "${tts_train_config}")"

    tr_json=${feat_tr_dir}/ae_data.json
    dt_json=${feat_dt_dir}/ae_data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/ae_train.log \
        vc_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --minibatches ${N} \
           --outdir ${expdir}/ae_results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --srcspk jsut \
           --trgspk jsut \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config} \
           --config2 ${ae_train_config}
fi

outdir=${expdir}/ae_outputs_${model}
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: Mel autoencoder decoding"

    pids=() # initialize pids
    for name in ${train_dev} ${eval_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}/ae_data.json ${outdir}/${name}
        splitjson.py --parts ${nj} ${outdir}/${name}/ae_data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            vc_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${name}/feats.JOB \
                --json ${outdir}/${name}/split${nj}utt/ae_data.JOB.json \
                --model ${expdir}/ae_results/${model} \
                --config ${decode_config}
        # concatenate scp files
        for n in $(seq ${nj}); do
            cat "${outdir}/${name}/feats.$n.scp" || exit 1;
        done > ${outdir}/${name}/feats.scp
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "stage 9: Mel autoencoder synthesis"
    pids=() # initialize pids
    for name in ${train_dev} ${eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
        apply-cmvn --norm-vars=true --reverse=true data/${train_set}/cmvn.ark \
            scp:${outdir}/${name}/feats.scp \
            ark,scp:${outdir}_denorm/${name}/feats.ark,${outdir}_denorm/${name}/feats.scp
        convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            --iters ${griffin_lim_iters} \
            ${outdir}_denorm/${name} \
            ${outdir}_denorm/${name}/log \
            ${outdir}_denorm/${name}/wav

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "stage 10: Mel autoencoder: generate hdf5"

    # generate h5 for WaveNet vocoder
    for name in ${dev_set} ${eval_set}; do
        feats2hdf5.py \
            --scp_file ${outdir}_denorm/${name}/feats.scp \
            --out_dir ${outdir}_denorm/${name}/hdf5/
        (find "$(cd ${outdir}_denorm/${name}/hdf5; pwd)" -name "*.h5" -print &) | head > ${outdir}_denorm/${name}/hdf5_feats.scp
        echo "generated hdf5 for ${name} set"
    done
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    echo "stage 11: Objective Evaluation"

    for name in ${dev_set} ${eval_set}; do
        local/ob_eval/evaluate_cer.sh --nj ${nj} \
            --do_delta false \
            --eval_gt ${eval_gt} \
            --eval_tts ${eval_tts} \
            --db_root ${db_root} \
            --backend pytorch \
            --wer ${wer} \
            --api v2 \
            ${asr_model} \
            ${outdir} \
            ${name} \
            ${spk}
    done

    echo "Finished."
fi
