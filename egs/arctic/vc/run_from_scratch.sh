#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1
stop_stage=100
ngpu=1       # number of gpus ("0" uses cpu, otherwise use gpu)
nj=16        # numebr of parallel jobs
dumpdir=dump # directory to dump full features
verbose=0    # verbose option (if set > 0, get more log)
N=0          # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=16000      # sampling frequency
fmax=7600     # maximum frequency
fmin=80       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length

# config files
#train_config=conf/train_pytorch_transformer.v1.single.finetune.yaml # you can select from conf or conf/tuning.
                                                                    # now we support tacotron2 and transformer for TTS.
                                                                    # see more info in the header of each config.
train_config=conf/vc_from_scratch_v1.yaml
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
n_average=1           # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

download_dir=downloads

# objective evaluation related
db_root=downloads
asr_model="librispeech.transformer.ngpu4"
eval_model=true                                # true: evaluate trained model, false: evaluate ground truth
wer=true                                       # true: evaluate CER & WER, false: evaluate only CER
vocoder=                                       # select vocoder type (GL, WNV, PWG)
mcd=true                                       # true: evaluate MCD
mcep_dim=24
shift_ms=5

# dataset configuration
srcspk=clb  # see local/data_prep.sh to check available speakers
trgspk=slt
num_train_utts=-1

# exp tag
tag=""  # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

pair=${srcspk}_${trgspk}
src_org_set=${srcspk}
src_train_set=${srcspk}_train_no_dev
src_dev_set=${srcspk}_dev
src_eval_set=${srcspk}_eval
trg_org_set=${trgspk}
trg_train_set=${trgspk}_train_no_dev
trg_dev_set=${trgspk}_dev
trg_eval_set=${trgspk}_eval
pair_train_set=${pair}_train_no_dev
pair_dev_set=${pair}_dev
pair_eval_set=${pair}_eval

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data and Pretrained Model Download"
    local/data_download.sh ${download_dir} ${srcspk}
    local/data_download.sh ${download_dir} ${trgspk}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    
    local/data_prep.sh ${download_dir} ${srcspk} data/${src_org_set}
    utils/fix_data_dir.sh data/${src_org_set}
    utils/validate_data_dir.sh --no-feats data/${src_org_set}
    
    local/data_prep.sh ${download_dir} ${trgspk} data/${trg_org_set}
    utils/fix_data_dir.sh data/${trg_org_set}
    utils/validate_data_dir.sh --no-feats data/${trg_org_set}
fi

src_feat_tr_dir=${dumpdir}/${src_train_set}; mkdir -p ${src_feat_tr_dir}
src_feat_dt_dir=${dumpdir}/${src_dev_set}; mkdir -p ${src_feat_dt_dir}
src_feat_ev_dir=${dumpdir}/${src_eval_set}; mkdir -p ${src_feat_ev_dir}
trg_feat_tr_dir=${dumpdir}/${trg_train_set}; mkdir -p ${trg_feat_tr_dir}
trg_feat_dt_dir=${dumpdir}/${trg_dev_set}; mkdir -p ${trg_feat_dt_dir}
trg_feat_ev_dir=${dumpdir}/${trg_eval_set}; mkdir -p ${trg_feat_ev_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev name by yourself.
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
        data/${src_org_set} \
        exp/make_fbank/${src_org_set} \
        ${fbankdir}
    make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
        --fs ${fs} \
        --fmax "${fmax}" \
        --fmin "${fmin}" \
        --n_fft ${n_fft} \
        --n_shift ${n_shift} \
        --win_length "${win_length}" \
        --n_mels ${n_mels} \
        data/${trg_org_set} \
        exp/make_fbank/${trg_org_set} \
        ${fbankdir}

    # make a dev set
    utils/subset_data_dir.sh --last data/${src_org_set} 200 data/${src_org_set}_tmp
    utils/subset_data_dir.sh --last data/${src_org_set}_tmp 100 data/${src_eval_set}
    utils/subset_data_dir.sh --first data/${src_org_set}_tmp 100 data/${src_dev_set}
    n=$(( $(wc -l < data/${src_org_set}/wav.scp) - 200 ))
    utils/subset_data_dir.sh --first data/${src_org_set} ${n} data/${src_train_set}
    rm -rf data/${src_org_set}_tmp
    
    utils/subset_data_dir.sh --last data/${trg_org_set} 200 data/${trg_org_set}_tmp
    utils/subset_data_dir.sh --last data/${trg_org_set}_tmp 100 data/${trg_eval_set}
    utils/subset_data_dir.sh --first data/${trg_org_set}_tmp 100 data/${trg_dev_set}
    n=$(( $(wc -l < data/${trg_org_set}/wav.scp) - 200 ))
    utils/subset_data_dir.sh --first data/${trg_org_set} ${n} data/${trg_train_set}
    rm -rf data/${trg_org_set}_tmp

    # compute statistics for global mean-variance normalization
    if [ ! -e "${pair_tr_dir}/feats.scp" ]; then
        cp data/${src_train_set}/feats.scp ${pair_tr_dir}/feats.scp
        cat data/${trg_train_set}/feats.scp >> ${pair_tr_dir}/feats.scp
    fi
    cmvn=${pair_tr_dir}/cmvn.ark
    compute-cmvn-stats scp:${pair_tr_dir}/feats.scp ${cmvn}

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${src_train_set}/feats.scp ${cmvn} exp/dump_feats/${src_train_set} ${src_feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${src_dev_set}/feats.scp ${cmvn} exp/dump_feats/${src_dev_set} ${src_feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${src_eval_set}/feats.scp ${cmvn} exp/dump_feats/${src_eval_set} ${src_feat_ev_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${trg_train_set}/feats.scp ${cmvn} exp/dump_feats/${trg_train_set} ${trg_feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${trg_dev_set}/feats.scp ${cmvn} exp/dump_feats/${trg_dev_set} ${trg_feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${trg_eval_set}/feats.scp ${cmvn} exp/dump_feats/${trg_eval_set} ${trg_feat_ev_dir}
fi

pair_tr_dir=${dumpdir}/${pair_train_set}; mkdir -p ${pair_tr_dir}
pair_dt_dir=${dumpdir}/${pair_dev_set}; mkdir -p ${pair_dt_dir}
pair_ev_dir=${dumpdir}/${pair_eval_set}; mkdir -p ${pair_ev_dir}
# dummy dict
dict=downloads/mailabs.en_US.judy.transformer.v1.single/data/lang_1char/en_US_judy_train_trim_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"

    # make json labels using pretrained model dict
    data2json.sh --feat ${src_feat_tr_dir}/feats.scp \
         data/${src_train_set} ${dict} > ${src_feat_tr_dir}/data.json
    data2json.sh --feat ${src_feat_dt_dir}/feats.scp \
         data/${src_dev_set} ${dict} > ${src_feat_dt_dir}/data.json
    data2json.sh --feat ${src_feat_ev_dir}/feats.scp \
         data/${src_eval_set} ${dict} > ${src_feat_ev_dir}/data.json
    
    data2json.sh --feat ${trg_feat_tr_dir}/feats.scp \
         data/${trg_train_set} ${dict} > ${trg_feat_tr_dir}/data.json
    data2json.sh --feat ${trg_feat_dt_dir}/feats.scp \
         data/${trg_dev_set} ${dict} > ${trg_feat_dt_dir}/data.json
    data2json.sh --feat ${trg_feat_ev_dir}/feats.scp \
         data/${trg_eval_set} ${dict} > ${trg_feat_ev_dir}/data.json

    # make pair json
    local/make_pair_json.py --src-json ${src_feat_tr_dir}/data.json \
         --trg-json ${trg_feat_tr_dir}/data.json -O ${pair_tr_dir}/data.json
    local/make_pair_json.py --src-json ${src_feat_dt_dir}/data.json \
         --trg-json ${trg_feat_dt_dir}/data.json -O ${pair_dt_dir}/data.json
    local/make_pair_json.py --src-json ${src_feat_ev_dir}/data.json \
         --trg-json ${trg_feat_ev_dir}/data.json -O ${pair_ev_dir}/data.json
fi


if [ -z ${tag} ]; then
    expname=${srcspk}_${trgspk}_${backend}_$(basename ${train_config%.*})
else
    expname=${srcspk}_${trgspk}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: VC model training"

    if [ ${num_train_utts} -ge 0 ]; then
        tr_json=${pair_tr_dir}/data_n${num_train_utts}.json
    else
        tr_json=${pair_tr_dir}/data.json
    fi
    dt_json=${pair_dt_dir}/data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        vc_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --minibatches ${N} \
           --outdir ${expdir}/results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --srcspk ${srcspk} \
           --trgspk ${trgspk} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config}
fi

if [ ${n_average} -gt 0 ]; then
    model=model.last${n_average}.avg.best
fi
outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Decoding"
    if [ ${n_average} -gt 0 ]; then
        average_checkpoints.py --backend ${backend} \
                               --snapshots ${expdir}/results/snapshot.ep.* \
                               --out ${expdir}/results/${model} \
                               --num ${n_average}
    fi
    pids=() # initialize pids
    for name in ${pair_dev_set} ${pair_eval_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}/data.json ${outdir}/${name}
        splitjson.py --parts ${nj} ${outdir}/${name}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            vc_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${name}/feats.JOB \
                --json ${outdir}/${name}/split${nj}utt/data.JOB.json \
                --model ${expdir}/results/${model} \
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

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Synthesis"
    pids=() # initialize pids
    for name in ${pair_dev_set} ${pair_eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
        # use pretrained model cmvn
        cmvn=$(find ${download_dir}/${pretrained_model} -name "cmvn.ark" | head -n 1)
        apply-cmvn --norm-vars=true --reverse=true ${cmvn} \
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
    echo "Finished."
fi
