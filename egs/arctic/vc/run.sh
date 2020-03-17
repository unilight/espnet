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
train_config=conf/vc_v1.yaml
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
n_average=1           # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# pretrained model related
download_dir=downloads
pretrained_model=
load_partial_pretrained_model="encoder"
pretrained_model2=
load_partial_pretrained_model2="encoder"
params_to_train=
pretrained_cmvn=

pretrained_asr_model=librispeech.transformer_large
pretrained_dec_model=

# non parallel training related
encoder_model_path=
decoder_model_path=
decoder_model_json=
np_outdir=

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
specified_train_json=
num_train_utts=-1
aespk=

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
ae_train_set=${aespk}_train_no_dev
ae_dev_set=${aespk}_dev
ae_eval_set=${aespk}_eval

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data and Pretrained Model Download"
    local/data_download.sh ${download_dir} ${srcspk}
    local/data_download.sh ${download_dir} ${trgspk}
    local/pretrained_model_download.sh ${download_dir} ${pretrained_model}
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
    
    fbankdir=fbank
    for spk_org_set in ${src_org_set} ${trg_org_set}; do
        # Generate the fbank features; by default 80-dimensional fbanks on each frame
        make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            data/${spk_org_set} \
            exp/make_fbank/${spk_org_set} \
            ${fbankdir}
    
        # Generate the fbank+pitch features; by default 80-dimensional fbanks on each frame
        steps/make_fbank_pitch.sh --cmd "${train_cmd}" --nj ${nj} \
            --write_utt2num_frames true \
            data/${spk_org_set}_pitch \
            exp/make_fbank/${spk_org_set}_pitch \
            ${fbankdir}
        utils/fix_data_dir.sh data/${spk_org_set}_pitch
    done
    

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

    # use pretrained model cmvn
    cmvn=$(find ${download_dir}/${pretrained_model} -name "cmvn.ark" | head -n 1)

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
dict=$(find ${download_dir}/${pretrained_model} -name "*_units.txt" | head -n 1)
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
    if [ ${num_train_utts} -ge 0 ]; then
        local/make_pair_json.py --src-json ${src_feat_tr_dir}/data.json \
             --trg-json ${trg_feat_tr_dir}/data.json -O ${pair_tr_dir}/data_n${num_train_utts}.json \
             --num_utts ${num_train_utts}
    else
        local/make_pair_json.py --src-json ${src_feat_tr_dir}/data.json \
             --trg-json ${trg_feat_tr_dir}/data.json -O ${pair_tr_dir}/data.json
    fi
    local/make_pair_json.py --src-json ${src_feat_dt_dir}/data.json \
         --trg-json ${trg_feat_dt_dir}/data.json -O ${pair_dt_dir}/data.json
    local/make_pair_json.py --src-json ${src_feat_ev_dir}/data.json \
         --trg-json ${trg_feat_ev_dir}/data.json -O ${pair_ev_dir}/data.json
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: VC model training"

    # add pretrained model info in config
    if [ ! -z ${pretrained_model} ]; then
        pretrained_model_path=$(find ${download_dir}/${pretrained_model} -name "model*.best" | head -n 1)
        train_config="$(change_yaml.py -a pretrained-model="${pretrained_model_path}" \
            -a load-partial-pretrained-model="${load_partial_pretrained_model}" \
            -a params-to-train="${params_to_train}" \
            -o "conf/$(basename "${train_config}" .yaml).${pretrained_model}.yaml" "${train_config}")"
    fi

    if [ ! -z ${pretrained_model2} ]; then
        pretrained_model2_path=$(find ${download_dir}/${pretrained_model2} -name "model*.best" | head -n 1)
        train_config="$(change_yaml.py -a pretrained-model2="${pretrained_model2_path}" \
            -a load-partial-pretrained-model2="${load_partial_pretrained_model2}" \
            -o "conf/$(basename "${train_config}" .yaml).${pretrained_model2}.yaml" "${train_config}")"
    fi

    if [ -z ${tag} ]; then
        expname=${srcspk}_${trgspk}_${backend}_$(basename ${train_config%.*})
    else
        expname=${srcspk}_${trgspk}_${backend}_${tag}
    fi
    expdir=exp/${expname}

    if [ -z ${specified_train_json} ]; then
        if [ ${num_train_utts} -ge 0 ]; then
            tr_json=${pair_tr_dir}/data_n${num_train_utts}.json
        else
            tr_json=${pair_tr_dir}/data.json
        fi
    else
        tr_json=${specified_train_json}
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

if [ -z ${tag} ]; then
    expname=${srcspk}_${trgspk}_${backend}_$(basename ${train_config%.*})
else
    expname=${srcspk}_${trgspk}_${backend}_${tag}
fi
expdir=exp/${expname}
outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})
echo $outdir
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Decoding"

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
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: generate hdf5"

    # generate h5 for WaveNet vocoder
    for name in ${pair_dev_set} ${pair_eval_set}; do
        feats2hdf5.py \
            --scp_file ${outdir}_denorm/${name}/feats.scp \
            --out_dir ${outdir}_denorm/${name}/hdf5/
        (find "$(cd ${outdir}_denorm/${name}/hdf5; pwd)" -name "*.h5" -print &) | head > ${outdir}_denorm/${name}/hdf5_feats.scp
        echo "generated hdf5 for ${name} set"
    done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Objective Evaluation"

    for set_type in dev eval; do

        local/ob_eval/evaluate.sh --nj ${nj} \
            --do_delta false \
            --eval_model ${eval_model} \
            --db_root ${db_root} \
            --backend pytorch \
            --wer ${wer} \
            --mcd ${mcd} \
            --vocoder ${vocoder} \
            --api v2 \
            ${asr_model} \
            ${outdir} \
            ${srcspk}_${trgspk}_${set_type} \
            ${trgspk}_${set_type} \
            ${trgspk}
    done
fi

#####################################################################################

ae_tr_dir=${dumpdir}/${ae_train_set}
ae_dt_dir=${dumpdir}/${ae_dev_set}
ae_ev_dir=${dumpdir}/${ae_eval_set}
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: mel autoencoder training"
    
    if [ -z ${aespk} ]; then
        echo "Please specify the speaker to train autoencoder."
        exit 1
    fi

    if [ ${num_train_utts} -ge 0 ]; then
        if [ ! -e ${ae_tr_dir}/ae_data_n${num_train_utts}.json ]; then
            echo "Making training json file for mel autoencoder"
            local/make_ae_json.py --json ${ae_tr_dir}/data.json -O ${ae_tr_dir}/ae_data_n${num_train_utts}.json \
                 --num_utts ${num_train_utts}
        fi
    else
        if [ ! -e ${ae_tr_dir}/ae_data.json ]; then
            echo "Making training json file for mel autoencoder"
            local/make_ae_json.py --json ${ae_tr_dir}/data.json -O ${ae_tr_dir}/ae_data.json
        fi
    fi
    if [ ! -e ${ae_dt_dir}/ae_data.json ]; then
        echo "Making development json file for mel autoencoder"
        local/make_ae_json.py --json ${ae_dt_dir}/data.json -O ${ae_dt_dir}/ae_data.json
    fi
    if [ ! -e ${ae_ev_dir}/ae_data.json ]; then
        echo "Making evaluation json file for mel autoencoder"
        local/make_ae_json.py --json ${ae_ev_dir}/data.json -O ${ae_ev_dir}/ae_data.json
        exit 1
    fi

    # check input arguments
    if [ -z ${tag} ]; then
        echo "Please specify exp tag."
        exit 1
    fi
    if [ -z ${pretrained_model} ]; then
        echo "Please specify pre-trained tts model."
        exit 1
    fi

    expname=${ae_train_set}_${tag}
    expdir=exp/${expname}
    mkdir -p ${expdir}

    # add pretrained model info in config
    pretrained_model_path=$(find ${download_dir}/${pretrained_model} -name "model*.best" | head -n 1)
    train_config="$(change_yaml.py -a pretrained-model="${pretrained_model_path}" \
        -a load-partial-pretrained-model="${load_partial_pretrained_model}" \
        -a params-to-train="${params_to_train}" \
        -o "conf/ae.$(basename "${train_config}" .yaml).${pretrained_model}.yaml" "${train_config}")"

    if [ ! -z ${pretrained_model2} ]; then
        pretrained_model2_path=$(find ${download_dir}/${pretrained_model2} -name "model*.best" | head -n 1)
        train_config="$(change_yaml.py -a pretrained-model2="${pretrained_model2_path}" \
            -a load-partial-pretrained-model2="${load_partial_pretrained_model2}" \
            -o "conf/$(basename "${train_config}" .yaml).${pretrained_model2}.yaml" "${train_config}")"
    fi

    if [ ${num_train_utts} -ge 0 ]; then
        tr_json=${ae_tr_dir}/ae_data_n${num_train_utts}.json
    else
        tr_json=${ae_tr_dir}/ae_data.json
    fi
    dt_json=${ae_dt_dir}/ae_data.json
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
           --srcspk ${aespk} \
           --trgspk ${aespk} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config}
fi

expname=${ae_train_set}_${tag}
expdir=exp/${expname}
outdir=${expdir}/ae_outputs_${model}
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "stage 9: Mel autoencoder decoding"

    pids=() # initialize pids
    for name in ${ae_dev_set} ${ae_eval_set}; do
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

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "stage 10: Mel autoencoder synthesis"
    pids=() # initialize pids
    for name in ${ae_dev_set} ${ae_eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
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
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    echo "stage 11: Objective Evaluation for Mel Autoencoder"

    for name in ${ae_dev_set} ${ae_eval_set}; do
        local/ob_eval/evaluate_cer.sh --nj ${nj} \
            --do_delta false \
            --eval_tts_model true \
            --db_root ${db_root} \
            --backend pytorch \
            --wer ${wer} \
            --api v2 \
            ${asr_model} \
            ${outdir} \
            ${name} \
            ${aespk}
    done

    echo "Finished."
fi

###################################################################3

# Guide:
# ./run.sh --stage 12 --stop_stage 14 \
#   --tag n932_baseline 
#   --encoder_model_path exp/clb_train_no_dev_932/ae_results/snapshot.ep.100 \
#   --decoder_model_path exp/slt_train_no_dev_932/ae_results/snapshot.ep.100 \
#   --verbose 1 --decoder_model_json

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    echo "stage 12: Non-parallel Decoding, Synthesizing and Hdf5 Generation"
    
    expdir=exp/np_${srcspk}_${trgspk}_${tag}
    if [ ! -z ${encoder_model_path} ]; then
        model=${srcspk}_$(basename ${encoder_model_path} | sed -e "s/snapshot\.ep\.//g")_${trgspk}_$(basename ${decoder_model_path} | sed -e "s/snapshot\.ep\.//g")
    fi
    outdir=${expdir}/outputs_${model}
    mkdir -p $outdir
    echo $outdir

    local/merge_models.py \
        --enc_snapshot ${encoder_model_path} \
        --dec_snapshot ${decoder_model_path} \
        --out ${expdir}/${model}

    # copy model.json from decoder side and delete pretrained model related keys
    local/remove_pretrained_keys.py \
        --json_file ${decoder_model_json} \
        --out ${expdir}/model.json

    
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
                --model ${expdir}/${model} \
                --model-conf ${expdir}/model.json \
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
    
    echo "stage 13: Synthesis"
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

    echo "stage 14: generate hdf5"

    # generate h5 for WaveNet vocoder
    for name in ${pair_dev_set} ${pair_eval_set}; do
        feats2hdf5.py \
            --scp_file ${outdir}_denorm/${name}/feats.scp \
            --out_dir ${outdir}_denorm/${name}/hdf5/
        (find "$(cd ${outdir}_denorm/${name}/hdf5; pwd)" -name "*.h5" -print &) | head > ${outdir}_denorm/${name}/hdf5_feats.scp
        echo "generated hdf5 for ${name} set"
    done
fi

if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
    echo "stage 15: Objective Evaluation"

    for set_type in dev eval; do

        local/ob_eval/evaluate.sh --nj ${nj} \
            --do_delta false \
            --eval_model ${eval_model} \
            --db_root ${db_root} \
            --backend pytorch \
            --wer ${wer} \
            --mcd ${mcd} \
            --vocoder ${vocoder} \
            --api v2 \
            ${asr_model} \
            ${np_outdir} \
            ${srcspk}_${trgspk}_${set_type} \
            ${trgspk}_${set_type} \
            ${trgspk}
    done
fi

###########################################################################

ae_tr_dir=${dumpdir}/${ae_train_set}
ae_dt_dir=${dumpdir}/${ae_dev_set}
ae_ev_dir=${dumpdir}/${ae_eval_set}
if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ]; then
    echo "stage 16: target decoder training"
    
    if [ -z ${aespk} ]; then
        echo "Please specify the speaker to train target decoder."
        exit 1
    fi

    if [ ${num_train_utts} -ge 0 ]; then
        if [ ! -e ${ae_tr_dir}/ae_data_n${num_train_utts}.json ]; then
            echo "Making training json file for training target decoder"
            local/make_ae_json.py --json ${ae_tr_dir}/data.json -O ${ae_tr_dir}/ae_data_n${num_train_utts}.json \
                 --num_utts ${num_train_utts}
        fi
    else
        if [ ! -e ${ae_tr_dir}/ae_data.json ]; then
            echo "Making training json file for training target decoder"
            local/make_ae_json.py --json ${ae_tr_dir}/data.json -O ${ae_tr_dir}/ae_data.json
        fi
    fi
    if [ ! -e ${ae_dt_dir}/ae_data.json ]; then
        echo "Making development json file for training target decoder"
        local/make_ae_json.py --json ${ae_dt_dir}/data.json -O ${ae_dt_dir}/ae_data.json
    fi
    if [ ! -e ${ae_ev_dir}/ae_data.json ]; then
        echo "Making evaluation json file for training target decoder"
        local/make_ae_json.py --json ${ae_ev_dir}/data.json -O ${ae_ev_dir}/ae_data.json
        exit 1
    fi

    # check input arguments
    if [ -z ${tag} ]; then
        echo "Please specify exp tag."
        exit 1
    fi
    if [ -z ${pretrained_model2} ]; then
        echo "Please specify pre-trained asr model."
        exit 1
    fi

    expname=${ae_train_set}_${tag}
    expdir=exp/${expname}
    mkdir -p ${expdir}

    # add pretrained model info in config
    if [ ! -z ${pretrained_model} ]; then
        pretrained_model_path=$(find ${download_dir}/${pretrained_model} -name "model*.best" | head -n 1)
        train_config="$(change_yaml.py -a pretrained-model="${pretrained_model_path}" \
            -a load-partial-pretrained-model="${load_partial_pretrained_model}" \
            -o "conf/$(basename "${train_config}" .yaml).${pretrained_model}.yaml" "${train_config}")"
    fi
    
    pretrained_model2_path=$(find ${download_dir}/${pretrained_model2} -name "model*.best" | head -n 1)
    train_config="$(change_yaml.py -a pretrained-model2="${pretrained_model2_path}" \
        -a load-partial-pretrained-model2="${load_partial_pretrained_model2}" \
        -o "conf/m2o.$(basename "${train_config}" .yaml).${pretrained_model2}.yaml" "${train_config}")"

    if [ ${num_train_utts} -ge 0 ]; then
        tr_json=${ae_tr_dir}/ae_data_n${num_train_utts}.json
    else
        tr_json=${ae_tr_dir}/ae_data.json
    fi
    dt_json=${ae_dt_dir}/ae_data.json
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
           --srcspk ${aespk} \
           --trgspk ${aespk} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config}
fi

if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ]; then
    echo "stage 17: Many-to-one Decoding, Synthesis, and hdf5 generation"

    if [ -z ${tag} ]; then
        echo 'Please specify experiment tag'
        exit 1
    fi
    
    expname=${ae_train_set}_${tag}
    expdir=exp/${expname}
    outdir=${expdir}/${srcspk}_outputs_${model}_$(basename ${decode_config%.*})
    echo $outdir
    
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
                --model ${expdir}/ae_results/${model} \
                --model-conf ${expdir}/ae_results/model.json \
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
    
    # Synthesis
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
    
    # generate h5 for neural vocoder
    for name in ${pair_dev_set} ${pair_eval_set}; do
        feats2hdf5.py \
            --scp_file ${outdir}_denorm/${name}/feats.scp \
            --out_dir ${outdir}_denorm/${name}/hdf5/
        (find "$(cd ${outdir}_denorm/${name}/hdf5; pwd)" -name "*.h5" -print &) | head > ${outdir}_denorm/${name}/hdf5_feats.scp
        echo "generated hdf5 for ${name} set"
    done

fi

if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ]; then
    echo "stage 18: Objective Evaluation"
    
    if [ -z ${tag} ]; then
        echo 'Please specify experiment tag'
        exit 1
    fi
    
    expname=${ae_train_set}_${tag}
    expdir=exp/${expname}
    outdir=${expdir}/${srcspk}_outputs_${model}_$(basename ${decode_config%.*})

    for set_type in dev eval; do

        local/ob_eval/evaluate.sh --nj ${nj} \
            --do_delta false \
            --eval_model ${eval_model} \
            --db_root ${db_root} \
            --backend pytorch \
            --wer ${wer} \
            --mcd ${mcd} \
            --vocoder ${vocoder} \
            --api v2 \
            ${asr_model} \
            ${outdir} \
            ${srcspk}_${trgspk}_${set_type} \
            ${trgspk}_${set_type} \
            ${trgspk}
    done
fi

#################################################################

# 20200212
# ASR pretraining (for TASLP journal)
# Note: this is still one-to-one VC, different from many-to-one VC

src_feat_tr_pitch_dir=${dumpdir}/${src_train_set}_pitch; mkdir -p ${src_feat_tr_pitch_dir}
src_feat_dt_pitch_dir=${dumpdir}/${src_dev_set}_pitch; mkdir -p ${src_feat_dt_pitch_dir}
src_feat_ev_pitch_dir=${dumpdir}/${src_eval_set}_pitch; mkdir -p ${src_feat_ev_pitch_dir}

if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ]; then
    echo "stage 21: Feature Generation with pitch (for src only)"
    
    [ -e data/${src_org_set}_pitch ] && rm -r data/${src_org_set}_pitch
    cp -r data/${src_org_set} data/${src_org_set}_pitch
    
    fbankdir=fbank
    # Generate the fbank+pitch features; by default 80-dimensional fbanks on each frame
    steps/make_fbank_pitch.sh --cmd "${train_cmd}" --nj ${nj} \
        --write_utt2num_frames true \
        data/${src_org_set}_pitch \
        exp/make_fbank/${src_org_set}_pitch \
        ${fbankdir}
    utils/fix_data_dir.sh data/${src_org_set}_pitch

    # make a dev set
    utils/subset_data_dir.sh --last data/${src_org_set}_pitch 200 data/${src_org_set}_pitch_tmp
    utils/subset_data_dir.sh --last data/${src_org_set}_pitch_tmp 100 data/${src_eval_set}_pitch
    utils/subset_data_dir.sh --first data/${src_org_set}_pitch_tmp 100 data/${src_dev_set}_pitch
    n=$(( $(wc -l < data/${src_org_set}_pitch/wav.scp) - 200 ))
    utils/subset_data_dir.sh --first data/${src_org_set}_pitch ${n} data/${src_train_set}_pitch
    rm -rf data/${src_org_set}_pitch_tmp

    # use pretrained model cmvn
    cmvn=$(find ${download_dir}/${pretrained_asr_model} -name "cmvn.ark" | head -n 1)

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${src_train_set}_pitch/feats.scp ${cmvn} exp/dump_feats/${src_train_set}_pitch ${src_feat_tr_pitch_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${src_dev_set}_pitch/feats.scp ${cmvn} exp/dump_feats/${src_dev_set}_pitch ${src_feat_dt_pitch_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${src_eval_set}_pitch/feats.scp ${cmvn} exp/dump_feats/${src_eval_set}_pitch ${src_feat_ev_pitch_dir}
fi

pair_tr_dir=${dumpdir}/${pair_train_set}; mkdir -p ${pair_tr_dir}
pair_dt_dir=${dumpdir}/${pair_dev_set}; mkdir -p ${pair_dt_dir}
pair_ev_dir=${dumpdir}/${pair_eval_set}; mkdir -p ${pair_ev_dir}
dict=data/lang_1char/units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation for ASR input"
    
    # dummy dict
    mkdir -p data/lang_1char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC

    # make json labels using pretrained model dict
    data2json.sh --feat ${src_feat_tr_pitch_dir}/feats.scp \
         data/${src_train_set}_pitch ${dict} > ${src_feat_tr_pitch_dir}/data.json
    data2json.sh --feat ${src_feat_dt_pitch_dir}/feats.scp \
         data/${src_dev_set}_pitch ${dict} > ${src_feat_dt_pitch_dir}/data.json
    data2json.sh --feat ${src_feat_ev_pitch_dir}/feats.scp \
         data/${src_eval_set}_pitch ${dict} > ${src_feat_ev_pitch_dir}/data.json

    # make pair json
    if [ ${num_train_utts} -ge 0 ]; then
        local/make_pair_json.py --src-json ${src_feat_tr_pitch_dir}/data.json \
             --trg-json ${trg_feat_tr_dir}/data.json -O ${pair_tr_dir}/data_n${num_train_utts}_pitch.json \
             --num_utts ${num_train_utts}
    else
        local/make_pair_json.py --src-json ${src_feat_tr_pitch_dir}/data.json \
             --trg-json ${trg_feat_tr_dir}/data.json -O ${pair_tr_dir}/data_pitch.json
    fi
    local/make_pair_json.py --src-json ${src_feat_dt_pitch_dir}/data.json \
         --trg-json ${trg_feat_dt_dir}/data.json -O ${pair_dt_dir}/data_pitch.json
    local/make_pair_json.py --src-json ${src_feat_ev_pitch_dir}/data.json \
         --trg-json ${trg_feat_ev_dir}/data.json -O ${pair_ev_dir}/data_pitch.json
fi

if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ]; then
    echo "stage 23: VC model training"

    # add pretrained model info in config
    if [ ! -z ${pretrained_dec_model} ]; then
        pretrained_dec_model_path=$(find ${pretrained_dec_model} -name "model*.best" | head -n 1)
        train_config="$(change_yaml.py -a pretrained-model="${pretrained_dec_model_path}" \
            -a load-partial-pretrained-model=encoder \
            -o "conf/$(basename "${train_config}" .yaml).$(basename "${pretrained_dec_model}").yaml" "${train_config}")"
    fi

    if [ ! -z ${pretrained_asr_model} ]; then
        pretrained_asr_model_path=$(find ${download_dir}/${pretrained_asr_model} -name "model*.best" | head -n 1)
        train_config="$(change_yaml.py -a pretrained-model2="${pretrained_asr_model_path}" \
            -a load-partial-pretrained-model2=encoder \
            -o "conf/$(basename "${train_config}" .yaml).${pretrained_asr_model}.yaml" "${train_config}")"
    fi

    if [ -z ${tag} ]; then
        expname=${srcspk}_${trgspk}_${backend}_$(basename ${train_config%.*})
    else
        expname=${srcspk}_${trgspk}_${backend}_${tag}
    fi
    expdir=exp/${expname}

    if [ ${num_train_utts} -ge 0 ]; then
        tr_json=${pair_tr_dir}/data_n${num_train_utts}_pitch.json
    else
        tr_json=${pair_tr_dir}/data_pitch.json
    fi
    dt_json=${pair_dt_dir}/data_pitch.json

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

if [ ${stage} -le 24 ] && [ ${stop_stage} -ge 24 ]; then
    echo "stage 24: Decoding, synthesis and HDF5 generation"
    pids=() # initialize pids

    if [ -z ${tag} ]; then
        expname=${srcspk}_${trgspk}_${backend}_$(basename ${train_config%.*})
    else
        expname=${srcspk}_${trgspk}_${backend}_${tag}
    fi
    expdir=exp/${expname}
    outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})

    for name in ${pair_dev_set} ${pair_eval_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}/data_pitch.json ${outdir}/${name}
        splitjson.py --parts ${nj} ${outdir}/${name}/data_pitch.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            vc_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${name}/feats.JOB \
                --json ${outdir}/${name}/split${nj}utt/data_pitch.JOB.json \
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
    
    echo "Synthesis"
    pids=() # initialize pids
    for name in ${pair_dev_set} ${pair_eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
        # use pretrained model cmvn
        apply-cmvn --norm-vars=true --reverse=true ${pretrained_cmvn} \
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
    
    echo "Generate hdf5"
    # generate h5 for WaveNet vocoder
    for name in ${pair_dev_set} ${pair_eval_set}; do
        feats2hdf5.py \
            --scp_file ${outdir}_denorm/${name}/feats.scp \
            --out_dir ${outdir}_denorm/${name}/hdf5/
        (find "$(cd ${outdir}_denorm/${name}/hdf5; pwd)" -name "*.h5" -print &) | head > ${outdir}_denorm/${name}/hdf5_feats.scp
        echo "generated hdf5 for ${name} set"
    done
fi

#################################################################

if [ ${stage} -le 51 ] && [ ${stop_stage} -ge 51 ]; then
    echo "stage 51: Ground Truth Objective Evaluation"

    for set_type in dev eval; do

        local/ob_eval/evaluate.sh --nj ${nj} \
            --do_delta false \
            --eval_model ${eval_model} \
            --db_root ${db_root} \
            --backend pytorch \
            --wer ${wer} \
            --mcd false \
            --vocoder None \
            --api v2 \
            ${asr_model} \
            ${outdir} \
            ${srcspk}_${trgspk}_${set_type} \
            ${trgspk}_${set_type} \
            ${trgspk}
    done
fi
