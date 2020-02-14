#!/bin/bash

# Copyright 2019 Tomoki Hayashi
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
verbose=1    # verbose option (if set > 0, get more log)
N=0          # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs_asr=16000        # sampling frequency
fs_tts=24000        # sampling frequency
fmax=7600       # maximum frequency
fmin=80         # minimum frequency
n_mels=80       # number of mel basis
n_fft=1024      # number of fft points
n_shift=256     # number of shift points
win_length=""   # window length

# config files
train_config=conf/tuning/train_pytorch_transformer.v1.single.yaml
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
n_average=1 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# root directory of db
db_root=downloads

# pretrained model related
pretrained_asr_model=../../vcc2020/vc/downloads/li10_transformer

tts_train_config=
pretrained_tts_model_path=
ae_train_config=conf/ae_R2R2.yaml

# objective evaluation related
asr_model="aishell.transformer"
eval_gt=false                                  # true: evaluate ground truth, false: evaluate tts model
eval_tts=true                                  # true: evaluate tts model, false: evaluate autoencoded speech
wer=true                                       # true: evaluate CER & WER, false: evaluate only CER

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_no_dev"
dev_set="dev"
eval_set="eval"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/data_download.sh ${db_root}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/data_prep.sh ${db_root}/CSMSC data/train

    # make copies for each fs
    [ -e  data/train_${fs_asr} ] && rm -r data/train_${fs_asr}
    [ -e  data/train_${fs_tts} ] && rm -r data/train_${fs_tts}
    cp -r data/train data/train_${fs_asr}
    cp -r data/train data/train_${fs_tts}

    # Downsample to fs from 48k (TODO: check if this only works for ASR, not TTS)
    utils/data/resample_data_dir.sh ${fs_asr} data/train_${fs_asr}
    utils/data/resample_data_dir.sh ${fs_tts} data/train_${fs_tts}

    utils/validate_data_dir.sh --no-feats data/train_${fs_asr}
    utils/validate_data_dir.sh --no-feats data/train_${fs_tts}
fi

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}; mkdir -p ${feat_dt_dir}
feat_ev_dir=${dumpdir}/${eval_set}; mkdir -p ${feat_ev_dir}
feat_tr_pitch_dir=${dumpdir}/${train_set}_pitch; mkdir -p ${feat_tr_pitch_dir}
feat_dt_pitch_dir=${dumpdir}/${dev_set}_pitch; mkdir -p ${feat_dt_pitch_dir}
feat_ev_pitch_dir=${dumpdir}/${eval_set}_pitch; mkdir -p ${feat_ev_pitch_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    fbankdir=fbank
    # Generate the fbank+pitch features; by default 80-dimensional fbanks on each frame
    steps/make_fbank_pitch.sh --cmd "${train_cmd}" --nj ${nj} \
        --write_utt2num_frames true \
        data/train_${fs_asr} \
        exp/make_fbank/${train_set}_pitch \
        ${fbankdir}
    utils/fix_data_dir.sh data/${train_set}_pitch
        
    # Generate the fbank features; by default 80-dimensional fbanks on each frame
    make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
        --fs ${fs_tts} \
        --fmax "${fmax}" \
        --fmin "${fmin}" \
        --n_fft ${n_fft} \
        --n_shift ${n_shift} \
        --win_length "${win_length}" \
        --n_mels ${n_mels} \
        data/train_${fs_tts} \
        exp/make_fbank/${train_set} \
        ${fbankdir}
    
    # make a dev set
    utils/subset_data_dir.sh --last data/train_${fs_tts} 200 data/deveval
    utils/subset_data_dir.sh --first data/deveval 100 data/${dev_set}
    utils/subset_data_dir.sh --last data/deveval 100 data/${eval_set}
    n=$(( $(wc -l < data/train/wav.scp) - 200 ))
    utils/subset_data_dir.sh --first data/train_${fs_tts} ${n} data/${train_set}
    
    utils/subset_data_dir.sh --last data/train_${fs_asr} 200 data/deveval_pitch
    utils/subset_data_dir.sh --first data/deveval_pitch 100 data/${dev_set}_pitch
    utils/subset_data_dir.sh --last data/deveval_pitch 100 data/${eval_set}_pitch
    n=$(( $(wc -l < data/train/wav.scp) - 200 ))
    utils/subset_data_dir.sh --first data/train_${fs_asr} ${n} data/${train_set}_pitch

    # compute statistics for global mean-variance normalization
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
    cmvn=data/${train_set}/cmvn.ark
    cmvn_asr=$(find ${pretrained_asr_model} -name "cmvn.ark" | head -n 1)
    #echo "cmvn:${cmvn}"

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp ${cmvn} exp/dump_feats/${train_set} ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}/feats.scp ${cmvn} exp/dump_feats/${dev_set} ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/feats.scp ${cmvn} exp/dump_feats/${eval_set} ${feat_ev_dir}
    
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}_pitch/feats.scp ${cmvn_asr} exp/dump_feats/${train_set}_pitch ${feat_tr_pitch_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}_pitch/feats.scp ${cmvn_asr} exp/dump_feats/${dev_set}_pitch ${feat_dt_pitch_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}_pitch/feats.scp ${cmvn_asr} exp/dump_feats/${eval_set}_pitch ${feat_ev_pitch_dir}
fi

dict=data/lang_phn/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_phn/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for padding idx
    text2token.py -s 1 -n 1 --trans_type phn data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --trans_type phn \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --trans_type phn \
         data/${dev_set} ${dict} > ${feat_dt_dir}/data.json
    data2json.sh --feat ${feat_ev_dir}/feats.scp --trans_type phn \
         data/${eval_set} ${dict} > ${feat_ev_dir}/data.json
    data2json.sh --feat ${feat_tr_pitch_dir}/feats.scp --trans_type phn \
         data/${train_set}_pitch ${dict} > ${feat_tr_pitch_dir}/data.json
    data2json.sh --feat ${feat_dt_pitch_dir}/feats.scp --trans_type phn \
         data/${dev_set}_pitch ${dict} > ${feat_dt_pitch_dir}/data.json
    data2json.sh --feat ${feat_ev_pitch_dir}/feats.scp --trans_type phn \
         data/${eval_set}_pitch ${dict} > ${feat_ev_pitch_dir}/data.json
    
    # make ae json, using 83-d input and 80-d output
    local/make_asr_ae_json.py --input_json ${feat_tr_pitch_dir}/data.json \
        --output_json ${feat_tr_dir}/data.json -O ${feat_tr_dir}/asr_ae_data.json
    local/make_asr_ae_json.py --input_json ${feat_dt_pitch_dir}/data.json \
        --output_json ${feat_dt_dir}/data.json -O ${feat_dt_dir}/asr_ae_data.json
    local/make_asr_ae_json.py --input_json ${feat_ev_pitch_dir}/data.json \
        --output_json ${feat_ev_dir}/data.json -O ${feat_ev_dir}/asr_ae_data.json
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Text-to-speech model training"

    if [ -z ${tag} ]; then
        expname=${train_set}_${backend}_$(basename ${train_config%.*})
    else
        expname=${train_set}_${backend}_${tag}
    fi
    expdir=exp/${expname}
    mkdir -p ${expdir}

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

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Decoding"

    outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})

    pids=() # initialize pids
    for sets in ${dev_set} ${eval_set}; do
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


    for sets in ${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${sets} ] && mkdir -p ${outdir}_denorm/${sets}
        apply-cmvn --norm-vars=true --reverse=true data/${train_set}/cmvn.ark \
            scp:${outdir}/${sets}/feats.scp \
            ark,scp:${outdir}_denorm/${sets}/feats.ark,${outdir}_denorm/${sets}/feats.scp
        convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
            --fs ${fs_tts} \
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
    
    if [ -z ${tag} ]; then
        expname=${train_set}_${backend}_$(basename ${train_config%.*})
    else
        expname=${train_set}_${backend}_${tag}
    fi
    expdir=exp/${expname}
    outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})

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
            csmsc
    done

    echo "Finished."
fi


#############################################################

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: encoder pretraining"

    # make pair json
    if [ ! -e ${feat_tr_dir}/ae_data.json ]; then
        echo "Making training json file for encoder pretraining"
        local/make_ae_json.py --input_json ${feat_tr_dir}/data.json --output_json ${feat_tr_dir}/data.json \
            -O ${feat_tr_dir}/ae_data.json
    fi
    if [ ! -e ${feat_dt_dir}/ae_data.json ]; then
        echo "Making development json file for encoder pretraining"
        local/make_ae_json.py --input_json ${feat_dt_dir}/data.json --output_json ${feat_dt_dir}/data.json \
            -O ${feat_dt_dir}/ae_data.json
    fi
    if [ ! -e ${feat_ev_dir}/ae_data.json ]; then
        echo "Making evaluation json file for encoder pretraining"
        local/make_ae_json.py --input_json ${feat_ev_dir}/data.json --output_json ${feat_ev_dir}/data.json \
            -O ${feat_ev_dir}/ae_data.json
    fi

    # check input arguments
    if [ -z ${tag} ]; then
        echo "Please specify exp tag."
        exit 1
    fi
    if [ -z ${pretrained_tts_model_path} ]; then
        echo "Please specify pre-trained tts model path."
        exit 1
    fi
    if [ -z ${tts_train_config} ]; then
        echo "Please specify pre-trained tts model config."
        exit 1
    fi

    expname=${train_set}_${backend}_${tag}
    expdir=exp/${expname}

    train_config="$(change_yaml.py -a pretrained-model="${pretrained_tts_model_path}" \
        -a load-partial-pretrained-model=encoder \
        -a params-to-train=encoder \
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
           --srcspk csmsc \
           --trgspk csmsc \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config} \
           --config2 ${ae_train_config}
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: Pretrained encoder decoding"
    
    expname=${train_set}_${backend}_${tag}
    outdir=${expdir}/ae_outputs_${model}

    pids=() # initialize pids
    for name in ${dev_set} ${eval_set}; do
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

    echo "Pretrained encoder synthesis"
    pids=() # initialize pids
    for name in ${dev_set} ${eval_set}; do
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
    
    echo "Pretrained encoder: generate hdf5"

    # generate h5 for WaveNet vocoder
    for name in ${dev_set} ${eval_set}; do
        feats2hdf5.py \
            --scp_file ${outdir}_denorm/${name}/feats.scp \
            --out_dir ${outdir}_denorm/${name}/hdf5/
        (find "$(cd ${outdir}_denorm/${name}/hdf5; pwd)" -name "*.h5" -print &) | head > ${outdir}_denorm/${name}/hdf5_feats.scp
        echo "generated hdf5 for ${name} set"
    done
fi

#############################################################

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    echo "stage 12: decoder pretraining using ASR encoder"

    # check input arguments
    if [ -z ${tag} ]; then
        echo "Please specify exp tag."
        exit 1
    fi
    if [ -z ${pretrained_asr_model} ]; then
        echo "Please specify pre-trained asr model."
        exit 1
    fi
    pretrained_asr_model_path=$(find ${pretrained_asr_model} -name "model*.best" | head -n 1)
    
    # here, pretrained_model is the TTS decoder, pretrained_model2 is the ASR encoder
    # add pretrained model info in config
    
    if [ -z ${pretrained_tts_model_path} ]; then
        train_config="$(change_yaml.py \
            -a pretrained-model2="${pretrained_asr_model_path}" \
            -a load-partial-pretrained-model2=encoder \
            -o "conf/$(basename "${train_config}" .yaml).$(basename ${pretrained_asr_model})" "${train_config}")"
    else
        train_config="$(change_yaml.py \
            -a pretrained-model="${pretrained_tts_model_path}" \
            -a load-partial-pretrained-model=encoder \
            -a pretrained-model2="${pretrained_asr_model_path}" \
            -a load-partial-pretrained-model2=encoder \
            -o "conf/$(basename "${train_config}" .yaml).$(basename ${pretrained_asr_model})" "${train_config}")"
    fi

    tr_json=${feat_tr_dir}/asr_ae_data.json
    dt_json=${feat_dt_dir}/asr_ae_data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/second_ae_train.log \
        vc_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --minibatches ${N} \
           --outdir ${expdir}/second_ae_results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --srcspk csmsc \
           --trgspk csmsc \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config}
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    echo "stage 13: Pretrained decoder: decoding, Synthesis, and hdf5 generation"

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
