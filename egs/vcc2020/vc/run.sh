#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
# [stage 6] 2019 Okayama University (Katsuki Inoue)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1
stop_stage=100
ngpu=1       # number of gpus ("0" uses cpu, otherwise use gpu)
nj=10        # numebr of parallel jobs
dumpdir=dump # directory to dump full features
verbose=0    # verbose option (if set > 0, get more log)
N=0          # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=16000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length

# char or phn
# In the case of phn, input transcription is convered to phoneem using https://github.com/Kyubyong/g2p.
trans_type="char"

# config files
train_config=conf/train_pytorch_tacotron2.yaml # you can select from conf or conf/tuning.
                                               # now we support tacotron2, transformer, and fastspeech
                                               # see more info in the header of each config.
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
n_average=1 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# objective evaluation related
asr_model="librispeech.transformer.ngpu4"
eval_tts_model=true                            # true: evaluate tts model, false: evaluate ground truth
wer=true                                       # true: evaluate CER & WER, false: evaluate only CER

# dataset configuration
available_spks=(
    "SEF1" "SEF2" "SEM1" "SEM2" "TEF1" "TEF2" "TEM1" "TEM2" "TFF1" "TFM1" "TGF1" "TGM1" "TMF1" "TMM1"
)
available_langs=(
    "Eng" "Ger" "Fin" "Man" 
)
srcspk=SEF1
srclang=Eng
trgspk=TMF1
trglang=Man
num_train_utts=-1
aespk=

# pretrained model related
pretrained_asr_model=li10_transformer
pretrained_dec_model=
pretrained_enc_model=

# pseudo parallel data for nonparallel training
src_decoded_feat_tr=                   # denormalized
trg_decoded_feat_tr=                   # denormalized
src_decoded_feat_dt=                   # denormalized
trg_decoded_feat_dt=                   # denormalized
pseudo_data_tag=

# objective evaluation related
asr_model="librispeech.transformer.ngpu4"
eval_model=true                                # true: evaluate trained model, false: evaluate ground truth
wer=true                                       # true: evaluate CER & WER, false: evaluate only CER
vocoder=                                       # select vocoder type (GL, WNV, PWG)

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
src_pseudo_train_set=${srcspk}_pseudo_train_${trglang}
src_pseudo_dev_set=${srcspk}_pseudo_dev_${trglang}
trg_pseudo_train_set=${trgspk}_pseudo_train_${srclang}
trg_pseudo_dev_set=${trgspk}_pseudo_dev_${srclang}

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"

    # TODO: write dataset download
    #local/download.sh ${db_root}

    # asr pretrained model download
    local/pretrained_model_download.sh ${db_root} ${pretrained_asr_model}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"

    for spk in "${available_spks[@]}"
    do
        org_set=${spk}
        train_set=${spk}_train_no_dev
        dev_set=${spk}_dev
        eval_set=${spk}_eval
        local/data_prep.sh ${db_root} ${spk} data/${org_set} ${trans_type}
        utils/data/resample_data_dir.sh ${fs} data/${org_set} # Downsample to fs from 24k
        utils/validate_data_dir.sh --no-feats data/${org_set}
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev name by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    # Generate the fbank features; by default 80-dimensional fbanks on each frame
    for spk in "${available_spks[@]}"
    do
        echo "==== Generating $spk"
        org_set=${spk}
        train_set=${spk}_train_no_dev
        dev_set=${spk}_dev
        eval_set=${spk}_eval
        feat_tr_pitch_dir=${dumpdir}/${train_set}_pitch; mkdir -p ${feat_tr_pitch_dir}
        feat_dt_pitch_dir=${dumpdir}/${dev_set}_pitch; mkdir -p ${feat_dt_pitch_dir}
        feat_ev_pitch_dir=${dumpdir}/${eval_set}_pitch; mkdir -p ${feat_ev_pitch_dir}
        feat_tr_nopitch_dir=${dumpdir}/${train_set}_nopitch; mkdir -p ${feat_tr_nopitch_dir}
        feat_dt_nopitch_dir=${dumpdir}/${dev_set}_nopitch; mkdir -p ${feat_dt_nopitch_dir}
        feat_ev_nopitch_dir=${dumpdir}/${eval_set}_nopitch; mkdir -p ${feat_ev_nopitch_dir}
        fbankdir=fbank
        lang_char=$(echo ${spk} | head -c 2 | tail -c 1)
        
        # Input: use pretrained ASR model cmvn
        input_cmvn=$(find ${db_root}/${pretrained_asr_model} -name "cmvn.ark" | head -n 1)
        # Output: use pretrained TTS model cmvn (language dependent)
        case "${lang_char}" in
            "E") output_cmvn=$(find ${db_root}/E_mailabs_judy -name "cmvn.ark" | head -n 1) ;;
            "M") output_cmvn=$(find ${db_root}/M_csmsc -name "cmvn.ark" | head -n 1) ;;
            *) echo "We don't have a pretrained model for this language now."; continue ;;
        esac
        echo $input_cmvn
        echo $output_cmvn
        
        # make train/dev/test set
        utils/subset_data_dir.sh --utt-list ${db_root}/lists/eval_list.txt data/${org_set} data/${eval_set}_pitch
        utils/subset_data_dir.sh --utt-list ${db_root}/lists/${lang_char}_train_list.txt data/${org_set} data/${train_set}_pitch
        utils/subset_data_dir.sh --utt-list ${db_root}/lists/${lang_char}_dev_list.txt data/${org_set} data/${dev_set}_pitch
        utils/subset_data_dir.sh --utt-list ${db_root}/lists/eval_list.txt data/${org_set} data/${eval_set}_nopitch
        utils/subset_data_dir.sh --utt-list ${db_root}/lists/${lang_char}_train_list.txt data/${org_set} data/${train_set}_nopitch
        utils/subset_data_dir.sh --utt-list ${db_root}/lists/${lang_char}_dev_list.txt data/${org_set} data/${dev_set}_nopitch

        # Generate the fbank+pitch features; by default 80-dimensional fbanks on each frame
        for x in ${train_set} ${dev_set} ${eval_set}; do
            steps/make_fbank_pitch.sh --cmd "${train_cmd}" --nj 10 \
                --write_utt2num_frames true \
                data/${x}_pitch \
                exp/make_fbank/${x}_pitch \
                ${fbankdir}
            utils/fix_data_dir.sh data/${x}_pitch
        done

        for x in ${train_set} ${dev_set} ${eval_set}; do
        # Generate the fbank features; by default 80-dimensional fbanks on each frame
            make_fbank.sh --cmd "${train_cmd}" --nj 10 \
                --fs ${fs} \
                --fmax "${fmax}" \
                --fmin "${fmin}" \
                --n_fft ${n_fft} \
                --n_shift ${n_shift} \
                --win_length "${win_length}" \
                --n_mels ${n_mels} \
                data/${x}_nopitch \
                exp/make_fbank/${x}_nopitch \
                ${fbankdir}
        done
        
        # dump features
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
            data/${train_set}_pitch/feats.scp ${input_cmvn} exp/dump_feats/${train_set}_pitch ${feat_tr_pitch_dir}
        dump.sh --cmd "$train_cmd" --nj 10 --do_delta false \
            data/${dev_set}_pitch/feats.scp ${input_cmvn} exp/dump_feats/${dev_set}_pitch ${feat_dt_pitch_dir}
        dump.sh --cmd "$train_cmd" --nj 25 --do_delta false \
            data/${eval_set}_pitch/feats.scp ${input_cmvn} exp/dump_feats/${eval_set}_pitch ${feat_ev_pitch_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
            data/${train_set}_nopitch/feats.scp ${output_cmvn} exp/dump_feats/${train_set}_nopitch ${feat_tr_nopitch_dir}
        dump.sh --cmd "$train_cmd" --nj 10 --do_delta false \
            data/${dev_set}_nopitch/feats.scp ${output_cmvn} exp/dump_feats/${dev_set}_nopitch ${feat_dt_nopitch_dir}
        dump.sh --cmd "$train_cmd" --nj 25 --do_delta false \
            data/${eval_set}_nopitch/feats.scp ${output_cmvn} exp/dump_feats/${eval_set}_nopitch ${feat_ev_nopitch_dir}
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"

    dict=data/lang_1${trans_type}/train_no_dev_units.txt
    echo "dictionary: ${dict}"
   
    # dummy dict
    mkdir -p data/lang_1${trans_type}/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    #text2token.py -s 1 -n 1 --trans_type ${trans_type} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    #| sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    #wc -l ${dict}

    # make json labels

    # use make_ae_json.py from arctic/vc? 
    for spk in "${available_spks[@]}"
    do
        echo "==== Generating $spk"
        org_set=${spk}
        train_set=${spk}_train_no_dev
        dev_set=${spk}_dev
        eval_set=${spk}_eval
        feat_tr_pitch_dir=${dumpdir}/${train_set}_pitch
        feat_dt_pitch_dir=${dumpdir}/${dev_set}_pitch
        feat_ev_pitch_dir=${dumpdir}/${eval_set}_pitch
        feat_tr_nopitch_dir=${dumpdir}/${train_set}_nopitch
        feat_dt_nopitch_dir=${dumpdir}/${dev_set}_nopitch
        feat_ev_nopitch_dir=${dumpdir}/${eval_set}_nopitch
        feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
        feat_dt_dir=${dumpdir}/${dev_set}; mkdir -p ${feat_dt_dir}
        feat_ev_dir=${dumpdir}/${eval_set}; mkdir -p ${feat_ev_dir}
        lang_char=$(echo ${spk} | head -c 2 | tail -c 1)
        
        case "${lang_char}" in
            "E")  ;;
            "M")  ;;
            *) echo "Skip this language now."; continue ;;
        esac

        data2json.sh --feat ${feat_tr_pitch_dir}/feats.scp --trans_type ${trans_type} \
             data/${train_set}_pitch ${dict} > ${feat_tr_pitch_dir}/data.json
        data2json.sh --feat ${feat_dt_pitch_dir}/feats.scp --trans_type ${trans_type} \
             data/${dev_set}_pitch ${dict} > ${feat_dt_pitch_dir}/data.json
        data2json.sh --feat ${feat_ev_pitch_dir}/feats.scp --trans_type ${trans_type} \
             data/${eval_set}_pitch ${dict} > ${feat_ev_pitch_dir}/data.json
        data2json.sh --feat ${feat_tr_nopitch_dir}/feats.scp --trans_type ${trans_type} \
             data/${train_set}_nopitch ${dict} > ${feat_tr_nopitch_dir}/data.json
        data2json.sh --feat ${feat_dt_nopitch_dir}/feats.scp --trans_type ${trans_type} \
             data/${dev_set}_nopitch ${dict} > ${feat_dt_nopitch_dir}/data.json
        data2json.sh --feat ${feat_ev_nopitch_dir}/feats.scp --trans_type ${trans_type} \
             data/${eval_set}_nopitch ${dict} > ${feat_ev_nopitch_dir}/data.json

        # make ae json, using 83-d input and 80-d output
        local/make_ae_json.py --input_json ${feat_tr_pitch_dir}/data.json \
            --output_json ${feat_tr_nopitch_dir}/data.json -O ${feat_tr_dir}/ae_data.json
        local/make_ae_json.py --input_json ${feat_dt_pitch_dir}/data.json \
            --output_json ${feat_dt_nopitch_dir}/data.json -O ${feat_dt_dir}/ae_data.json
        local/make_ae_json.py --input_json ${feat_ev_pitch_dir}/data.json \
            --output_json ${feat_ev_nopitch_dir}/data.json -O ${feat_ev_dir}/ae_data.json

     done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Target speaker dependent decoder training"
    
    # check input arguments
    if [ -z ${tag} ]; then
        echo "Please specify exp tag."
        exit 1
    fi
    if [ -z ${pretrained_asr_model} ]; then
        echo "Please specify pretrained asr model."
        exit 1
    fi

    # decide pretrained TTS model
    lang_char=$(echo ${trgspk} | head -c 2 | tail -c 1)
    case "${lang_char}" in
        "E") pretrained_tts_model=E_mailabs_judy ;;
        "M") pretrained_tts_model=M_csmsc ;;
        *) echo "We don't have a pretrained TTS model for this language now."; continue ;;
    esac
    
    trg_tr_dir=${dumpdir}/${trg_train_set}
    trg_dt_dir=${dumpdir}/${trg_dev_set}
    trg_ev_dir=${dumpdir}/${trg_eval_set}

    expname=${trg_train_set}_${tag}
    expdir=exp/${expname}
    mkdir -p ${expdir}
    
    # add pretrained model info in config
    pretrained_tts_model_path=$(find ${db_root}/${pretrained_tts_model} -name "model*.best" | head -n 1)
    pretrained_asr_model_path=$(find ${db_root}/${pretrained_asr_model} -name "model*.best" | head -n 1)
    train_config="$(change_yaml.py \
        -a pretrained-model="${pretrained_tts_model_path}" \
        -a pretrained-model2="${pretrained_asr_model_path}" \
        -a load-partial-pretrained-model=encoder \
        -a load-partial-pretrained-model2=encoder \
        -o "conf/${trgspk}.$(basename "${train_config}" .yaml).${pretrained_asr_model}.${pretrained_tts_model}.yaml" "${train_config}")"

    tr_json=${trg_tr_dir}/ae_data.json
    dt_json=${trg_dt_dir}/ae_data.json
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
           --srcspk ${trgspk} \
           --trgspk ${trgspk} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config}
fi





#if [ ${n_average} -gt 0 ]; then
#    model=model.last${n_average}.avg.best
#fi
#outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Decoding"
    if [ ${n_average} -gt 0 ]; then
        average_checkpoints.py --backend ${backend} \
                               --snapshots ${expdir}/results/snapshot.ep.* \
                               --out ${expdir}/results/${model} \
                               --num ${n_average}
    fi
    pids=() # initialize pids
    for name in ${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}/data.json ${outdir}/${name}
        splitjson.py --parts ${nj} ${outdir}/${name}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            tts_decode.py \
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
    echo "Finished."
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Objective Evaluation"

    for name in ${dev_set} ${eval_set}; do
        local/ob_eval/evaluate_cer.sh --nj ${nj} \
            --do_delta false \
            --eval_tts_model ${eval_tts_model} \
            --db_root ${db_root} \
            --backend pytorch \
            --wer ${wer} \
            --api v2 \
            ${asr_model} \
            ${outdir} \
            ${name}
    done

    echo "Finished."
fi


#########################################################################

pair_tr_dir=${dumpdir}/${pair_train_set}; mkdir -p ${pair_tr_dir}
pair_dt_dir=${dumpdir}/${pair_dev_set}; mkdir -p ${pair_dt_dir}
pair_ev_dir=${dumpdir}/${pair_eval_set}; mkdir -p ${pair_ev_dir}
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 11: Json Data Preparation"
    
    src_train_small_set=${srcspk}_train_small
    trg_train_small_set=${trgspk}_train_small
    src_tr_nopitch_dir=${dumpdir}/${src_train_small_set}_nopitch; mkdir -p ${src_tr_nopitch_dir}
    trg_tr_nopitch_dir=${dumpdir}/${trg_train_small_set}_nopitch; mkdir -p ${trg_tr_nopitch_dir}

    src_lang_char=$(echo ${srcspk} | head -c 2 | tail -c 1)
    trg_lang_char=$(echo ${trgspk} | head -c 2 | tail -c 1)

    # make smalle parallel training set
    utils/subset_data_dir.sh --utt-list ${db_root}/lists/${src_lang_char}_train_list_small.txt \
        data/${src_train_set}_nopitch data/${src_train_small_set}_nopitch
    utils/subset_data_dir.sh --utt-list ${db_root}/lists/${trg_lang_char}_train_list_small.txt \
        data/${trg_train_set}_nopitch data/${trg_train_small_set}_nopitch
        
    case "${src_lang_char}" in
        "E") src_cmvn=$(find ${db_root}/E_mailabs_judy -name "cmvn.ark" | head -n 1) ;;
        "M") src_cmvn=$(find ${db_root}/M_csmsc -name "cmvn.ark" | head -n 1) ;;
        *) echo "We don't have a pretrained model for this language now."; continue ;;
    esac
    dump.sh --cmd "$train_cmd" --nj 10 --do_delta false \
            data/${src_train_small_set}_nopitch/feats.scp ${src_cmvn} exp/dump_feats/${src_train_small_set}_nopitch ${src_tr_nopitch_dir}
    case "${trg_lang_char}" in
        "E") trg_cmvn=$(find ${db_root}/E_mailabs_judy -name "cmvn.ark" | head -n 1) ;;
        "M") trg_cmvn=$(find ${db_root}/M_csmsc -name "cmvn.ark" | head -n 1) ;;
        *) echo "We don't have a pretrained model for this language now."; continue ;;
    esac
    dump.sh --cmd "$train_cmd" --nj 10 --do_delta false \
            data/${trg_train_small_set}_nopitch/feats.scp ${trg_cmvn} exp/dump_feats/${trg_train_small_set}_nopitch ${trg_tr_nopitch_dir}

    exit 1

    # make json labels using pretrained model dict
    data2json.sh --feat ${src_tr_nopitch_dir}/feats.scp \
         data/${src_train_set} ${dict} > ${src_feat_tr_dir}/data.json
    
    data2json.sh --feat ${trg_feat_tr_dir}/feats.scp \
         data/${trg_train_set} ${dict} > ${trg_feat_tr_dir}/data.json

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

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    echo "stage 12: parallel VC model training"

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

#########################################################################

if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
    echo "stage 15: Json Data Preparation"
    
    cmvn=$(find ${db_root}/${pretrained_dec_model} -name "cmvn.ark" | head -n 1)
    dict=data/lang_1${trans_type}/train_no_dev_units.txt

    # normalize the features first
    for spk in ${srcspk} ${trgspk};
    do
        org_set=${spk}
        train_set=${spk}_train_no_dev
        dev_set=${spk}_dev
        eval_set=${spk}_eval
        feat_tr_nopitch_dir=${dumpdir}/${train_set}_nopitch_${pretrained_dec_model}; mkdir -p ${feat_tr_nopitch_dir}
        feat_dt_nopitch_dir=${dumpdir}/${dev_set}_nopitch_${pretrained_dec_model}; mkdir -p ${feat_dt_nopitch_dir}
        feat_ev_nopitch_dir=${dumpdir}/${eval_set}_nopitch_${pretrained_dec_model}; mkdir -p ${feat_ev_nopitch_dir}
       
        #if [ ! -f ${feat_tr_nopitch_dir}/feats.scp ]; then
            dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
                data/${train_set}_nopitch/feats.scp ${cmvn} \
                exp/dump_feats/${train_set}_nopitch_${pretrained_dec_model} \
                ${feat_tr_nopitch_dir}
        #fi
        
        #if [ ! -f ${feat_dt_nopitch_dir}/feats.scp ]; then
            dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
                data/${dev_set}_nopitch/feats.scp ${cmvn} \
                exp/dump_feats/${dev_set}_nopitch_${pretrained_dec_model} \
                ${feat_dt_nopitch_dir}
        #fi

        #if [ ! -f ${feat_ev_nopitch_dir}/feats.scp ]; then
            dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
                data/${eval_set}_nopitch/feats.scp ${cmvn} \
                exp/dump_feats/${eval_set}_nopitch_${pretrained_dec_model} \
                ${feat_ev_nopitch_dir}
        #fi
    done

    echo "### Pseudo ###"
    src_feat_pseudo_tr_nopitch_dir=${dumpdir}/${src_pseudo_train_set}_nopitch_${pretrained_dec_model}; mkdir -p ${src_feat_pseudo_tr_nopitch_dir}
    src_feat_pseudo_dt_nopitch_dir=${dumpdir}/${src_pseudo_dev_set}_nopitch_${pretrained_dec_model}; mkdir -p ${src_feat_pseudo_dt_nopitch_dir}
    trg_feat_pseudo_tr_nopitch_dir=${dumpdir}/${trg_pseudo_train_set}_nopitch_${pretrained_dec_model}; mkdir -p ${trg_feat_pseudo_tr_nopitch_dir}
    trg_feat_pseudo_dt_nopitch_dir=${dumpdir}/${trg_pseudo_dev_set}_nopitch_${pretrained_dec_model}; mkdir -p ${trg_feat_pseudo_dt_nopitch_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        ${src_decoded_feat_tr} ${cmvn} \
        exp/dump_feats/${src_pseudo_train_set}_nopitch_${pretrained_dec_model} \
        ${src_feat_pseudo_tr_nopitch_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        ${src_decoded_feat_dt} ${cmvn} \
        exp/dump_feats/${src_pseudo_dev_set}_nopitch_${pretrained_dec_model} \
        ${src_feat_pseudo_dt_nopitch_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        ${trg_decoded_feat_tr} ${cmvn} \
        exp/dump_feats/${trg_pseudo_train_set}_nopitch_${pretrained_dec_model} \
        ${trg_feat_pseudo_tr_nopitch_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        ${trg_decoded_feat_dt} ${cmvn} \
        exp/dump_feats/${trg_pseudo_dev_set}_nopitch_${pretrained_dec_model} \
        ${trg_feat_pseudo_dt_nopitch_dir}

    src_feat_tr_nopitch_dir=${dumpdir}/${src_train_set}_nopitch_${pretrained_dec_model}
    src_feat_dt_nopitch_dir=${dumpdir}/${src_dev_set}_nopitch_${pretrained_dec_model}
    src_feat_ev_nopitch_dir=${dumpdir}/${src_eval_set}_nopitch_${pretrained_dec_model}
    trg_feat_tr_nopitch_dir=${dumpdir}/${trg_train_set}_nopitch_${pretrained_dec_model}
    trg_feat_dt_nopitch_dir=${dumpdir}/${trg_dev_set}_nopitch_${pretrained_dec_model}
    trg_feat_ev_nopitch_dir=${dumpdir}/${trg_eval_set}_nopitch_${pretrained_dec_model}
    # make json labels for natural data
    data2json.sh --feat ${src_feat_tr_nopitch_dir}/feats.scp \
         data/${src_train_set}_nopitch ${dict} > ${src_feat_tr_nopitch_dir}/data.json
    data2json.sh --feat ${src_feat_dt_nopitch_dir}/feats.scp \
         data/${src_dev_set}_nopitch ${dict} > ${src_feat_dt_nopitch_dir}/data.json
    data2json.sh --feat ${src_feat_ev_nopitch_dir}/feats.scp \
         data/${src_eval_set}_nopitch ${dict} > ${src_feat_ev_nopitch_dir}/data.json
    data2json.sh --feat ${trg_feat_tr_nopitch_dir}/feats.scp \
         data/${trg_train_set}_nopitch ${dict} > ${trg_feat_tr_nopitch_dir}/data.json
    data2json.sh --feat ${trg_feat_dt_nopitch_dir}/feats.scp \
         data/${trg_dev_set}_nopitch ${dict} > ${trg_feat_dt_nopitch_dir}/data.json
    data2json.sh --feat ${trg_feat_ev_nopitch_dir}/feats.scp \
         data/${trg_eval_set}_nopitch ${dict} > ${trg_feat_ev_nopitch_dir}/data.json

    # make pair json
    local/make_pseudo_pair_json.py \
        --src_json ${src_feat_tr_nopitch_dir}/data.json \
        --trg_json ${trg_feat_tr_nopitch_dir}/data.json \
        --src_decoded_feat ${src_feat_pseudo_tr_nopitch_dir}/feats.scp \
        --trg_decoded_feat ${trg_feat_pseudo_tr_nopitch_dir}/feats.scp \
        -O ${pair_tr_dir}/data_${pseudo_data_tag}.json
    local/make_pseudo_pair_json.py \
        --src_json ${src_feat_dt_nopitch_dir}/data.json \
        --trg_json ${trg_feat_dt_nopitch_dir}/data.json \
        --src_decoded_feat ${src_feat_pseudo_dt_nopitch_dir}/feats.scp \
        --trg_decoded_feat ${trg_feat_pseudo_dt_nopitch_dir}/feats.scp \
        -O ${pair_dt_dir}/data_${pseudo_data_tag}.json
    
    # make normal pair json for eval
    local/make_pair_json.py \
        --src-json ${src_feat_ev_nopitch_dir}/data.json \
        --trg-json ${trg_feat_ev_nopitch_dir}/data.json \
        -O ${pair_ev_dir}/data.json

fi

if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ]; then
    echo "stage 16: parallel VC model training"

    # add pretrained model info in config
    if [ ! -z ${pretrained_dec_model} ]; then
        pretrained_dec_model_path=$(find ${db_root}/${pretrained_dec_model} -name "model*.best" | head -n 1)
        train_config="$(change_yaml.py -a pretrained-model="${pretrained_dec_model_path}" \
            -a load-partial-pretrained-model=encoder \
            -o "conf/$(basename "${train_config}" .yaml).${pretrained_dec_model}.yaml" "${train_config}")"
    fi

    if [ ! -z ${pretrained_enc_model} ]; then
        pretrained_enc_model_path=$(find ${db_root}/${pretrained_enc_model} -name "model*.best" | head -n 1)
        train_config="$(change_yaml.py -a pretrained-model2="${pretrained_enc_model_path}" \
            -a load-partial-pretrained-model2=encoder \
            -o "conf/$(basename "${train_config}" .yaml).${pretrained_enc_model}.yaml" "${train_config}")"
    fi

    if [ -z ${tag} ]; then
        expname=${srcspk}_${trgspk}_${backend}_$(basename ${train_config%.*})
    else
        expname=${srcspk}_${trgspk}_${backend}_${tag}
    fi
    expdir=exp/${expname}

    tr_json=${pair_tr_dir}/data_${pseudo_data_tag}.json
    dt_json=${pair_dt_dir}/data_${pseudo_data_tag}.json

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

if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ]; then
    echo "stage 17: Decoding, Synthesis and HDF5 Generation"
    
    # check cmvn
    if [ -z ${pretrained_dec_model} ]; then
        echo "Please specify --pretrain_dec_model to fetch cmvn for denormalizing."
        exit 1
    fi
    
    if [ -z ${tag} ]; then
        expname=${srcspk}_${trgspk}_${backend}_$(basename ${train_config%.*})
    else
        expname=${srcspk}_${trgspk}_${backend}_${tag}
    fi
    expdir=exp/${expname}
    outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})


    pids=() # initialize pids
    for name in ${pair_eval_set}; do
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
    
    echo "Synthesis"
    cmvn=$(find ${db_root}/${pretrained_dec_model} -name "cmvn.ark" | head -n 1)
    pids=() # initialize pids
    for name in ${pair_eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
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
    
    echo "generate hdf5"

    # generate h5 for WaveNet vocoder
    for name in ${pair_eval_set}; do
        feats2hdf5.py \
            --scp_file ${outdir}_denorm/${name}/feats.scp \
            --out_dir ${outdir}_denorm/${name}/hdf5/
        (find "$(cd ${outdir}_denorm/${name}/hdf5; pwd)" -name "*.h5" -print &) | head > ${outdir}_denorm/${name}/hdf5_feats.scp
        echo "generated hdf5 for ${name} set"
    done
    echo "Finished."
fi

if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ]; then
    echo "stage 18: Objective Evaluation"
    
    if [ -z ${tag} ]; then
        expname=${srcspk}_${trgspk}_${backend}_$(basename ${train_config%.*})
    else
        expname=${srcspk}_${trgspk}_${backend}_${tag}
    fi
    expdir=exp/${expname}
    outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})

    local/ob_eval/evaluate.sh --nj ${nj} \
        --do_delta false \
        --eval_model ${eval_model} \
        --db_root ${db_root} \
        --backend pytorch \
        --wer ${wer} \
        --vocoder ${vocoder} \
        --api v2 \
        ${asr_model} \
        ${outdir} \
        ${srcspk}_${trgspk}_eval \
        ${trgspk}_eval_nopitch \
        ${trgspk}

    echo "Finished."
fi


