#!/usr/bin/env bash

# model_id="outputs/big/wav2vec2-FR-7K-large-ft"
# model_id="outputs/big/wav2vec2-FR-7K-large-ft-augment"
model_id="outputs/big/wav2vec2-FR-7K-large-ft-augment-bs256-lr1e4"
lm_arpa_path="outputs/lm/general/lm_5gram.arpa"
lm_bin_path="outputs/lm/general/lm_5gram.bin"

# model_id="outputs/hmhm_merged_and_raw/wav2vec2-FR-7K-large_ft_ep30"
# lm_arpa_path="outputs/lm/hmhm/lm_hm_hm_merged.arpa"
# lm_bin_path="outputs/lm/hmhm/lm_hm_hm_merged.bin"
# lm_arpa_path="outputs/lm/hmhm/lm_tgsmall.arpa"
# lm_bin_path="outputs/lm/hmhm/lm_tgsmall.bin"
# lm_arpa_path="outputs/lm/carglass/lm_3gram_hmhm_carglass_merged.arpa"
# lm_bin_path="outputs/lm/carglass/lm_3gram_hmhm_carglass_merged.bin"
# lm_arpa_path="outputs/lm/dekuple/lm_5gram_lv_hm_w2v_script_v2.arpa"
# lm_bin_path="outputs/lm/dekuple/lm_5gram_lv_hm_w2v_script_v2.bin"

outdir=${model_id}_with_lm
# outdir=${model_id}_with_hmhm_lm
# outdir=${model_id}_with_dekuple_lm
# outdir=${model_id}_with_carglass_lm

rsync -av --progress ${model_id}/ $outdir --exclude "checkpoint-*" --exclude "*results.json" --exclude "training_args.bin" --exclude "trainer_state.json"

python scripts/create_wav2vec2_processor_with_lm.py \
    --model_id $model_id \
    --arpa_path $lm_arpa_path \
    --outdir $outdir

cp $lm_bin_path $outdir/language_model
lm_arpa_filename=$(basename -- "$lm_arpa_path")
rm $outdir/language_model/$lm_arpa_filename
