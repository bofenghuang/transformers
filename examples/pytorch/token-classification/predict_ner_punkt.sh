#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

# ./predict_ner_punc_multiple_choice.py --model_name_or_path outputs/recasepunc-multilingual/xlm-roberta-large_ft --test_data_dir /projects/bhuang/corpus/text/flaubert/raw/fr_europarl/data/test --tmp_outdir outputs/recasepunc-multilingual/xlm-roberta-large_ft/results/fr/predict_words
# ./predict_ner_punc_multiple_choice.py --model_name_or_path outputs/recasepunc-multilingual/xlm-roberta-large_ft --test_data_dir /projects/bhuang/corpus/text/flaubert/raw/en_europarl/data/test --tmp_outdir outputs/recasepunc-multilingual/xlm-roberta-large_ft/results/en/predict_words
# ./predict_ner_punc_multiple_choice.py --model_name_or_path outputs/recasepunc-multilingual/xlm-roberta-large_ft --test_data_dir /projects/bhuang/corpus/text/flaubert/raw/de_europarl/data/test --tmp_outdir outputs/recasepunc-multilingual/xlm-roberta-large_ft/results/de/predict_words
# ./predict_ner_punc_multiple_choice.py --model_name_or_path outputs/recasepunc-multilingual/xlm-roberta-large_ft --test_data_dir /projects/bhuang/corpus/text/flaubert/raw/es_europarl/data/test --tmp_outdir outputs/recasepunc-multilingual/xlm-roberta-large_ft/results/es/predict_words


./predict_ner_punc_multiple_choice.py --model_name_or_path outputs/recasepunc-multilingual_plus/xlm-roberta-base_ft --test_data_dir /projects/bhuang/corpus/text/flaubert/raw/fr_europarl/data/test --tmp_outdir outputs/recasepunc-multilingual_plus/xlm-roberta-base_ft/results/fr/predict_words
./predict_ner_punc_multiple_choice.py --model_name_or_path outputs/recasepunc-multilingual_plus/xlm-roberta-large_ft --test_data_dir /projects/bhuang/corpus/text/flaubert/raw/fr_europarl/data/test --tmp_outdir outputs/recasepunc-multilingual_plus/xlm-roberta-large_ft/results/fr/predict_words


