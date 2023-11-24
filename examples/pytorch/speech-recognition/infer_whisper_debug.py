import os

os.environ["TRANSFORMERS_CACHE"] = "/projects/bhuang/.cache/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/projects/bhuang/.cache/huggingface/datasets"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from infer_whisper import main

main(
    model_name_or_path="openai/whisper-large-v3",
    output_file_path="tmp.json",
    dataset_file="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-13/test_asr_mcv13_manifest_normalized_pnc_head128.json",
    audio_column_name="audio_filepath",
    language="french",
    # return_timestamps=True,
)

