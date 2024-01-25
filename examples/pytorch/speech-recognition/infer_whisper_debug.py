import os

os.environ["TRANSFORMERS_CACHE"] = "/projects/bhuang/.cache/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/projects/bhuang/.cache/huggingface/datasets"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["BITSANDBYTES_NOWELCOME"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from infer_whisper_c import main

main(
    model_name_or_path="/home/bhuang/transformers/examples/pytorch/speech-recognition/outputs/hmhm_merged_and_raw/bofenghuang-whisper_large_v2_french-ft-ep2-bs256-lr4e6-wd1e2-aug-specaug",
    sort_by_length=True,
    language="french",
    task="transcribe",
    generation_num_beams=1,
    per_device_eval_batch_size=4,
    dataloader_num_workers=4,
    num_processing_workers=16,
    dataset_file="/projects/corpus/voice/zaion/renault/2023-11-24/BO_max30s.json",
    id_column_name="id",
    audio_column_name="audio_filepath",
    start_column_name="offset",
    duration_column_name="duration",
    output_file_path="/home/bhuang/transformers/examples/pytorch/speech-recognition/tmp.json"
)
