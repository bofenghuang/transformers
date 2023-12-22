
# install dependencies
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# trannsformers
pip install -e .

pip install -U -qqq transformers accelerate datasets soundfile librosa jiwer evaluate wandb fire audiomentations
pip install deepspeed

# download models
python -c 'from transformers import AutoProcessor; processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")'
python -c 'from transformers import AutoModelForSpeechSeq2Seq; model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3")'
python -c 'import evaluate; metric = evaluate.load("wer")'

python -c 'from transformers import AutoProcessor; processor = AutoProcessor.from_pretrained("bofenghuang/whisper-large-v3-french")'
python -c 'from transformers import GenerationConfig; model = GenerationConfig.from_pretrained("bofenghuang/whisper-large-v3-french")'
python -c 'from transformers import AutoModelForSpeechSeq2Seq; model = AutoModelForSpeechSeq2Seq.from_pretrained("bofenghuang/whisper-large-v3-french")'

# wandb
wandb offline