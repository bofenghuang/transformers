#!/usr/bin/env python3
from transformers import SpeechEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer, Wav2Vec2Processor
import torch


# encoder_id = "facebook/wav2vec2-base"
# decoder_id = "facebook/bart-base"
# out_dir = "./"

encoder_id = "LeBenchmark/wav2vec2-FR-7K-large"
# todo: gpt-fr, bloom, t5, barthez
decoder_id = 
out_dir = "./outputs/models/"

# load and save speech-encoder-decoder model
model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_id, decoder_id, encoder_add_adapter=True)
model.config.decoder_start_token_id = model.decoder.config.bos_token_id
model.config.pad_token_id = model.decoder.config.pad_token_id
model.config.eos_token_id = model.decoder.config.eos_token_id

# set some hyper-parameters for training and evaluation
model.config.use_cache = False
model.config.processor_class = "Wav2Vec2Processor"
model.config.encoder.feat_proj_dropout = 0.0
model.config.encoder.mask_time_prob = 0.0
model.config.encoder.layerdrop = 0.0
model.config.max_length = 40

# check if generation works
out = model.generate(torch.ones((1, 2000)))
# save model
model.save_pretrained(out_dir)

# load and save processor
feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id)
# feature_extractor.save_pretrained(out_dir)
tokenizer = AutoTokenizer.from_pretrained(decoder_id)
# tokenizer.save_pretrained(out_dir)
processor = Wav2Vec2Processor(feature_extractor, tokenizer)
processor.save_pretrained(out_dir)
