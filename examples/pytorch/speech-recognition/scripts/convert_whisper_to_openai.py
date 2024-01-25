#!/usr/bin/env python
# Copyright 2022  Bofeng Huang
# coding=utf-8

"""
Usage:
./scripts/convert_whisper_to_openai.py \
    --hf_model_name_or_path bofenghuang/whisper-large-v3-french \
    --whisper_state_path models/bofenghuang/whisper-large-v3-french/model_openai.pt \
    --torch_dtype float16
"""

from copy import deepcopy

import fire
import torch

from transformers import WhisperForConditionalGeneration

WHISPER_MAPPING = {
    "layers": "blocks",
    "fc1": "mlp.0",
    "fc2": "mlp.2",
    "final_layer_norm": "mlp_ln",
    "layers": "blocks",
    ".self_attn.q_proj": ".attn.query",
    ".self_attn.k_proj": ".attn.key",
    ".self_attn.v_proj": ".attn.value",
    ".self_attn_layer_norm": ".attn_ln",
    ".self_attn.out_proj": ".attn.out",
    ".encoder_attn.q_proj": ".cross_attn.query",
    ".encoder_attn.k_proj": ".cross_attn.key",
    ".encoder_attn.v_proj": ".cross_attn.value",
    ".encoder_attn_layer_norm": ".cross_attn_ln",
    ".encoder_attn.out_proj": ".cross_attn.out",
    "decoder.layer_norm.": "decoder.ln.",
    "encoder.layer_norm.": "encoder.ln_post.",
    "embed_tokens": "token_embedding",
    "encoder.embed_positions.weight": "encoder.positional_embedding",
    "decoder.embed_positions.weight": "decoder.positional_embedding",
    "layer_norm": "ln_post",
}


def rename_keys(s_dict):
    keys = list(s_dict.keys())
    for key in keys:
        new_key = key
        for k, v in WHISPER_MAPPING.items():
            if k in key:
                new_key = new_key.replace(k, v)

        print(f"{key} -> {new_key}")

        s_dict[new_key] = s_dict.pop(key)
    return s_dict


def convert_hf_whisper(hf_model_name_or_path: str, whisper_state_path: str, torch_dtype: str = "float32"):
    # NB: smaller in fp16
    transformer_model = WhisperForConditionalGeneration.from_pretrained(
        hf_model_name_or_path, torch_dtype=getattr(torch, torch_dtype)
    )
    print(transformer_model.dtype)

    """
    transformer_model.save_pretrained(
        "/projects/bhuang/models/asr/public/whisper-large-v3-french/pytorch_model",
        max_shard_size="10GB",
        safe_serialization=False,
    )
    quit()
    """

    config = transformer_model.config

    # first build dims
    dims = {
        "n_mels": config.num_mel_bins,
        "n_vocab": config.vocab_size,
        "n_audio_ctx": config.max_source_positions,
        "n_audio_state": config.d_model,
        "n_audio_head": config.encoder_attention_heads,
        "n_audio_layer": config.encoder_layers,
        "n_text_ctx": config.max_target_positions,
        "n_text_state": config.d_model,
        "n_text_head": config.decoder_attention_heads,
        "n_text_layer": config.decoder_layers,
    }

    state_dict = deepcopy(transformer_model.model.state_dict())
    state_dict = rename_keys(state_dict)

    torch.save({"dims": dims, "model_state_dict": state_dict}, whisper_state_path)


if __name__ == "__main__":
    fire.Fire(convert_hf_whisper)
