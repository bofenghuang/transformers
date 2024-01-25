#!/usr/bin/env python
# Copyright 2022  Bofeng Huang

from time import perf_counter

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import fire

from predict_ner_case_punc import CasePunctuationRestorationPipeline


def benchmark_inference_time(pipe, seq_len, iterations):
    tokenizer = pipe.tokenizer
    # input_ids = torch.randint(0, tokenizer.vocab_size, (seq_len,), dtype=torch.int32).unsqueeze(0)
    input_ids = torch.randint(0, tokenizer.vocab_size, (seq_len,), dtype=torch.int32)
    sentences = [tokenizer.decode(input_ids)]

    latencies_per_seq = []

    # Warm up
    # initialize the GPU and prevent it from going into power-saving mode
    for _ in range(10):
        pipe(sentences)

    # Timed run
    for _ in tqdm(range(iterations)):
        start_time = perf_counter()
        pipe(sentences)
        latency = 1000 * (perf_counter() - start_time)  # Unit: ms
        latencies_per_seq.append(latency)

    # Compute run statistics
    time_avg_ms_per_seq = np.mean(latencies_per_seq)
    time_median_ms_per_seq = np.median(latencies_per_seq)
    time_p95_ms_per_seq = np.percentile(latencies_per_seq, 95)

    # Record statistics for each iteration
    stat_dict = {
        "num_iter": iterations,
        "seq_len": seq_len,
        # "model_id": model_id,
        # "framework": framework,
        # "device": device,
        # "time_ms_per_seq": latencies_per_seq,
        "time_avg_ms_per_seq": time_avg_ms_per_seq,
        "time_p95_ms_per_seq": time_p95_ms_per_seq,
        "time_median_ms_per_seq": time_median_ms_per_seq,
    }

    return stat_dict


# recase_punct_predictor = CasePunctuationRestorationPipeline(
#     model_name_or_path,
#     device=0,
#     normalizer_file="./normalizer.json",
#     do_pre_normalize=True,
#     stride=100,
#     # batch_size=64,
# )

# sentence = "bonjour comment ca va j'aimerais savoir quelle est la réponse quelle était la question déjà d'accord effectivement le véhicule peugeot307 est plus abordable que le kia xceed"
# sentence = "j'aimerais savoir quelle est la réponse quelle était la question déjà d'accord"
# res = recase_punct_predictor(sentence)
# print(res)

# model_name_or_path = "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/recasepunc-multilingual_plus/xlm-roberta-large_ft"
# model_name_or_path = "recasepunc-multilingual_plus/xlm-roberta-large_ft"
model_names_or_paths = [
    "outputs/fr/xlm_roberta_large_trimmed_54k_ft_fr_ep5_bs256_lr3e5",
    "outputs/fr/xlm_roberta_base_trimmed_54k_ft_fr_ep5_bs256_lr3e5",
    "outputs/fr/xlm_roberta_base_trimmed_54k_ft_fr_ep5_bs256_lr3e5_clipped_max_spaced_6_layers_distil_ep5_bs256_lr3e5",
    "outputs/fr/mdeberta_v3_base_vocab78k_ft_fr_ep5_bs256_lr3e5",
    "outputs/fr/mdeberta_v3_base_vocab78k_ft_fr_ep5_bs256_lr3e5_clipped_max_spaced_6_layers_distil_ep5_bs256_lr3e5",
    "outputs/fr/mdeberta_v3_base_vocab78k_ft_fr_ep5_bs256_lr3e5_clipped_max_spaced_4_layers_distil_ep5_bs256_lr3e5",
    "outputs/fr/mdeberta_v3_base_vocab78k_ft_fr_ep5_bs256_lr3e5_clipped_max_spaced_2_layers_distil_ep5_bs256_lr3e5",
]

# seq_lengths = [8, 16, 32, 64, 128, 256, 512, 1024]
# seq_lengths = [8, 16]
seq_lengths = [32]

results = []
for model_name_or_path in model_names_or_paths:
    print("model:", model_name_or_path)
    recase_punc_predictor = CasePunctuationRestorationPipeline(
        model_name_or_path,
        # device=0,
        device="cpu",
        normalizer_file="./normalizer.json",
        do_pre_normalize=True,
        stride=100,
        # batch_size=64,
    )

    for seq_len in seq_lengths:
        print("seq_len:", seq_len)
        res = benchmark_inference_time(recase_punc_predictor, seq_len=seq_len, iterations=100)
        res["model_name"] = model_name_or_path.rsplit("/", 1)[-1]

        results.append(res)

df = pd.DataFrame(results)

df["time_avg_ms_per_seq"] = df["time_avg_ms_per_seq"].round(decimals=2)
df["time_p95_ms_per_seq"] = df["time_p95_ms_per_seq"].round(decimals=2)
df["time_median_ms_per_seq"] = df["time_median_ms_per_seq"].round(decimals=2)

print(df.head())

# df.to_csv("stat.csv", index=False)
df.to_csv("stat_cpu4.csv", index=False)
# df.to_pickle(f"t4_res_ort_gpt2_beam5_{seq_len}.pkl")
