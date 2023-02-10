#!/usr/bin/env python
# Copyright 2022  Bofeng Huang

import os

import pandas as pd
from tqdm import tqdm


csv_path = "data/allenai_soda-text/test.tsv"

df = pd.read_csv(csv_path, sep="\t")
print(df.head())

# debug
# df = df[:10]

data = []
for example_idx, example_df in tqdm(df.groupby("idx")):
    # print(example_idx)
    # print(example_df)

    summary = " ".join(example_df[example_df["type"] == "summary"]["sentence"])
    # print(summary)

    example_dialog_df = example_df[example_df["type"] == "dialogue"]

    dialogs = []
    for spk_turn_idx, spk_turn_df in example_dialog_df.groupby("spk_turn_idx"):
        speaker_name = spk_turn_df["speaker"].unique()[0]
        # print(speaker_name)

        new_spk_turn = speaker_name + ": " + " ".join(spk_turn_df["sentence"])
        # print(new_spk_turn)
        dialogs.append(new_spk_turn)

    dialog_text = "\n".join(dialogs)

    data.append(
        {
            "dialogue": dialog_text,
            "summary": summary,
        }
    )

new_df = pd.DataFrame(data)
print(new_df.shape)
print(new_df)

output_dir = "data/allenai_soda-fr"
# output_dir = "data/tmp"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

new_df.to_csv(f"{output_dir}/test.tsv", index=False, sep="\t")
