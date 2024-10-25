# coding=utf-8
# Copyright 2024  Bofeng Huang


import json
import os
from typing import Any

import numpy as np
from tqdm import tqdm


def write_dataset_to_json(
    dataset: Any,
    output_file_path: str,
    mode: str = "w",
    encoding: str = "utf-8",
    default: Any = str,
    ensure_ascii: bool = False,
) -> None:
    """Write dataset to a JSON file, line by line."""
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, mode, encoding=encoding) as fo:
        for sample in tqdm(dataset, desc="Writing to json", total=dataset.num_rows, unit=" samples"):
            # sample = {k: v for k, v in sample.items() if isinstance(v, (str, int, float))}
            fo.write(f"{json.dumps(sample, default=default, ensure_ascii=ensure_ascii)}\n")
    print(f"Processed manifest has been saved into {output_file_path}")


# fmt: off
def print_dataset_info(ds: Any, duration_column_name:str="duration"):
    print()
    print(f"#rows: {ds.num_rows}")
    print(f"Columns: {ds.column_names}")
    # ds_df = ds.to_pandas()
    durations = np.asarray(ds[duration_column_name])
    print(f"Duration statistics: tot {durations.sum() / 3600:.2f}h, mean {durations.mean():.2f}s, median {np.median(durations):.2f}s, min {durations.min():.2f}s, max {durations.max():.2f}s")
    print()
# fmt: on
