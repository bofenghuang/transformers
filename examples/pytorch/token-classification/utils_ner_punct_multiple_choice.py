from dataclasses import dataclass
from transformers import DataCollatorForTokenClassification

@dataclass
class DataCollatorForRecasePunct(DataCollatorForTokenClassification):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    def torch_call(self, features):
        import torch

        label_names = [k for k in features[0].keys() if k[-7:] == "_labels"]
        labels = {label_name: [feature[label_name] for feature in features] for label_name in label_names}

        # label_name = "label" if "label" in features[0].keys() else "labels"
        # labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            for label_name in label_names:
                batch[label_name] = [
                    list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels[label_name]
                ]
        else:
            for label_name in label_names:
                batch[label_name] = [
                    [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels[label_name]
                ]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch
