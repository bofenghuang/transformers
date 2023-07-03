#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang


from transformers.integrations import WandbCallback


class WandbProgressResultsCallback(WandbCallback):
    def __init__(self, trainer, sample_dataset): 
        super().__init__()

        self.trainer = trainer
        self.sample_dataset = sample_dataset
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        super().on_log(args, state, control, model, logs)

        predictions = self.trainer.predict(self.sample_dataset)

        

        predictions = decode_predictions(self.trainer, predictions)
        measures_df = compute_measures(predictions, self.records_df["sentence"].tolist())
        records_df = pd.concat([self.records_df, measures_df], axis=1)
        records_df["prediction"] = predictions
        records_df["step"] = state.global_step
        records_table = self._wandb.Table(dataframe=records_df)
        self._wandb.log({"sample_predictions": records_table})
        
    # def on_save(self, args, state, control, model=None, tokenizer=None, **kwargs):
