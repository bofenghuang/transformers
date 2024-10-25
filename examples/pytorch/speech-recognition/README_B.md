
## Whisper Fine-tuning

*French ASR data collection*

|            Dataset            | Number of Files | Total Duration | Avg. Duration | Punctuation | Casing | Description                                                     |
| :---------------------------: | :-------------: | :------------: | :-----------: | :---------: | :----: | --------------------------------------------------------------- |
|        MCV-13/fr/train        |     509,300     |    732.02h     |     5.17s     |      ✅      |   ✅    | Crowd workers recording text from Wikipedia                     |
|         MLS/fr/train          |     258,213     |    1076.58h    |    15.01s     |      ❌      |   ❌    | LibriVox read audiobooks                                        |
|      Voxpopuli/fr/train       |     73,561      |    205.70h     |    10.07s     |      ✅      |   ❌?   | European Parliament event recordings (2009-2020)                |
|        Fleurs/fr/train        |      3,193      |     10.32h     |    11.64s     |      ✅      |   ❌    | FLoRes in 102 languages                                         |
|        mTEDx/fr/train         |     116,045     |    175.83h     |     5.45s     |      ✅      |   ✅    | TEDx talks                                                      |
|        MediaSpeech/fr         |      2,498      |     10.00h     |    14.41s     |      ❌      |   ❌    | short speech segments extracted from YouTube                    |
|          M-AILABS/fr          |     86,597      |    181.90h     |     7.56s     |      ✅      |   ✅    | Most of the data is based on LibriVox and Project Gutenberg     |
| African-Accented-French/train |      9,401      |     11.68h     |     4.47s     |      ✅      |   ❌    | From Cameroon, Chad, Congo, Gabon, and Niger                    |
|        Lingua-Libre/fr        |     257,927     |     91.52h     |     1.28s     |      ❌      |   ❌    | Wikimédia France, short audios                                  |
|           Att-HACK            |     36,634      |     27.12h     |     2.66s     |      ❌      |   ❌    | Acted expressive speech in French (from 3 to 5 for each phrase) |
|        PolyAI/minds14         |       539       |     1.25h      |     8.36s     |      ❌      |   ❌    | SLU in e-banking domain                                         |
|             Total             |    1,353,908    |    2523.92h    |     6.71s     |             |        |                                                                 |

*Common issues found in the datasets*

- Complete mismatch: language, content
- Bad segmentation: transitions in MLS/voxpopuli; can be filtered by alignment/ctc loss?
- Omitted / Mismatched words: scripted speech in mTEDx and mtedx/minds14

*Common failures of the previous version (`bofenghuang/whisper-large-v2-french`)*

- Hallucination
    - Ground truth: [GREEK] + [Monsieur le Président, je pense qu'il est absolument indispensable d'avoir un texte européen qui protège]
    - Label: [Monsieur le Président, je pense qu'il est absolument indispensable d'avoir un texte européen qui protège] + [les femmes au travail, y compris dans la dimension de la maternité.] (bad segmentation)
    - Predition: [monsieur le président je pense qu'il est absolument indispensable d'avoir un texte européen qui protège] + [les états membres et les états membres européens de la protection des ressources de l'environnement] (hallucination)
    - May have been caused by poorly segmented utterances in voxpopuli
- Ignorance
    - Ground truth: Fin de la fable treize Jupiter et le passager cet enregistrement fait partie du domaine public.
    - Label: Xi. Er et le Passager.
    - Predition: xiii jupiter et le passager
    - May have been caused by poorly segmented utterances in MLS

*Data preprocessing*

- Filter out empty text (at least one alphabet)
- Filter by num of words
- Filter by audio duration (1s < dur < 30s)
- Filter by spotted common errors in specific datasets (e.g., voice-over/transition in MLS)
- Dedup by text (max 128)

*Filtering*

- Transcibe all utterances using a CTC model `bofenghuang/asr-wav2vec2-ctc-french` (`bofenghuang/whisper-large-v2-french` has similar issues)
- ~~Filter by label-prediction compression ratio: The utterance where the other models (seq2seq) failed (hallucinated)~~
- Filter out short utterances in lingualibre, uncommon & ambiguous words doesn't help much with training
- Filter by character num difference in label and prediction, usually indicating bad segmentation or mismatch
- ~~Filter by audio language identification: only infer on first 5 seconds (speed-up, some utterances in Voxpopuli start with speech in other language), poor model performance~~
- Filter by levenchetein distance
    - Remove mismatch and bad segmentation, but tolerate suffix changement (petit stylo, petits stylos) and word boudary errors (information, un formation)
    - Considered WER, Stemmed WER, CER, Stemmed CER, Grapheme ER, PER
    - Finally used levenchetein distance between phonemes of labels and predictions (bootphon/phonemizer using the EspeakBackend)
- (TODO): Use a CTC-based phonemizer to predict phonemes directly
- (TODO): Filter by CTC loss

*PnC Restoration & Text Normalization*

- Certain datasets (e.g., MLS and MediaSpeech) lack punctuation and proper casing in their transcriptions
- Generate pseudo PnC (only `!,.?`) on the non-PnC datasets, using the HF SpeechBox, with the `bofenghuang/whisper-large-v2-cv11-french` model
- Apply text normalization on all datasets, retaining `'-,.?!:;$%@&#~()`, and converting text to numbers (erros: "une seconde" -> "une 2nd")

*Model (capabilities to keep in fine-tuning idealy)*

- Generalization capabilities
- Ability to predict punctuation/case/number
- Timestamp prediction
- Condition on previous segments
- Translation

*Model*

- https://github.com/openai/whisper/discussions/1762
- More repetitions and hallucinations?
    - https://github.com/openai/whisper/discussions/1783
    - https://github.com/ggerganov/whisper.cpp/pull/1444
    - https://deepgram.com/learn/whisper-v3-results

*Training tips of the author (Jong Wook Kim)*

- 16 epochs over the 960h train data
- batch size 256
- linear LR decay from 6.25e-06 to zero
- no weight decay
- no dropout

recommendation was to select a learning rate about 40x smaller than pre-training, and linearly decay it to 0 over the course of training. For the small checkpoint, this would be 5e-4 / 40 = 1.25e-5, near enough 1e-5! So my empirical observations align with his (from sanchit?)

*Tricks on small datasets (Zaion)*

- Merge adjacent segments to create varying lengths, serving as a form of data augmentation
- Audio augmentation (noise, music, etc)
- Train for fewer epochs to prevent overfitting

*Training notes*

- Experienced underfitting issues when using online audio augmentation, even with a low ratio (0.04 for 4 augmentations)
- Encountered overfitting at 6th epoch, with a rebound in eval loss. Retrained for 5 epochs
- Should have traind with timestamps / previous context to preserve these capabilites, but not available for all datasets
    - No timestamps, presence of starting / trailing silence in segments
        - Train a 1st model w/o timestamps, then generata peseudo timestamps as labels, iteratively train another model?
        - whisper-x
    - Not conversation-derived segments
        - Generate pseudo context using LLM?

*Key takeaways*

- Your dataset is almost the most important thing, check it randomly
- ~~Online audio augmentation (0.2), SpecAugment~~
- Can select a bigger BS since the LR is decayed during training
- Don't trust open-source library, ton of bugs in allomedia/text2num (but I can understand)

*Nice features to have in the demo*

- whisper content/style prompting
- gradio: new version, initial promopt, examples (https://www.youtube.com/watch?v=tWjdz8A1KLU)

*todo*

- better data
- freeze encoder



## Whisper Distillation

Pure fine-tuning trains on one or more datasets using the standard cross-entropy loss. As such, there is no involvement from the teacher checkpoint during training, and so the fine-tuned model is permitted to overfit to the distribution of the training data we provide.

*Notes*

- distil-large-v3-en trained on 21k hours (10% wer filtering) for 80k steps (11 epochs), and found that ~~13k~~ 1k hours and 50k steps was required to reach convergence
- distil-large-v3-fr trained on 2.1k hours (20% wer filtering from 4.5k hours) for 18k steps (14 epochs) using batch size of 256, best checkpoint got at 14k steps (11 epochs)
- WER threshold of 10% provides the optimal trade-off between high-quality and quantity. For multilingual distillation, threshold should be set in accordance with the WER achieved by pre-trained model on test set
- timestamp_probability is the per-sample probability for retaining timestamp tokens in the labels, set it to 0.2 to maintain the capability to predict timestamps, recommend setting to a value below 0.5
- condition_on_prev_probability is the per-sample probability for conditioning on previous labels, set it to 0.2 to enable use with sequential long-form transcription, OpenAI pretrained with 0.5
    - ~~Since don't have real previous context, setting it to 0.1 instead of 0.5 as recommended by OpenAI seems reasonable~~
- freeze_embed_positions / freeze_embed_embeddings / freeze_decoder (only train token_embed/proj)
- Distilled models w/ 16/8 decoder layers converge faster than w/ 4/2 decoder layers, and require fewer epochs
- Training script crashes occasionally, deleting the last checkpoint and re-launching the training resolves the issue
- Make sure to explicitly disable kv cache during the training (lot of memory)!

*Data augmentations*

- No improvement with the use of dropout within the model (activation/attention dropout)
- Online audio augmentation (speed, gaussian noise, natural noise, pitch shift) result in underfitting, even with a low ratio (0.04 for 4 augmentations), need to double-check used parameters
- SpecAugment can actully help, still room to decrease eval loss / wer

*It's unusal that the dec16 distilled model performed better than original finetuned model, should always see a monotonic increase in WER as I reduced decoder layers*

- Should evaluate on more & high-quality ID / OOD datasets
- Issues with fine-tuning, w/o previous context conditioning and timestamps

*Solutions to keep `distil-large-v3` within 1% WER of large-v3 on the long-form datasets with sequential decoding (Each of these steps gets you about a 1% WER improvement using the long-form algorithm)*
*Failure cases for sequential long-form decoding (discussion with Sanchit)*

- Freeze the decoder embeddings, which has been pretrained to handle longer context lengths
- Pack your dataset to 30s samples: the sequential algorithm uses the last predicted timestamp in each 30s chunk to shift the sliding window. If this timestamp is incorrect (due to your training data being centred around a shorter time length, like 15s), then your chunks quickly become inaccurate and decoding breaks down
- Make sure you train on timestamps (we're using 50% timestamp probability): again, the sequential algorithm relies on timestamp prediction, so you need at least some data with accurate timestamps to make sure you get proper timestamp prediction
- For `condition_on_prev`, I'm finding that you don't need a super high prob to make this work (e.g. 20% is sufficient), since during pre-training 50% of the data is conditioned (>300k hours for large-v2), so there's already good representations. But I've bumped this up to 50% prob when distilling large-v3 just to follow the OpenAI hyperparameters.
- Found (weirdly) that large-v3 has a tendency to sometimes generate entirely in upper-case. This looks like an artefact of OpenAI's weak supervision process. If you're using pseudo-labels, you can filter out these upper-case transcription (got a 2% WER improvement on Meanwhile)

*Which kind of data to train/fine-tune/distil whisper models?*

- 30-second chunks of single speaker; the sequential long-form decoding algorithm uses the last predicted timestamp in each 30s chunk to shift the sliding window, thus requires accurate timestamp prediction and well adapted to 30s audio
- w/ case/punctuation/number (word2num, case/punc restoration by speechbox or LLM but conditioneing on pure text, re-predict using whisper then WER-filter)
- w/ timestamps; sentence-level instead of utterance-level w/ only full start/end (CTC segmentation, re-predict using whisper)
- w/ condition_on_prev
- multi-task w/ ASR/MT/VAD
- w/ prompting

*Constraints*

- Whisper's timestamps prone to inaccuracies
- Pseudo-labels using timestamps are marginally less accurate than not using when predicting w/ whisper
- Several datasets' transcriptions aren't cased nor punctuated
- For individual segments like in mcv, predicting individually gets better quality transcription than merging than preciting

*Tuning hyperparams*

Most of the conclusions were derived from the WERs on mcv17 validation set.
Although Fleurs is a good ood test set, its small size causes the performance curve to be quite unstable.
Training was conducted for 20 epochs on the mcv17 Italian dataset, which consists of 206 hours of data after filtering by a 10% WER threshold. However, the models did not fully converge, so some conclusions may not be scalable.

- min_audio_duration 0->5s, didn't help, but resulted in less data and higher loss
- lr_scheduler_type cosine->linear, converged slower but reached a spot with a lower loss/wer
- condition_on_prev_probability 0.5->0.1, improved but might be due to training on fake prev_text from mcv17
- Removing adam specific parameters (adam_beta1 0.9, adam_beta2 0.95, adam_epsilon 1e-5), helped a lot
- lr_rate 1e-4->3e-4, improved a lot
- wer_threshold 10%->20%, improved a lot
- Language transfer; training on distil-large-v3 instead of a reinitialized checkpoint, improved a lot
- Audio augmentation improved on id data (prob 0.2>0.1>0.05>0), but not as effective on ood (0.2=0.05>0.1>0); a good data augmentation method to crack iid benchmark
- Weight decay 0.01 improved; while 0.1 produced similar wer as not using wd when trained 20 eps, perhaps required longer training; cannot be used with dropout, wer was higher than only using dropout
- dropout (embedding+self-attn/cross-attn/mlp) rate 0.05 improved, while 0.1 led to undertraining and instability
- attn_dropout rate 0.05 improved a little, while 0.1 also resulted in undertraining, but not as unstable as dropout
- dropout > dropout+attn_dropout > attn_dropout
- SpecAug (time_prob/feat_prob)
    - mcv: 0.1/0.05 > 0.1/0.1 > 0.2/0.05 > 0.05/0 > 0.05/0.05
    - fleurs: 0.1/0.1 > 0.05/0.05 > 0.2/0.05 > 0.1/0.05 > 0.05/0
- SpecAug (time_prob/time_length/feat_prob/feat_length)
    - 0.1/10/0.1/10 > 0.1/20/0.1/10 > 0.1/40/0.1/10 (time_length 10 is enough, higher hurted performance)
    - 0.1/10/0.1/10 > 0.05/20/0.05/20 > 0.05/10/0.05/10 ~= 0.05/20/0.05/10 (0.1/10 better than 0.05/20, favoring more sparse masking)
- training w/o regularization (audioaug/specuagment/dropout) resulted in overfitting with a much lower training loss but a much higher eval wer
- schedule_free need to train longer; actually worse than linear
- label_smoothing_factor of 0.1 slightly degraded wer; try 0.01?

*Tuning hyperparams2*

Trained on a subset of ~1000 hours of French data and evaluated on all test sets (short, long, id, ood, chunked/sequential)

- Reducing condition_on_prev_probability from 0.5 to 0.2 led to faster convergence (model struggled with sequential long-form tasks even at 0.5; training longer helped it first learn to predict only current and then predict with randomly introduced previous context), improved on short-form id/ood test sets, and even slightly enhanced sequential-long-form perf (weird)
- Label-smoothing w/ 0.01/0.1 slightly degraded on short/long-form id, but improved on ood (no need to overcomplicate the setup)
- Unfreezing emb_pos improved on short-form id/ood, but slightly degraded on chunked/sequential long-form, significantly degraded (from 57 to 81) when condition_on_prev was activated (actual text is shifted)
- SpecAugment improved across all evaluations, including shor-form id/ood, chunked/sequential long-form, and should be used by default
- Audio augmentation improved on short-form id (clean datasets like mtedx, af_accented), slightly on sequential long-form with condition_on_prev; However, it is not compatible with dropout
- Audio augmentation (speed_perturb, reverb) might advance/delay waveform, making it unsuitable for timestamp prediction or sequential long-form (if only augmented on the fly)
- Dropout improved on short-form, but not compatible with audio_aug?
- BPE-dropout degraded on short-form, but improved on long-form (though w/o condition_on_prev); Activate this to bias more towards long-form perf
- Trained for longer, starting from 20 epochs and finding that it was still converging, then extended to 30 epochs (more patience is necessary for distillation). My intuition is that this was beneficial due to 1) dynamically added timestamps and previous context, 2) aggressive SpecAugment, and 3) the model being small, which makes it okay to repeat training for longer

*Tuning hyperparams3*

Trained on a subset of ~2000 hours of French data and evaluated on validation sets of mcv17 and fleurs

- SpecAug time 0.1x10x2 got similar training curve with 0.1x10x10 (even not same behavior on shorter audios)
- Tried NeMo's SpecAug (making consistent numbers of spans, with length sampled from 0 to constant or proportional to audio duration) but worse than HF's (making span of constant length without sampling, number proportional to audio duration with a minimum)
- Training for longer with increasing epoch from 30 to 60, is more effective when using more agressive augmentation (SpecAug)
- Found SpecAug time022x15x2 feat03x14x2 a good spot when trained for 80 epochs (44k steps); when scale training data from 2k jours to 10k hours, need to use less agressive augmentation to enter into a safe zone to avoid learning unexpected patterns (wer flys up)
- weight decay 0.1 got higher ce/kl loss but lower wer, wd 0.01 worse than w/o wd
- linear better than cosine if trained for enough longer (sometime cosine better means we can't trained for longer??), both better than wsd
- lr 1e-4 worse than 3e-4 even trained for enough long


*todo*

why pseudo-label concatenated yodas utt? 1) more context improves decoding 2) reduce bad-segmented inserted/deleted words at beginning/end 3) pack to 30s to speed up decoding
why dynamically read data? 1) dynamically set time_prob/prev_prob 2) audio augmentation

- Freeze token_embeddings/position_embeddings/both
- [Musique]
- <|0.00|> « On y veillera, monsieur, mais pensez-vous que cette oasis soit connue ? »<|9.76|>

ASR data filtering (WIP)

- concatenate (optional; don't filter out min-duration since badly segmented; filter by empty text after norm eg [Musique])
- speech lid
- normalize number, case/punc
- whisper restores case/punc (optional; non-batch); llm restores vowel (mls)
- phoneme wav2vec2-ctc model predicts text/alignment (strip extra words/audio)
- text2num (seconde, 40 pour 100)

ast (llm generated labels)

audio augmentation (various compression codecs, simulating issues such as packet loss)

```
# by defaut save data to pcm16 which degrades the quality? should have saved in float?
sf.write(audio_filepath, sample['audio']['array'], samplerate=sample_rate, format='wav') # , subtype="FLOAT")
```
