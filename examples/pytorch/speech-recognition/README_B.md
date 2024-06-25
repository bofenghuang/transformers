
## Fine-tune openai/whisper-large-v3

*Data curation*

Data collection

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

Preprocessing

- Filter out empty text (at least one alphabet)
- Filter by num of words
- Filter by audio duration (1s < dur < 30s)
- Filter by spotted common errors in specific datasets (e.g., voice-over/transition in MLS)
- Dedup by text (max 128)

PnC Restoration & Text Normalization

- Certain datasets (e.g., MLS and MediaSpeech) lack punctuation and proper casing in their transcriptions
- Generate pseudo PnC (only `!,.?`) on the non-PnC datasets, using the HF SpeechBox, with the `bofenghuang/whisper-large-v2-cv11-french` model
- Apply text normalization on all datasets, retaining `'-,.?!:;$%@&#~()`, and converting text to numbers (erros: "une seconde" -> "une 2nd")

Common issues found in the datasets

- Complete mismatch: language, content
- Bad segmentation: transitions in MLS/voxpopuli; can be filtered by alignment/ctc loss?
- Omitted / Mismatched words: scripted speech in mTEDx and mtedx/minds14

Common failures of the previous version (`bofenghuang/whisper-large-v2-french`)

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

Filtering

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

*Model*

- https://github.com/openai/whisper/discussions/1762
- More repetitions and hallucinations?
    - https://github.com/openai/whisper/discussions/1783
    - https://github.com/ggerganov/whisper.cpp/pull/1444
    - https://deepgram.com/learn/whisper-v3-results

*Training*

Tips of the author

- 16 epochs over the 960h train data
- batch size 256
- linear LR decay from 6.25e-06 to zero
- no weight decay
- no dropout

Tricks on small datasets

- Merge adjacent segments to create varying lengths, serving as a form of data augmentation
- Audio augmentation (noise, music, etc)
- Train for fewer epochs to prevent overfitting

Notes

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

*Whisper*

- Generalization capabilities
- Ability to predict PnC
- Timestamp prediction
- Condition on previous segments
- Translation

## Whisper Distillation

*Notes*

- Set timestamp_probability to 0.2 to maintain the capability to predict timestamps
- Set condition_on_prev_probability to 0.1 to enable use with sequential long-form transcription
    - Since don't have real previous context, setting it to 0.1 instead of 0.5 as recommended by OpenAI seems reasonable
- Distilled models w/ 16/8 decoder layers converge faster than w/ 4/2 decoder layers, and require fewer epochs
- Training script crashes occasionally, deleting the last checkpoint and re-launching the training resolves the issue
- Make sure to explicitly disable kv cache during the training!

*Augmentations*

- No improvement with the use of dropout within the model (activation/attention dropout)
- Online audio augmentation (speed, gaussian noise, natural noise, pitch shift) result in underfitting, even with a low ratio (0.04 for 4 augmentations), need to double-check used parameters
- SpecAugment can actully help, still room to decrease eval loss / wer

*It's unusal that the dec16 distilled model performed better than original finetuned model, should always see a monotonic increase in WER as I reduced decoder layers*

- Should evaluate on more & high-quality ID / OOD datasets
- Issues with fine-tuning, w/o previous context conditioning and timestamps

*Failure cases for sequential long-form decoding (discussion with Sanchit)*

- Solutions to keep `distil-large-v3` within 1% WER of large-v3 on the long-form datasets with sequential decoding (Each of these steps gets you about a 1% WER improvement using the long-form algorithm)
- Freeze the decoder embeddings, which has been pretrained to handle longer context lengths
- Pack your dataset to 30s samples: the sequential algorithm uses the last predicted timestamp in each 30s chunk to shift the sliding window. If this timestamp is incorrect (due to your training data being centred around a shorter time length, like 15s), then your chunks quickly become inaccurate and decoding breaks down
- Make sure you train on timestamps (we're using 50% timestamp probability): again, the sequential algorithm relies on timestamp prediction, so you need at least some data with accurate timestamps to make sure you get proper timestamp prediction
- For `condition_on_prev`, I'm finding that you don't need a super high prob to make this work (e.g. 20% is sufficient), since during pre-training 50% of the data is conditioned (>300k hours for large-v2), so there's already good representations. But I've bumped this up to 50% prob when distilling large-v3 just to follow the OpenAI hyperparameters.
- Found (weirdly) that large-v3 has a tendency to sometimes generate entirely in upper-case. This looks like an artefact of OpenAI's weak supervision process. If you're using pseudo-labels, you can filter out these upper-case transcription (got a 2% WER improvement on Meanwhile)


## todo:

better text2num:
- seconde
- 40 pour 100

