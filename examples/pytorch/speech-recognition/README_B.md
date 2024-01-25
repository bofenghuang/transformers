
## Fine-tune openai/whisper-large-v3

*Data curation*

PnC & Normalization

- Ran PnC restoration (only `!,.?`) on the non-PnC datasets, using the adapted SpeechBox with the `bofenghuang/whisper-large-v2-cv11-french` model
- Re-ran text normalization on the PnC datasets, retaining `'-,.?!:;$%@&#~()`, and converting text to numbers

Common problems in dataset

- Complete mismatch: language, content
- Bad segmentation: transition in MLS/voxpopuli; can be filtered by alignment/ctc loss?
- Omitted/Mismatched words: scripted speech in mtedx/minds14

Common failure of the last version (`bofenghuang/whisper-large-v2-french`)

- Hallucination
    - Ground truth: [GREEK] + [Monsieur le Président, je pense qu'il est absolument indispensable d'avoir un texte européen qui protège]
    - Label: [Monsieur le Président, je pense qu'il est absolument indispensable d'avoir un texte européen qui protège] + [les femmes au travail, y compris dans la dimension de la maternité.] (bad segmentation)
    - Predition: [monsieur le président je pense qu'il est absolument indispensable d'avoir un texte européen qui protège] + [les états membres et les états membres européens de la protection des ressources de l'environnement] (hallucination)
    - Might caused by poorly segmented utterances in voxpopuli
- Ignorance
    - Ground truth: Fin de la fable treize Jupiter et le passager cet enregistrement fait partie du domaine public.
    - Label: Xi. Er et le Passager.
    - Predition: xiii jupiter et le passager
    - Might caused by poorly segmented utterances in MLS

Filtering

- Transcibe all utterances in dataset using another model (`bofenghuang/whisper-large-v2-french` not working)
- Filter by compression ratio: The utterance where the other models (seq2seq) failed (hallucinated)
- Filter out short utterances in lingualibre (uncommon words), ambiguous words and doesn't help much training
- Filter by character num difference in label and prediction, usually including bad segmentation or mismatch (if model doesn't hallucinate)
- Filter by audio language identification: only infer on first 5 seconds as some utterances in Voxpopuli start with speech in other languages (poor model performance)
- Filter by Levenchetein Distance
    - Remove mismatch and bad segmentation, but tolerate suffix changement (petit stylo, petits stylos) and word boudary erros (information, un formation)
    - WER, Stemmed WER, CER, Stemmed CER, Grapheme ER, PER
- Filter by CTC loss
- **TODO**: Check YODAS, Speech-text alignment and CTC-loss-based filtering, using Allosaurus and epistran

Tricks on small datasets

- Merge adjacent segments to learn on various lengths, served as data augmentation
- Audio augmentation (noise, music, etc)
- Train for fewer epochs to prevent overfitting

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

Notes

- Experienced underfitting issues using online audio augmentation, despite with a low ratio (0.04 for 4 augmentations)
- Encountered overfitting at 6th epoch, with a rebound in eval loss. Retrained for 5 epochs
- Should train with timestamps / previous context to preserve these capabilites, but can't get them for all data
    - No timestamps, starting / trailing silence in segments; train a model w/o timestamps, then generata peseudo timestamps as label, iteratively train another
    - Not conversation derived segments

*Key takeaways*

- Your dataset is almost the most important thing, check it randomly
- ~~Online audio augmentation (0.2), SpecAugment~~
- Can select a bigger BS since the LR is decayed during training
- Don't trust open-source library, ton of bugs in allomedia/text2num (which I can understand)

*Distillation*

- timestamp_probability 0.2 to keep the capability to predict timestamps
- condition_on_prev_probability 0.1 to be able to used with sequential long-form transcription
    - here don't really have previous context since only short segments not full audio, set to 0.1 instead of 0.5 by OpenAI seems reasonable
- Found SpecAugment still helps, got more space to go
- Distilled models dec16/8 converge faster than dec4/2, need fewer epochs
- Training script crashes, deleting the last checkpoint and re-launching the training
- Don't forget to disable kv cache


- whisper content/style prompting
- gradio: new version, initial promopt, exampels (https://www.youtube.com/watch?v=tWjdz8A1KLU)

case, punc
timestamps
condition on previous segment
translation
