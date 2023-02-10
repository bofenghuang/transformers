#!/usr/bin/env python
# Copyright 2021  Bofeng Huang

import logging
import warnings

import gradio as gr
from transformers.utils.logging import disable_progress_bar

from predict_ner_punc_multiple_choice import AggregationStrategy, TokenClassificationPredictor

# warnings.filterwarnings("ignore")

disable_progress_bar()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

model_name_or_path = (
    "/home/bhuang/transformers/examples/pytorch/token-classification/outputs/recasepunc-multilingual/xlm-roberta-large_ft"
)

tc = TokenClassificationPredictor(
    model_name_or_path,
    "cpu",
    # batch_size=64,
    stride=100,
    overlap=100,
    aggregation_strategy=AggregationStrategy.LAST,
)
logger.info("TokenClassificationPredictor has been initialized")


def run(sentence):
    processed_sentence = tc(sentence)[0]
    logger.info(f"Raw sentence: {sentence}")
    logger.info(f"Generated sentence: {processed_sentence}")
    return processed_sentence


examples = [
    [
        "d'accord effectivement le véhicule peugeot307 est plus abordable que le kia xceed"
    ],
    [
        "i have all the necessary information for the declaration of broken glass for the invoicing so there will be of course the deductible of sixty of sixty euros to be paid the day of the intervention the invoice and the declaration of broken glass will be sent at the same time once the intervention finished forgive to the insurance and i remind you the appointment therefore of the nineteenth of january at nine o'clock in the morning at mireille lauze for the replacement of the rear window on the subject do you have any questions sir"
    ],
    [
        "d'accord effectivement le véhicule il est de deux mille le vingt-cinq janvier deux mille très bien donc j'ai toutes les informations nécessaires pour la déclaration de bris de glace pour la facturation donc y aura bien sûr la franchise de soixante de soixante euros à régler le jour de l'intervention la facture et la déclaration de bris de glace sera envoyé en même temps une fois l'intervention fini pardon à l'assurance et je vous rappelle le rendez vous donc du dix-neuf janvier à neuf heures trente à mireille lauze pour le remplacement de la lunette arrière sur le sujet qui est ce que vous avez des questions monsieur"
    ],
    [
        "tengo toda la informacion necesaria para la declaracion de rotura de cristales para la facturacion por lo que habra por supuesto la franquicia de sesenta de sesenta euros a pagar el dia de la intervencion la factura y la declaracion de rotura de cristales se enviara al mismo tiempo una vez terminada la intervencion perdone al seguro y le recuerdo la cita por lo tanto del diecinueve de enero a las nueve de la mañana en mireille lauze para la sustitucion de la luna trasera sobre el tema tiene alguna pregunta señor"
    ],
    [
        "ich habe alle notwendigen informationen für die erklärung von glasbruch für die rechnungsstellung so wird es natürlich die selbstbeteiligung von sechzig von sechzig euro zu zahlen am tag der intervention die rechnung und die erklärung von glasbruch wird zur gleichen zeit gesendet werden sobald die intervention beendet vergeben an die versicherung und ich erinnere sie die verabredung daher der neunzehnten januar um neun uhr morgens bei mireille lauze für den ersatz der heckscheibe auf das thema haben sie fragen herr"
    ],
]

iface = gr.Interface(
    fn=run,
    inputs=gr.Textbox(label="Type something..."),
    outputs=gr.Text(label="Generated sentences"),
    examples=examples,
    css=".footer {display:none !important}",
    title="Zaion Punctuation & Capitalization",
    description="Realtime demo for text punctuation and capitalization on FR/EN/ES/DE.",
    # article=f"The dataset repo is [{DATASET_REPO_URL}]({DATASET_REPO_URL})",
    article="*Merci d'envoyer vos retours à zaion lab nlp team :)*",
    allow_flagging="never",
)

iface.launch(server_name="0.0.0.0", debug=True, share=True)
# iface.launch()
