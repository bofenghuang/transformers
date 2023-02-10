#! /usr/bin/env python3
# coding=utf-8
# Copyright 2022 Bofeng Huang

"""Unit test for French text normalization"""

from utils.normalize_french import FrenchTextNormalizer

text_normalizer = FrenchTextNormalizer()

text = "Les solutions sont exactement les applications ℚ-linéaires de ℝ dans ℝ."
formatted_text = "les solutions sont exactement les applications q-linéaires de r dans r"
assert text_normalizer(text) == formatted_text

text = "La manifestation ont affirmé qu'environ 100 000 personnes"
formatted_text = "la manifestation ont affirmé qu'environ cent mille personnes"
assert text_normalizer(text) == formatted_text

text = "La manifestation ont affirmé qu'environ 100,000 personnes"
formatted_text = "la manifestation ont affirmé qu'environ cent mille personnes"
assert text_normalizer(text) == formatted_text

text = "à whitehall peu après 11 h 00 des manifestants ont bloqué la circulation"
formatted_text = "à whitehall peu après onze heures des manifestants ont bloqué la circulation"
assert text_normalizer(text) == formatted_text
