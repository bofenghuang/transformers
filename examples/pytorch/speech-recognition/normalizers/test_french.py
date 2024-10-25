#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 Bofeng Huang

"""
Unit test for French text normalization.

Usage:

python -m unittest test_french.py
python -m unittest test_french.TestFrenchTextNormalizer.test_normalize_pnc
"""


import unittest

from french import FrenchTextNormalizer


class TestFrenchTextNormalizer(unittest.TestCase):
    def setUp(self):
        self.text_normalizer = FrenchTextNormalizer()

    def test_normalize(self):
        text = "Les solutions sont exactement les applications ℚ-linéaires de ℝ dans ℝ."
        generated_text = self.text_normalizer(text)
        expected_text = "les solutions sont exactement les applications q-linéaires de r dans r"
        self.assertEqual(generated_text, expected_text)

        text = "-- Que reste-t-il à jeter au dehors? -- Rien! -- Si!... La nacelle!"
        generated_text = self.text_normalizer(text)
        expected_text = "que reste-t-il à jeter au dehors rien si la nacelle"
        self.assertEqual(generated_text, expected_text)

        text = (
            "que les relevés du Bureau-Veritas chiffrèrent par centaines, territoires entiers nivelés par des trombes qui"
            " broyaient tout sur leur passage, plusieurs milliers de personnes écrasées sur terre ou englouties en mer:"
        )
        generated_text = self.text_normalizer(text)
        expected_text = (
            "que les relevés du bureau-veritas chiffrèrent par centaines territoires entiers nivelés par des trombes qui"
            " broyaient tout sur leur passage plusieurs milliers de personnes écrasées sur terre ou englouties en mer"
        )
        self.assertEqual(generated_text, expected_text)

        text = "--Eh bien?"
        generated_text = self.text_normalizer(text)
        expected_text = "eh bien"
        self.assertEqual(generated_text, expected_text)

    def test_normalize_num2text(self):
        text = "l'arménie s'est écrasé tuant les 168 personnes à bord"
        generated_text = self.text_normalizer(text)
        expected_text = "l'arménie s'est écrasé tuant les cent soixante-huit personnes à bord"
        self.assertEqual(generated_text, expected_text)

        text = "La manifestation ont affirmé qu'environ 100000 personnes"
        generated_text = self.text_normalizer(text, do_num2text=True)
        expected_text = "la manifestation ont affirmé qu'environ cent mille personnes"
        self.assertEqual(generated_text, expected_text)

        # text = "La manifestation ont affirmé qu'environ 100, 000 personnes"
        # generated_text = self.text_normalizer(text, do_num2text=True)
        # expected_text = "la manifestation ont affirmé qu'environ cent mille personnes"
        # self.assertEqual(generated_text, expected_text)

        # text = "à whitehall peu après 11 h 00 des manifestants ont bloqué la circulation"
        # generated_text = self.text_normalizer(text, do_num2text=True)
        # expected_text = "à whitehall peu après onze heures des manifestants ont bloqué la circulation"
        # self.assertEqual(generated_text, expected_text)

        text = "155 chemin des Grades, 07, 170, Lavilledieu"
        generated_text = self.text_normalizer(text, do_num2text=True)
        expected_text = "cent cinquante-cinq chemin des grades zéro sept cent soixante-dix lavilledieu"
        self.assertEqual(generated_text, expected_text)

        text = "le club retrouve ainsi la 2nde division"
        generated_text = self.text_normalizer(text, do_num2text=True)
        expected_text = "le club retrouve ainsi la seconde division"
        self.assertEqual(generated_text, expected_text)

        text = "3 place Urbain 5, 48, 000, Mende"
        generated_text = self.text_normalizer(text, do_num2text=True)
        expected_text = "trois place urbain cinq quarante-huit zéro zéro zéro mende"
        self.assertEqual(generated_text, expected_text)

        text = "76 rue Rodier, 75, 009 à Paris"
        generated_text = self.text_normalizer(text, do_num2text=True)
        expected_text = "soixante-seize rue rodier soixante-quinze zéro zéro neuf à paris"
        self.assertEqual(generated_text, expected_text)

    def test_normalize_text2num(self):
        text = "la manifestation ont affirmé qu'environ cent mille personnes"
        generated_text = self.text_normalizer(text, do_num2text=False, do_text2num=True)
        expected_text = "la manifestation ont affirmé qu'environ 100000 personnes"
        self.assertEqual(generated_text, expected_text)

    def test_normalize_pnc(self):
        symbols_to_keep = "'-,.?!:;$%@&#~()…"

        text = "Sur leur passage ,plusieurs milliers de personnes écrasées sur terre ou englouties en mer ."
        generated_text = self.text_normalizer(text, do_lowercase=False, symbols_to_keep=symbols_to_keep)
        expected_text = "Sur leur passage, plusieurs milliers de personnes écrasées sur terre ou englouties en mer."
        self.assertEqual(generated_text, expected_text)

        # text = """Décret du Gouvernement provisoire…" Parlé."""
        # generated_text = self.text_normalizer(
        #     text, do_lowercase=False, symbols_to_keep=symbols_to_keep, do_num2text=False, do_text2num=True
        # )
        # expected_text = "Décret du Gouvernement provisoire… Parlé."
        # self.assertEqual(generated_text, expected_text)

        text = (
            "Le tombeau de Toutankhamon (KV62). KV62 est peut-être la plus célèbre des tombes de la vallée, le lieu où Howard"
            " Carter a découvert en 1922 la sépulture royale presque intacte du jeune roi."
        )
        generated_text = self.text_normalizer(
            text, do_lowercase=False, symbols_to_keep=symbols_to_keep, do_num2text=False, do_text2num=True
        )
        expected_text = (
            "Le tombeau de Toutankhamon (KV62). KV62 est peut-être la plus célèbre des tombes de la vallée, le lieu où Howard"
            " Carter a découvert en 1922 la sépulture royale presque intacte du jeune roi."
        )
        self.assertEqual(generated_text, expected_text)

    def test_normalize_text2num_pnc(self):
        symbols_to_keep = "'-,.?!:;$%@&#~()…"

        text = "La manifestation ont affirmé qu'environ 100 000 personnes!"
        generated_text = self.text_normalizer(
            text, do_lowercase=False, symbols_to_keep=symbols_to_keep, do_num2text=False, do_text2num=True
        )
        expected_text = "La manifestation ont affirmé qu'environ 100 000 personnes !"
        self.assertEqual(generated_text, expected_text)

        text = "trois place Urbain cinq, quarante-huit, zéro zéro zéro, Mende"
        generated_text = self.text_normalizer(
            text, do_lowercase=False, symbols_to_keep=symbols_to_keep, do_num2text=False, do_text2num=True
        )
        expected_text = "3 place Urbain 5, 48, 000, Mende"
        self.assertEqual(generated_text, expected_text)

        text = "trente-cinq BIS avenue des Martyrs de la Résistance, zéro huit, zéro zéro zéro Charleville-Mézières"
        generated_text = self.text_normalizer(
            text, do_lowercase=False, symbols_to_keep=symbols_to_keep, do_num2text=False, do_text2num=True
        )
        expected_text = "35 BIS avenue des Martyrs de la Résistance, 08, 000 Charleville-Mézières"
        self.assertEqual(generated_text, expected_text)

        text = "Je vois 0,5 et 0,2 sur 10."
        generated_text = self.text_normalizer(
            text, do_lowercase=False, symbols_to_keep=symbols_to_keep, do_num2text=False, do_text2num=True
        )
        expected_text = "Je vois 0,5 et 0,2 sur 10."
        self.assertEqual(generated_text, expected_text)

        text = "Il s'agit d'une résidence noble en bois, d'au moins deux étages."
        generated_text = self.text_normalizer(
            text, do_lowercase=False, symbols_to_keep=symbols_to_keep, do_num2text=False, do_text2num=True
        )
        expected_text = "Il s'agit d'une résidence noble en bois, d'au moins 2 étages."
        self.assertEqual(generated_text, expected_text)

        text = "Récit imaginaire entre un corbeau freux et un homme."
        generated_text = self.text_normalizer(
            text, do_lowercase=False, symbols_to_keep=symbols_to_keep, do_num2text=False, do_text2num=True
        )
        expected_text = "Récit imaginaire entre un corbeau freux et un homme."
        self.assertEqual(generated_text, expected_text)

        text = "Élégant, le torero de Jaén possède une esthétique fine et une estocade de qualité"
        generated_text = self.text_normalizer(
            text, do_lowercase=False, symbols_to_keep=symbols_to_keep, do_num2text=False, do_text2num=True
        )
        expected_text = "Élégant, le torero de Jaén possède une esthétique fine et une estocade de qualité"
        self.assertEqual(generated_text, expected_text)

        text = "Rue du Docteur Pierre Balme, soixante-trois, zéro zéro zéro Clermont-Ferrand"
        generated_text = self.text_normalizer(
            text, do_lowercase=False, symbols_to_keep=symbols_to_keep, do_num2text=False, do_text2num=True
        )
        expected_text = "Rue du Docteur Pierre Balme, 63, 000 Clermont-Ferrand"
        self.assertEqual(generated_text, expected_text)

        text = "soixante-seize rue Rodier, soixante-quinze, zéro zéro neuf à Paris"
        generated_text = self.text_normalizer(
            text, do_lowercase=False, symbols_to_keep=symbols_to_keep, do_num2text=False, do_text2num=True
        )
        expected_text = "76 rue Rodier, 75, 009 à Paris"
        self.assertEqual(generated_text, expected_text)

        text = "le club retrouve ainsi la seconde division"
        generated_text = self.text_normalizer(
            text, do_lowercase=False, symbols_to_keep=symbols_to_keep, do_num2text=False, do_text2num=True
        )
        expected_text = "le club retrouve ainsi la 2nde division"
        self.assertEqual(generated_text, expected_text)

        text = "Lundi quinze avril dix-neuf cent quatre-vingt-cinq."
        generated_text = self.text_normalizer(
            text, do_lowercase=False, symbols_to_keep=symbols_to_keep, do_num2text=False, do_text2num=True
        )
        expected_text = "Lundi 15 avril 1985."
        self.assertEqual(generated_text, expected_text)

        text = "Trois rue Paulin Durand, quarante et un, trois cents à Salbris"
        generated_text = self.text_normalizer(
            text, do_lowercase=False, symbols_to_keep=symbols_to_keep, do_num2text=False, do_text2num=True
        )
        expected_text = "3 rue Paulin Durand, 41, 300 à Salbris"
        self.assertEqual(generated_text, expected_text)

        text = "Sept personnes constituent chaque équipe, six concurrents et un capitaine, chacun avec un numéro."
        generated_text = self.text_normalizer(
            text, do_lowercase=False, symbols_to_keep=symbols_to_keep, do_num2text=False, do_text2num=True
        )
        expected_text = "7 personnes constituent chaque équipe, 6 concurrents et un capitaine, chacun avec un numéro."
        self.assertEqual(generated_text, expected_text)
