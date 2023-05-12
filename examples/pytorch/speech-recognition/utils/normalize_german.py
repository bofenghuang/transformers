#! /usr/bin/env python3
# coding=utf-8
# Copyright 2022 Bofeng Huang

"""Normalize text data for ASR."""

import re
import unicodedata

from num2words import num2words


ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space,
    and drop any diacritics (category 'Mn' and some manual mappings)
    """
    return "".join(
        c
        if c in keep
        else ADDITIONAL_DIACRITICS[c]
        if c in ADDITIONAL_DIACRITICS
        else ""
        if unicodedata.category(c) == "Mn"
        else " "
        if unicodedata.category(c)[0] in "MSP"
        else c
        for c in unicodedata.normalize("NFKD", s)
    )


def remove_symbols(s: str, keep=""):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    return "".join(
        c if c in keep else " " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s)
    )


def escape_string(s):
    return r"".join([rf"\{s_}" for s_ in s])


class BaiscTextNormalizer:
    def __init__(self, remove_diacritics: bool = False):
        self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um)\b"
        # self.ignore_patterns = None

        self.replacers = {
            # standarize symbols
            r"’|´|′|ʼ|‘|ʻ|`": "'",  # replace special quote
            r"−|‐": "-",  # replace special dash
        }

        self.clean = remove_symbols_and_diacritics if remove_diacritics else remove_symbols

        # self.standardize_numbers = FrenchNumberNormalizer()
        # self.standardize_spellings = EnglishSpellingNormalizer()

        # latin chars
        # bh: speechbrain version for DE
        self.allowed_chars = "A-Za-z0-9öÖäÄüÜß"

    def __call__(self, s: str, do_lowercase=True, symbols_to_keep="'"):
        if do_lowercase:
            s = s.lower()

        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        # s = re.sub(r"^\s*#\d{3}\s", "", s)  # remove beginning http response code

        if self.ignore_patterns is not None:
            s = re.sub(self.ignore_patterns, "", s)

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        # if do_standardize_numbers:
        #     s = self.standardize_numbers(s)  # convert number to words
        # s = self.standardize_spellings(s)

        # s = self.clean(s, keep="-'")
        # s = re.sub(rf"[^{self.allowed_chars}\'\- ]", "", s)
        # don't keep dash for hf event
        s = self.clean(s, keep=symbols_to_keep)  # remove any other markers, symbols, punctuations with a space
        # s = re.sub(rf"[^{self.allowed_chars}\' ]", "", s)  # remove unnecessary alphabet characters
        s = re.sub(rf"[^{self.allowed_chars}{escape_string(symbols_to_keep)}\s]", " ", s)  # remove unnecessary alphabet characters

        s = re.sub(r"\s+'", "'", s)  # standardize when there's a space before an apostrophe
        s = re.sub(r"'\s+", "'", s)  # standardize when there's a space after an apostrophe
        # s = re.sub(rf"([{self.allowed_chars}])\s+'", r"\1'", s)  # standardize when there's a space before an apostrophe
        # s = re.sub(rf"([{self.allowed_chars}])'([{self.allowed_chars}])", r"\1' \2", s)  # add an espace after an apostrophe
        # s = re.sub(rf"(?<!aujourd)(?<=[{self.allowed_chars}])'(?=[{self.allowed_chars}])", "' ", s)  # add an espace after an apostrophe (except)

        # s = re.sub(r"(?<!\b{})-(?=\S)".format(r")(?<!\b".join(self.numbers_before_dash)), " ", s)  # remove dash not after numbers
        # s = re.sub(r"(?:(?<=\s)|(?<=^))\-+(?=\w)", " ", s)  # remove beginning dash in words
        # s = re.sub(r"(?<=\w)\-+(?=\s|$)", " ", s)  # remove trailing dash in words
        # s = re.sub(r"(?:^|\s)\-+\w*", " ", s)  # remove words with beginning dash
        # s = re.sub(r"\w*\-+(?=\s|$)", " ", s)  # remove words with trailing dash

        # s = re.sub(r"(\d),(\d)", r"\1\2", s)  # remove commas between digits
        # s = re.sub(r"\.([^0-9]|$)", r" \1", s)  # remove periods not followed by numbers
        # now remove prefix/suffix symbols that are not preceded/followed by numbers
        # s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        # s = re.sub(r"([^0-9])%", r"\1 ", s)

        s = re.sub(r"\s+", " ", s).strip()  # replace any successive whitespace characters with a space

        return s
