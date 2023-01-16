import html

from typing import Union
from sacremoses import MosesPunctNormalizer, MosesTokenizer, MosesDetokenizer

from tokenizers.normalizers import NFKC


class TextNormalizer:
    def __init__(self, lang="en") -> None:
        self.lang = lang
        self.punct_normalizer = MosesPunctNormalizer(lang=self.lang)
        self.tokenizer = MosesTokenizer(lang=self.lang)
        self.detokenizer = MosesDetokenizer(lang=self.lang)
        self.nfkc = NFKC()
        self.normalization_map = {
            "¨": '"',
            "​": " ",
            "’": "'",
            "♪": " ",
            "♫": " ",
            "●": "-",
            "&lrm;": "",
            "&rlm;": "",
        }

    def unescape_html(self, text: Union[str, list]) -> Union[str, list]:
        """Normalises HTML encoded characters i.e. (&apos;) -> (‘)"""
        if isinstance(text, str):
            return html.unescape(text)
        elif isinstance(text, list):
            return [html.unescape(t) for t in text]

    def lowerize(self, text: Union[str, list]) -> Union[str, list]:
        """Lowerizes text according to the given language"""

        def _lowerize(text: str) -> str:
            return text.lower()

        if isinstance(text, str):
            return _lowerize(text)
        elif isinstance(text, list):
            return [_lowerize(t) for t in text]

    def moses_punct_normalize(self, text: Union[str, list]) -> Union[str, list]:
        """
        Normalises punctuations such as (-) and (–) into one. 
        «Hello World» — she said… becomes: "Hello W
        orld" - she said...
        """
        if isinstance(text, str):
            return self.punct_normalizer.normalize(text)
        elif isinstance(text, list):
            return [self.punct_normalizer.normalize(t) for t in text]

    def special_normalize(self, text: Union[str, list]) -> Union[str, list]:
        """
        Applies special normalization to the given string or list of strings.
        """
        if isinstance(text, str):
            for k, v in self.normalization_map.items():
                text = self.nfkc.normalize_str(text.replace(k, v))
            return text
        elif isinstance(text, list):
            for k, v in self.normalization_map.items():
                text = [self.nfkc.normalize_str(t.replace(k, v)) for t in text]
            return text

    def tokenize(self, text: Union[str, list]) -> Union[list, list[list]]:
        """Tokenises text using Moses Tokenizer"""
        if isinstance(text, str):
            return self.tokenizer.tokenize(text)
        elif isinstance(text, list):
            return [self.tokenizer.tokenize(t) for t in text]

    def detokenize(self, text: Union[list[str], list[list]]) -> Union[str, list]:
        """
        Detokenizes text using MosesDetokenizer.
        """
        if isinstance(text[0], str):
            return self.detokenizer.detokenize(text)
        elif isinstance(text[0], list):
            return [self.detokenizer.detokenize(t) for t in text]

    def fix_tokenized(self, text: Union[str, list]) -> Union[str, list]:
        if isinstance(text, str):
            text = text.split()
            return self.detokenizer.detokenize(text)
        elif isinstance(text, list):
            text = [t.split() for t in text]
            return [self.detokenizer.detokenize(t) for t in text]

    def __call__(self, text: Union[str, list]) -> Union[str, list]:
        """
        Applies all preprocessing pipeline to the given string or list of strings.
        """
        text = self.special_normalize(text)
        text = self.moses_punct_normalize(text)
        text = self.fix_tokenized(text)
        return text
