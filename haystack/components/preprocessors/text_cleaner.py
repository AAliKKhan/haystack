# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
import string
from typing import Any

from haystack import component, default_to_dict


@component
class TextCleaner:
    """
    Cleans text strings.

    It can remove substrings matching a list of regular expressions, convert text to lowercase,
    remove punctuation, and remove numbers.
    Use it to clean up text data before evaluation.

    ### Usage example

    ```python
    from haystack.components.preprocessors import TextCleaner

    text_to_clean = "1Moonlight shimmered softly, 300 Wolves howled nearby, Night enveloped everything."

    cleaner = TextCleaner(convert_to_lowercase=True, remove_punctuation=False, remove_numbers=True)
    result = cleaner.run(texts=[text_to_clean])
    ```
    """

    def __init__(
        self,
        remove_regexps: list[str] | None = None,
        convert_to_lowercase: bool = False,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
    ) -> None:
        """
        Initializes the TextCleaner component.

        :param remove_regexps: A list of regex patterns to remove matching substrings from the text.
        :param convert_to_lowercase: If `True`, converts all characters to lowercase.
        :param remove_punctuation: If `True`, removes punctuation from the text.
        :param remove_numbers: If `True`, removes numerical digits from the text.
        """
        self._remove_regexps = remove_regexps
        self._convert_to_lowercase = convert_to_lowercase
        self._remove_punctuation = remove_punctuation
        self._remove_numbers = remove_numbers

        # Each pattern is compiled on its own, so group numbers (backreferences) and inline flags of one pattern
        # can't leak into the others.
        self._regexes = [re.compile(pattern, flags=re.IGNORECASE) for pattern in remove_regexps or []]
        to_remove = ""
        if remove_punctuation:
            to_remove = string.punctuation
        if remove_numbers:
            to_remove += string.digits

        self._translator = str.maketrans("", "", to_remove) if to_remove else None

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            remove_regexps=self._remove_regexps,
            convert_to_lowercase=self._convert_to_lowercase,
            remove_punctuation=self._remove_punctuation,
            remove_numbers=self._remove_numbers,
        )

    @component.output_types(texts=list[str])
    def run(self, texts: list[str]) -> dict[str, Any]:
        """
        Cleans up the given list of strings.

        :param texts: List of strings to clean.
        :returns: A dictionary with the following key:
            - `texts`:  the cleaned list of strings.
        :raises TypeError:
            If `texts` is not a list or any element is not a string.
        """
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise TypeError("`texts` must be a list of strings.")

        if self._regexes:
            texts = [self._remove_matches(text) for text in texts]

        if self._convert_to_lowercase:
            texts = [text.lower() for text in texts]

        if self._translator:
            texts = [text.translate(self._translator) for text in texts]

        return {"texts": texts}

    @staticmethod
    def _search_non_empty(regex: re.Pattern[str], text: str, position: int) -> re.Match[str] | None:
        """Returns the leftmost non-empty match of `regex` in `text` starting from `position`, if any."""
        match = regex.search(text, position)
        while match is not None and match.end() == match.start():
            if match.start() >= len(text):
                return None
            match = regex.search(text, match.start() + 1)
        return match

    def _remove_matches(self, text: str) -> str:
        """
        Removes the substrings matching any of the regular expressions.

        At each position the leftmost match wins and, on a tie, the pattern listed first wins.
        The matches never overlap, like with a single alternation (`first|second|...`) of all the patterns.
        Empty matches remove nothing and are ignored.
        """
        matches = [self._search_non_empty(regex, text, 0) for regex in self._regexes]
        kept: list[str] = []
        last_end = 0
        while True:
            best: re.Match[str] | None = None
            for index, regex in enumerate(self._regexes):
                match = matches[index]
                if match is not None and match.start() < last_end:
                    # this match overlaps the one that was just removed: look for the next one
                    match = matches[index] = self._search_non_empty(regex, text, last_end)
                if match is not None and (best is None or match.start() < best.start()):
                    best = match
            if best is None:
                break
            kept.append(text[last_end : best.start()])
            last_end = best.end()
        kept.append(text[last_end:])
        return "".join(kept)
