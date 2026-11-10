# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Pipeline
from haystack.components.preprocessors import TextCleaner


def test_init_default():
    cleaner = TextCleaner()
    assert cleaner._remove_regexps is None
    assert not cleaner._convert_to_lowercase
    assert not cleaner._remove_punctuation
    assert not cleaner._remove_numbers
    assert cleaner._regexes == []
    assert cleaner._translator is None


def test_run():
    cleaner = TextCleaner()
    texts = ["Some text", "Some other text", "Yet another text"]
    result = cleaner.run(texts=texts)
    assert len(result) == 1
    assert result["texts"] == texts


def test_run_with_empty_inputs():
    cleaner = TextCleaner()
    result = cleaner.run(texts=[])
    assert len(result) == 1
    assert result["texts"] == []


def test_run_rejects_invalid_inputs():
    cleaner = TextCleaner()

    with pytest.raises(TypeError):
        cleaner.run(texts=["Hello, World!", None, 42, ""])  # type: ignore[list-item]

    with pytest.raises(TypeError):
        cleaner.run(texts=15)  # type: ignore[arg-type]


def test_run_with_regex():
    cleaner = TextCleaner(remove_regexps=[r"\d+"])
    result = cleaner.run(texts=["Open123 Source", "HaystackAI"])
    assert len(result) == 1
    assert result["texts"] == ["Open Source", "HaystackAI"]


def test_run_with_multiple_regexps():
    cleaner = TextCleaner(remove_regexps=[r"\d+", r"[^\w\s]"])
    result = cleaner.run(texts=["Open123! Source", "Haystack.AI"])
    assert len(result) == 1
    assert result["texts"] == ["Open Source", "HaystackAI"]


def test_run_regexps_do_not_share_group_numbers():
    # `\1` in the second pattern must refer to the group of the second pattern, not to the first one
    cleaner = TextCleaner(remove_regexps=[r"(a)\1", r"(b)\1"])
    result = cleaner.run(texts=["xxbbxx aa"])
    assert result["texts"] == ["xxxx "]


def test_run_regexps_do_not_share_inline_flags():
    # an inline flag at the start of a later pattern used to make the combined pattern invalid
    cleaner = TextCleaner(remove_regexps=["foo", "(?i)bar"])
    result = cleaner.run(texts=["foo Bar baz"])
    assert result["texts"] == ["  baz"]


def test_run_regexps_apply_leftmost_match_first():
    # same result as a single alternation of the patterns: the leftmost match wins, then the first pattern
    cleaner = TextCleaner(remove_regexps=["b", "ab", "abc"])
    result = cleaner.run(texts=["abc ab b"])
    assert result["texts"] == ["c  "]


def test_run_regexps_do_not_join_text_around_removed_match():
    # removing a match must not create a new match out of the surrounding text
    cleaner = TextCleaner(remove_regexps=["x", "ab"])
    result = cleaner.run(texts=["axb"])
    assert result["texts"] == ["ab"]


def test_run_regexps_ignore_empty_matches():
    cleaner = TextCleaner(remove_regexps=["a*", "b"])
    result = cleaner.run(texts=["baab c", ""])
    assert result["texts"] == [" c", ""]


def test_run_with_convert_to_lowercase():
    cleaner = TextCleaner(convert_to_lowercase=True)
    result = cleaner.run(texts=["Open123! Source", "Haystack.AI"])
    assert len(result) == 1
    assert result["texts"] == ["open123! source", "haystack.ai"]


def test_run_with_remove_punctuation():
    cleaner = TextCleaner(remove_punctuation=True)
    result = cleaner.run(texts=["Open123! Source", "Haystack.AI"])
    assert len(result) == 1
    assert result["texts"] == ["Open123 Source", "HaystackAI"]


def test_run_with_remove_numbers():
    cleaner = TextCleaner(remove_numbers=True)
    result = cleaner.run(texts=["Open123! Source", "Haystack.AI"])
    assert len(result) == 1
    assert result["texts"] == ["Open! Source", "Haystack.AI"]


def test_run_with_multiple_parameters():
    cleaner = TextCleaner(
        remove_regexps=[r"\d+", r"[^\w\s]"], convert_to_lowercase=True, remove_punctuation=True, remove_numbers=True
    )
    result = cleaner.run(texts=["Open%123. !$Source", "Haystack.AI##"])
    assert len(result) == 1
    assert result["texts"] == ["open source", "haystackai"]


def test_serialization_keeps_init_parameters():
    cleaner = TextCleaner(
        remove_regexps=["World"], convert_to_lowercase=True, remove_punctuation=True, remove_numbers=True
    )
    pipeline = Pipeline()
    pipeline.add_component("cleaner", cleaner)
    assert pipeline.to_dict()["components"]["cleaner"]["init_parameters"] == {
        "remove_regexps": ["World"],
        "convert_to_lowercase": True,
        "remove_punctuation": True,
        "remove_numbers": True,
    }

    restored = Pipeline.loads(pipeline.dumps()).get_component("cleaner")

    texts = ["Hello, World 42!"]
    assert restored.run(texts=texts) == cleaner.run(texts=texts) == {"texts": ["hello  "]}
