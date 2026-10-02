"""Tests for negative phrase generation, phrase schedules and voice splits."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from livekit.wakeword.config import WakeWordConfig
from livekit.wakeword.data import generate as gen
from livekit.wakeword.data.generate import (
    _edit_distance,
    _strip_stress,
    build_phrase_schedule,
    generate_adversarial_phrases,
    generate_word_swap_phrases,
    negative_phrase_schedules,
    phonetic_neighbours,
    split_holdout,
    split_phrase_parts,
)
from livekit.wakeword.data.piper.text import get_cmudict
from livekit.wakeword.data.tts.voices import split_voices


@pytest.fixture(scope="module")
def cmu() -> dict[str, list[str]]:
    return get_cmudict()


class TestPhoneticNeighbours:
    def test_one_phoneme_neighbours(self, cmu: dict[str, list[str]]) -> None:
        found = phonetic_neighbours("zuck", cmu=cmu)
        assert {"duck", "luck", "zach"} <= set(found)
        target = _strip_stress(cmu["zuck"])
        for word in found:
            assert _edit_distance(target, _strip_stress(cmu[word]), 1) == 1, word

    def test_no_longer_words_containing_the_phonemes(self, cmu: dict[str, list[str]]) -> None:
        # The old unanchored regex search matched any word containing AH K
        found = set(phonetic_neighbours("zuck", cmu=cmu))
        assert "abdicate" not in found
        assert len(found) < 200
        assert all(len(_strip_stress(cmu[w])) in (2, 3, 4) for w in found)

    def test_unknown_word(self, cmu: dict[str, list[str]]) -> None:
        assert phonetic_neighbours("qzxqzx", cmu=cmu) == []

    def test_edit_distance(self) -> None:
        assert _edit_distance(("Z", "AH", "K"), ("D", "AH", "K"), 1) == 1
        assert _edit_distance(("Z", "AH", "K"), ("Z", "AH", "K", "S"), 1) == 1
        assert _edit_distance(("Z", "AH", "K"), ("D", "AH", "T"), 1) == 2  # capped at limit+1


class TestAdversarialPhrases:
    def test_contains_near_misses_and_halves(self) -> None:
        phrases = generate_adversarial_phrases(["hey zuck"], seed=0)
        assert {"hey duck", "hey luck", "zuck", "hey"} <= set(phrases)
        assert "hey zuck" not in phrases

    def test_deterministic_with_seed(self) -> None:
        a = generate_adversarial_phrases(["hey zuck"], seed=3)
        b = generate_adversarial_phrases(["hey zuck"], seed=3)
        assert a == b


class TestWordSwap:
    def test_swaps_each_word(self) -> None:
        phrases = generate_word_swap_phrases(["hey zuck"], words=["there", "siri", "zuck"])
        assert phrases == ["hey siri", "hey there", "siri zuck", "there zuck", "zuck zuck"]

    def test_skips_homophones(self) -> None:
        phrases = generate_word_swap_phrases(["hey there"], words=["their", "you"])
        assert "hey their" not in phrases
        assert "hey you" in phrases

    def test_default_words(self) -> None:
        phrases = generate_word_swap_phrases(["hey zuck"])
        assert {"hey there", "hey siri", "hey jack"} <= set(phrases)


class TestSchedules:
    def test_shares(self) -> None:
        schedule = build_phrase_schedule(
            [(["a", "b", "c", "d"], 0.7), (["custom"], 0.3), ([], 0.5)], 1000, seed=0
        )
        counts = Counter(schedule)
        assert len(schedule) == 1000
        assert counts["custom"] == pytest.approx(1000 * 0.3 / 1.0, abs=1)
        # Round-robin inside a group: near-equal counts
        assert max(counts[p] for p in "abcd") - min(counts[p] for p in "abcd") <= 1

    def test_deterministic_and_mixed(self) -> None:
        groups = [(["a", "b"], 0.5), (["c"], 0.5)]
        assert build_phrase_schedule(groups, 50, 1) == build_phrase_schedule(groups, 50, 1)
        assert build_phrase_schedule(groups, 50, 1)[:25].count("c") not in (0, 25)

    def test_empty(self) -> None:
        assert build_phrase_schedule([([], 1.0)], 10) == []

    def test_holdout_is_disjoint(self) -> None:
        train, test = split_holdout([f"p{i}" for i in range(50)], 0.2, seed=0)
        assert len(test) == 10 and len(train) == 40
        assert not set(train) & set(test)
        assert split_holdout(["only"], 0.2) == (["only"], ["only"])

    def test_negative_schedules(self, sample_config: WakeWordConfig) -> None:
        cfg = sample_config.model_copy(
            update={
                "target_phrases": ["hey zuck"],
                "n_samples": 1000,
                "n_samples_val": 200,
                "custom_negative_phrases": ["hey zach", "hey chuck"],
                "word_swap_share": 0.2,
            }
        )
        train, test = negative_phrase_schedules(cfg)
        assert len(train) == 1000 and len(test) == 200
        counts = Counter(train)
        assert counts["hey zach"] + counts["hey chuck"] == pytest.approx(300, abs=1)
        near = set(generate_adversarial_phrases(["hey zuck"], seed=cfg.seed))
        swaps = set(generate_word_swap_phrases(["hey zuck"])) - near - {"hey zach", "hey chuck"}
        assert sum(n for p, n in counts.items() if p in swaps) == pytest.approx(200, abs=1)
        # Generated phrases in the test split never appear in training
        generated_test = set(test) - set(cfg.custom_negative_phrases)
        assert generated_test and not generated_test & set(train)

    def test_share_validation(self, sample_config: WakeWordConfig) -> None:
        with pytest.raises(ValueError):
            WakeWordConfig(
                **{
                    **sample_config.model_dump(),
                    "custom_negative_share": 0.6,
                    "word_swap_share": 0.5,
                }
            )


class TestSplitPhraseParts:
    def test_pairs(self) -> None:
        parts = split_phrase_parts(["hey zuck", "ok my computer"], 4, seed=0)
        assert len(parts) == 8
        assert parts[:2] == ["hey", "zuck"]
        assert " ".join(parts[2:4]) == "ok my computer"

    def test_single_word_phrase(self) -> None:
        assert split_phrase_parts(["computer"], 4) == []


class TestVoices:
    def test_split_voices_disjoint(self) -> None:
        train, test = split_voices(904, 0.1, seed=0)
        assert len(test) == 90 and len(train) == 814
        assert not set(train) & set(test)
        assert split_voices(904, 0.1, seed=0) == (train, test)
        assert split_voices(1, 0.1) == ([0], [0])

    def test_speaker_pairs_spread(self) -> None:
        from livekit.wakeword.data.piper.synthesis import speaker_pair_at_index

        pool = list(range(904))
        pairs = [speaker_pair_at_index(pool, i, seed=0) for i in range(2000)]
        firsts = {a for a, _ in pairs}
        assert len(firsts) > 700  # in-order iteration used to stay within speakers 0..2
        assert speaker_pair_at_index(pool, 17, 0) == pairs[17]
        assert set(speaker_pair_at_index([5, 9], 3)) <= {5, 9}


class _RecordingTts:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def validate_artifacts(self) -> None:
        pass

    def synthesize_clips(
        self,
        phrases: list[str],
        output_dir: Path,
        n_samples: int,
        *,
        start_index: int = 0,
        batch_size: int = 50,
        holdout_voices: bool = False,
        voice_group_size: int = 1,
    ) -> list[Path]:
        self.calls.append(
            {
                "split": output_dir.name,
                "phrases": phrases,
                "n": n_samples,
                "holdout": holdout_voices,
                "group": voice_group_size,
            }
        )
        return []


def test_run_generate_splits(sample_config: WakeWordConfig, monkeypatch) -> None:
    tts = _RecordingTts()
    monkeypatch.setattr(gen, "get_tts_backend", lambda config: tts)
    cfg = sample_config.model_copy(
        update={
            "target_phrases": ["hey zuck"],
            "n_background_samples": 0,
            "n_background_samples_val": 0,
            "n_context_samples": 7,
            "n_split_phrase_samples": 3,
            "n_split_phrase_samples_val": 2,
        }
    )
    gen.run_generate(cfg)
    calls = {c["split"]: c for c in tts.calls}
    assert set(calls) == {
        "positive_train",
        "positive_test",
        "negative_train",
        "negative_test",
        "context_speech",
        "positive_train_parts",
        "positive_test_parts",
    }
    for split, call in calls.items():
        assert call["holdout"] == ("test" in split), split
    assert calls["positive_train_parts"]["group"] == 2
    assert calls["positive_train_parts"]["n"] == 6
    assert calls["positive_train_parts"]["phrases"] == ["hey", "zuck"] * 3
    assert calls["context_speech"]["n"] == 7
    assert len(calls["negative_train"]["phrases"]) == cfg.n_samples  # type: ignore[arg-type]


def test_remove_silence_keeps_inner_pause() -> None:
    from livekit.wakeword.data.piper.synthesis import remove_silence

    sr = 16000
    t = np.arange(int(0.4 * sr)) / sr
    burst = (0.3 * np.sign(np.sin(2 * np.pi * 150 * t)) * np.hanning(len(t))).astype(np.float32)
    lead, gap, tail = np.zeros(8000), np.zeros(6400), np.zeros(8000)
    x = np.concatenate([lead, burst, gap, burst, tail]).astype(np.float32)
    y = remove_silence(x)
    # Edges trimmed, but both bursts and the 400 ms pause between them survive
    assert len(y) < len(x) - 8000
    assert len(y) >= 2 * len(burst) + len(gap) - 2 * 480 * 3
    assert remove_silence(np.zeros(16000, dtype=np.float32)).shape == (16000,)


def test_espeak_keeps_clause_punctuation(monkeypatch) -> None:
    from livekit.wakeword.data.piper import synthesis

    ipa = {"hey": "hˈeɪ", "zuck": "zˈʌk", "hey there": "hˈeɪ ðˈɛɹ"}

    class _Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    monkeypatch.setattr(synthesis, "_find_espeak_ng", lambda: "espeak-ng")
    monkeypatch.setattr(synthesis.subprocess, "run", lambda cmd, **kw: _Result(ipa[cmd[-1]] + "\n"))
    assert synthesis._espeak_phonemize("hey, zuck") == "hˈeɪ, zˈʌk"
    assert synthesis._espeak_phonemize("hey there. hey, zuck!") == "hˈeɪ ðˈɛɹ. hˈeɪ, zˈʌk!"
