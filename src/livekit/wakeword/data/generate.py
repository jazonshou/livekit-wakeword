"""Synthetic data generation: pluggable TTS + adversarial negatives (default: Piper VITS)."""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path

from ..config import WakeWordConfig
from .piper.text import expand_unknown_words, get_cmudict
from .tts import SpeechSynthesizer, get_tts_backend
from .tts.piper_backend import PiperVitsBackend
from .wordlists import COMMON_SWAP_WORDS, CONTEXT_SENTENCES

logger = logging.getLogger(__name__)

# Matches original clips (clip_000000.wav) but NOT augmented variants (clip_000000_r1.wav)
_ORIGINAL_CLIP_RE = re.compile(r"^clip_\d{6}\.wav$")


def _count_original_clips(directory: Path) -> int:
    """Count ``clip_######.wav`` files in *directory*, excluding augmented variants."""
    if not directory.is_dir():
        return 0
    return sum(1 for f in directory.iterdir() if _ORIGINAL_CLIP_RE.match(f.name))


_STRESS_RE = re.compile(r"\d+")


def _strip_stress(phones: list[str]) -> tuple[str, ...]:
    return tuple(_STRESS_RE.sub("", p) for p in phones)


_PRON_INDEX: dict[int, list[tuple[str, tuple[str, ...]]]] | None = None


def _pron_index() -> dict[int, list[tuple[str, tuple[str, ...]]]]:
    """CMUDict words grouped by phoneme count (stress stripped), loaded once."""
    global _PRON_INDEX
    if _PRON_INDEX is None:
        index: dict[int, list[tuple[str, tuple[str, ...]]]] = {}
        for word, phones in sorted(get_cmudict().items()):
            if not word.isalpha():  # skip "a.", "'em", "x-ray" style entries
                continue
            pron = _strip_stress(phones)
            index.setdefault(len(pron), []).append((word, pron))
        _PRON_INDEX = index
    return _PRON_INDEX


def _edit_distance(a: tuple[str, ...], b: tuple[str, ...], limit: int) -> int:
    """Phoneme-level Levenshtein distance, returning ``limit + 1`` once it exceeds *limit*."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, pa in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for k, pb in enumerate(b, 1):
            cur[k] = min(prev[k] + 1, cur[k - 1] + 1, prev[k - 1] + (pa != pb))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def phonetic_neighbours(
    word: str,
    max_distance: int | None = None,
    cmu: dict[str, list[str]] | None = None,
) -> list[str]:
    """Words whose CMUDict pronunciation is 1..*max_distance* phoneme edits from *word*.

    An edit substitutes, inserts or deletes one phoneme (stress ignored), so for
    "zuck" (``Z AH K``) this returns "duck", "zach", "zucker", "suck" ... and not
    every word that merely contains ``AH K``. Homophones of *word* and single
    letters are excluded.

    Args:
        word: Lowercase word in CMUDict.
        max_distance: Max phoneme edits (``None`` = 1 for words under 5 phonemes,
            2 for longer ones).
        cmu: CMUDict mapping (loaded if not given).
    """
    cmu = cmu if cmu is not None else get_cmudict()
    phones = cmu.get(word)
    if not phones:
        return []
    target = _strip_stress(phones)
    if max_distance is None:
        max_distance = 1 if len(target) < 5 else 2

    found: list[str] = []
    index = _pron_index()
    for length in range(len(target) - max_distance, len(target) + max_distance + 1):
        for cand, pron in index.get(length, []):
            if pron == target or cand == word or len(cand) < 2:
                continue
            if _edit_distance(target, pron, max_distance) <= max_distance:
                found.append(cand)
    return sorted(found)


def generate_adversarial_phrases(
    target_phrases: list[str],
    n_phrases: int | None = None,
    include_partial_phrase: float = 1.0,
    include_input_words: float = 0.2,
    max_distance: int | None = None,
    seed: int | None = None,
) -> list[str]:
    """Generate phonetically similar phrases to the target using CMUDict.

    Each word of the phrase is replaced in turn by its phonetic neighbours
    (:func:`phonetic_neighbours`: words one phoneme edit away, two for long
    words), so "hey computer" gives "hey commuter", "hay computer" and so on.

    Unknown words (not in CMUDict) are split into known subwords when possible,
    e.g. "livekit" → "live" + "kit", allowing substitutions on both parts.

    When *n_phrases* is ``None`` (the default) all unique adversarial phrases
    are returned — no cap is applied.

    Args:
        target_phrases: Target wake word phrases.
        n_phrases: Maximum number of adversarial phrases to return
            (``None`` = no cap).
        include_partial_phrase: Probability of generating partial phrases
            (each word removed in turn).
        include_input_words: Probability of including individual original
            words as adversarial entries.
        max_distance: Max phoneme edits per substituted word (see
            :func:`phonetic_neighbours`).
        seed: Seed for the random choices and the output order (``None`` = unseeded).
    """
    rng = random.Random(seed)
    cmu = get_cmudict()
    adversarial: list[str] = []

    for phrase in target_phrases:
        raw_words = phrase.lower().split()
        words = expand_unknown_words(raw_words, cmu)

        for word_idx, word in enumerate(words):
            for replacement in phonetic_neighbours(word, max_distance, cmu):
                new_words = words.copy()
                new_words[word_idx] = replacement
                adversarial.append(" ".join(new_words))

        # Partial phrase adversarials
        if include_partial_phrase > 0 and len(words) > 1 and rng.random() < include_partial_phrase:
            for i in range(len(words)):
                partial = " ".join(words[:i] + words[i + 1 :])
                if partial:
                    adversarial.append(partial)

        # Include original words individually
        if include_input_words > 0:
            for word in words:
                if rng.random() < include_input_words:
                    adversarial.append(word)

    # Deduplicate (sorted, so the order doesn't depend on hash seeds) and remove targets
    target_set = {p.lower() for p in target_phrases}
    target_set |= {" ".join(expand_unknown_words(p.lower().split(), cmu)) for p in target_phrases}
    adversarial = [p for p in sorted(set(adversarial)) if p not in target_set]
    rng.shuffle(adversarial)
    if n_phrases is not None:
        adversarial = adversarial[:n_phrases]
    return adversarial


def generate_word_swap_phrases(
    target_phrases: list[str],
    words: list[str] | None = None,
) -> list[str]:
    """Phrases with one word of each target phrase swapped for a common word.

    "hey computer" gives "hey there", "hey siri", "jack computer" ... so the model
    learns that one half of the phrase is not enough. Swaps that sound like the
    original word (homophones) or reproduce a target phrase are skipped.

    Args:
        target_phrases: Target wake word phrases.
        words: Words to swap in (``None`` = :data:`~.wordlists.COMMON_SWAP_WORDS`).
    """
    cmu = get_cmudict()
    vocab = sorted({w.lower() for w in (words if words is not None else COMMON_SWAP_WORDS)})
    targets = {" ".join(p.lower().split()) for p in target_phrases}
    out: set[str] = set()
    for phrase in target_phrases:
        orig = phrase.lower().split()
        for idx, word in enumerate(orig):
            word_pron = cmu.get(word)
            for swap in vocab:
                if swap == word:
                    continue
                swap_pron = cmu.get(swap)
                if word_pron and swap_pron and _strip_stress(swap_pron) == _strip_stress(word_pron):
                    continue
                new = orig.copy()
                new[idx] = swap
                candidate = " ".join(new)
                if candidate not in targets:
                    out.add(candidate)
    return sorted(out)


def split_holdout(
    phrases: list[str], fraction: float, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Split *phrases* into disjoint (train, test) lists with a seeded shuffle.

    With fewer than two phrases (or *fraction* 0) both lists are the full list.
    """
    unique = sorted(set(phrases))
    if len(unique) < 2 or fraction <= 0:
        return unique, unique
    random.Random(seed).shuffle(unique)
    n_test = min(len(unique) - 1, max(1, round(len(unique) * fraction)))
    return unique[n_test:], unique[:n_test]


def build_phrase_schedule(
    groups: list[tuple[list[str], float]],
    n_samples: int,
    seed: int = 0,
) -> list[str]:
    """Return one phrase per clip so each group gets its share of *n_samples*.

    *groups* is ``(phrases, share)`` pairs; shares of non-empty groups are
    normalized to sum to 1. Inside a group phrases are used round-robin, so each
    gets an equal number of clips. The clip order is shuffled with *seed*, so
    batches mix groups and the schedule is identical on every run (resume-safe).
    """
    active = [(sorted(set(p)), share) for p, share in groups if p and share > 0]
    if not active or n_samples <= 0:
        return []
    total = sum(share for _, share in active)
    counts = [int(n_samples * share / total) for _, share in active]
    # Hand the rounding remainder to the largest shares
    for k in sorted(range(len(active)), key=lambda k: -active[k][1])[: n_samples - sum(counts)]:
        counts[k] += 1

    rng = random.Random(seed)
    schedule: list[str] = []
    for (phrases, _), count in zip(active, counts):
        order = phrases.copy()
        rng.shuffle(order)
        schedule.extend(order[i % len(order)] for i in range(count))
    rng.shuffle(schedule)
    return schedule


def split_phrase_parts(target_phrases: list[str], n_pairs: int, seed: int = 0) -> list[str]:
    """Return ``2 * n_pairs`` texts: the left and right half of a target phrase per pair.

    Each pair splits a multi-word phrase at a random word boundary
    ("hey | computer"); the halves are synthesized in one voice and joined with a
    pause during augmentation. Single-word phrases can't be split and are skipped.
    """
    multi = [p.lower().split() for p in target_phrases if len(p.split()) > 1]
    if not multi or n_pairs <= 0:
        return []
    rng = random.Random(seed)
    parts: list[str] = []
    for j in range(n_pairs):
        words = multi[j % len(multi)]
        cut = rng.randint(1, len(words) - 1)
        parts.extend([" ".join(words[:cut]), " ".join(words[cut:])])
    return parts


def synthesize_clips(
    phrases: list[str],
    output_dir: Path,
    n_samples: int,
    vits_model_path: Path | None = None,
    noise_scales: list[float] | None = None,
    noise_scale_ws: list[float] | None = None,
    length_scales: list[float] | None = None,
    slerp_weights: list[float] | None = None,
    max_speakers: int | None = None,
    batch_size: int = 50,
    start_index: int = 0,
) -> list[Path]:
    """Synthesize speech clips using Piper VITS + SLERP (library / test helper).

    Returns list of paths to generated .wav files.
    """
    if vits_model_path is None or not vits_model_path.exists():
        raise FileNotFoundError(
            f"VITS model not found at {vits_model_path}. "
            "Cannot generate audio — refusing to produce silent placeholders. "
            "Download the model first: livekit-wakeword setup --config <your.yaml>"
        )

    backend = PiperVitsBackend(
        model_path=vits_model_path,
        noise_scales=noise_scales if noise_scales is not None else [0.98],
        noise_scale_ws=noise_scale_ws if noise_scale_ws is not None else [0.98],
        length_scales=length_scales if length_scales is not None else [0.75, 1.0, 1.25],
        slerp_weights=slerp_weights if slerp_weights is not None else [0.2, 0.35, 0.5, 0.65, 0.8],
        max_speakers=max_speakers,
    )
    return backend.synthesize_clips(
        phrases=phrases,
        output_dir=output_dir,
        n_samples=n_samples,
        start_index=start_index,
        batch_size=batch_size,
    )


def _generate_background_clips(
    config: WakeWordConfig,
    split_name: str,
    n_samples: int,
) -> None:
    """Generate background noise clips by randomly sampling from background audio.

    Short source files are tiled with random offsets and occasional reversal
    to avoid audible periodicity.  Always produces exactly *n_samples* clips
    regardless of how much source audio is available.
    """
    import numpy as np
    import soundfile as sf
    from tqdm import tqdm

    bg_paths: list[Path] = []
    for bg_dir in config.augmentation.background_paths:
        d = Path(bg_dir)
        if d.exists():
            bg_paths.extend(d.glob("**/*.wav"))

    if not bg_paths:
        logger.info("No background noise files found, skipping %s", split_name)
        return

    sample_rate = 16000
    chunk_samples = int(config.augmentation.clip_duration * sample_rate)

    out_dir = config.model_output_dir / split_name
    existing = _count_original_clips(out_dir)
    if existing >= n_samples:
        logger.info(
            "Split %s already complete (%d/%d clips), skipping",
            split_name,
            existing,
            n_samples,
        )
        return
    if existing > 0:
        logger.info("Resuming split %s from clip %d / %d", split_name, existing, n_samples)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load all background audio
    all_audio: list[np.ndarray] = []
    for bp in bg_paths:
        audio, sr = sf.read(str(bp))
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)
        if sr != sample_rate:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        all_audio.append(audio)

    logger.info(
        "Generating %d %s clips from %d source files",
        n_samples - existing,
        split_name,
        len(all_audio),
    )

    for i in tqdm(range(existing, n_samples), desc=f"Background ({split_name})", unit="clip"):
        # Pick a random source file
        audio = random.choice(all_audio)

        # Tile short files with varied segments to fill clip duration
        if len(audio) < chunk_samples:
            segments: list[np.ndarray] = []
            n = len(audio)
            while sum(len(s) for s in segments) < chunk_samples:
                start = random.randint(0, n - 1)
                seg = np.roll(audio, -start)
                if random.random() < 0.5:
                    seg = seg[::-1]
                segments.append(seg)
            audio = np.concatenate(segments)

        # Random offset into the (possibly tiled) audio
        max_start = len(audio) - chunk_samples
        start = random.randint(0, max(0, max_start))
        clip = audio[start : start + chunk_samples]

        out_path = out_dir / f"clip_{i:06d}.wav"
        sf.write(str(out_path), clip, sample_rate)

    logger.info("Wrote %d background clips to %s", n_samples, out_dir)


def _synthesize_split(
    tts: SpeechSynthesizer,
    split_dir: Path,
    phrases: list[str],
    n_target: int,
    batch_size: int,
    *,
    holdout_voices: bool = False,
    voice_group_size: int = 1,
) -> None:
    """Synthesize *split_dir* up to *n_target* clips, resuming from existing clips."""
    split_name = split_dir.name
    existing = _count_original_clips(split_dir)
    if existing >= n_target:
        logger.info(
            "Split %s already complete (%d/%d clips), skipping", split_name, existing, n_target
        )
        return
    if existing > 0:
        logger.info("Resuming split %s from clip %d / %d", split_name, existing, n_target)
    else:
        logger.info("Generating %d %s clips...", n_target, split_name)
    tts.synthesize_clips(
        phrases=phrases,
        output_dir=split_dir,
        n_samples=n_target,
        start_index=existing,
        batch_size=batch_size,
        holdout_voices=holdout_voices,
        voice_group_size=voice_group_size,
    )


def negative_phrase_schedules(config: WakeWordConfig) -> tuple[list[str], list[str]]:
    """Per-clip phrase lists for ``negative_train`` and ``negative_test``.

    Three groups share the clips: phonetic near-misses (the rest), word swaps
    (``word_swap_share``) and ``custom_negative_phrases`` (``custom_negative_share``).
    Generated phrases are split so ``negative_test`` uses phrases training never
    saw; custom phrases are used in both splits.
    """
    seed = config.seed
    holdout = config.negative_test_holdout
    custom = list(config.custom_negative_phrases)
    # Custom phrases get exactly their own share, so keep them out of the generated groups
    custom_set = {" ".join(p.lower().split()) for p in custom}
    near = generate_adversarial_phrases(target_phrases=config.target_phrases, seed=seed)
    near_train, near_test = split_holdout([p for p in near if p not in custom_set], holdout, seed)
    swap_train: list[str] = []
    swap_test: list[str] = []
    if config.word_swap_share > 0:
        swap = generate_word_swap_phrases(config.target_phrases, config.word_swap_words)
        exclude = custom_set | set(near)
        swap = [p for p in swap if p not in exclude]
        swap_train, swap_test = split_holdout(swap, holdout, seed + 1)
    near_share = max(0.0, 1.0 - config.word_swap_share - config.custom_negative_share)

    def schedule(near_p: list[str], swap_p: list[str], n: int, split_seed: int) -> list[str]:
        groups = [
            (near_p, near_share),
            (swap_p, config.word_swap_share),
            (custom, config.custom_negative_share),
        ]
        result = build_phrase_schedule(groups, n, split_seed)
        if not result and n > 0:
            # Every group with a share is empty: fall back to whatever phrases exist
            result = build_phrase_schedule([(near_p + swap_p + custom, 1.0)], n, split_seed)
        return result

    train = schedule(near_train, swap_train, config.n_samples, seed)
    test = schedule(near_test, swap_test, config.n_samples_val, seed + 1)
    logger.info(
        "Negative phrases: %d/%d near-miss (train/test), %d/%d word swap, %d custom",
        len(near_train),
        len(near_test),
        len(swap_train),
        len(swap_test),
        len(custom),
    )
    return train, test


def run_generate(config: WakeWordConfig) -> None:
    """Run the full generate pipeline for a wake word config.

    Supports resuming: counts existing ``clip_######.wav`` files in each split
    directory and skips completed splits or resumes partial ones from the
    existing count. Phrase lists and voices are seeded by ``config.seed``, so a
    resumed split continues the same sequence.

    Test splits (``*_test``) use TTS voices held out from training
    (``test_voice_fraction``), so test scores measure unseen voices.
    """
    model_dir = config.model_output_dir
    tts = get_tts_backend(config)
    tts.validate_artifacts()
    bs = config.tts_batch_size

    # --- Positive splits ---
    _synthesize_split(
        tts, model_dir / "positive_train", config.target_phrases, config.n_samples, bs
    )
    _synthesize_split(
        tts,
        model_dir / "positive_test",
        config.target_phrases,
        config.n_samples_val,
        bs,
        holdout_voices=True,
    )

    # --- Adversarial negative splits ---
    neg_train_dir = model_dir / "negative_train"
    neg_test_dir = model_dir / "negative_test"

    # Skip adversarial phrase generation entirely if both negative splits are complete
    if (
        _count_original_clips(neg_train_dir) >= config.n_samples
        and _count_original_clips(neg_test_dir) >= config.n_samples_val
    ):
        logger.info("Both negative splits already complete, skipping adversarial generation")
    else:
        logger.info("Generating adversarial negative phrases...")
        train_phrases, test_phrases = negative_phrase_schedules(config)
        if not train_phrases:
            logger.warning(
                "No adversarial phrases generated; using common English filler phrases as fallback"
            )
            train_phrases = test_phrases = ["hello", "okay", "hey", "stop", "go", "yes", "no"]
        _synthesize_split(tts, neg_train_dir, train_phrases, config.n_samples, bs)
        _synthesize_split(
            tts,
            neg_test_dir,
            test_phrases or train_phrases,
            config.n_samples_val,
            bs,
            holdout_voices=True,
        )

    # --- Speech heard before the phrase (augmentation context) ---
    if config.n_context_samples > 0:
        _synthesize_split(
            tts, model_dir / "context_speech", list(CONTEXT_SENTENCES), config.n_context_samples, bs
        )

    # --- Phrase halves in one voice, joined with a pause during augmentation ---
    for split_name, n_pairs, holdout in (
        ("positive_train_parts", config.n_split_phrase_samples, False),
        ("positive_test_parts", config.n_split_phrase_samples_val, True),
    ):
        if n_pairs <= 0:
            continue
        parts = split_phrase_parts(config.target_phrases, n_pairs, config.seed + int(holdout))
        if not parts:
            logger.warning("No multi-word target phrase to split; skipping %s", split_name)
            continue
        _synthesize_split(
            tts,
            model_dir / split_name,
            parts,
            len(parts),
            bs,
            holdout_voices=holdout,
            voice_group_size=2,
        )

    # --- Background noise splits ---
    if config.n_background_samples > 0:
        _generate_background_clips(config, "background_train", config.n_background_samples)
    if config.n_background_samples_val > 0:
        _generate_background_clips(config, "background_test", config.n_background_samples_val)
