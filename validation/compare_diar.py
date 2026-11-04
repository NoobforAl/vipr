"""Compare the Rust diarization worker with the Python pipeline's speaker labels.  Standard library only.

    python validation/compare_diar.py "<meeting name>" path/to/meeting.diar.json [path/to/meeting.asr.json]

The Python transcript (transcripts/<name>.txt, lines "[HH:MM:SS] Speaker N: text") is a BASELINE, not
ground truth.  It has line start times only, so the best comparison needs word times: pass the Rust
ASR output too (asr-worker --out ...).  Its words are aligned to the Python words, every matched word gets
the Python line's speaker, and that is compared with the diarization turn that covers the word's time
(largest overlap, else nearest turn - the same rule as engine::diar::assign_speaker).

Without the ASR file only a rough line-interval comparison is possible (each line runs until the next one).
"""

from __future__ import annotations

import difflib
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINE_RE = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\] Speaker (\d+): (.*)$")
TOKEN_RE = re.compile(r"[a-z0-9']+")
FILLERS = {"uh", "um", "uhm", "umm", "erm", "uhh", "hmm", "mm"}


def tokens(text: str) -> list[str]:
    return [
        w for w in TOKEN_RE.findall(text.lower().replace("’", "'")) if w not in FILLERS
    ]


def python_lines(name: str) -> list[tuple[float, int, list[str]]]:
    out = []
    for line in (
        (ROOT / "transcripts" / f"{name}.txt").read_text(encoding="utf-8").splitlines()
    ):
        m = LINE_RE.match(line)
        if m:
            out.append(
                (int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]), int(m[4]), tokens(m[5]))
            )
    return out


def speaker_at(turns: list[dict], t0: float, t1: float) -> int | None:
    """Same rule as engine::diar::assign_speaker."""
    if not turns:
        return None
    t1 = max(t0, t1)
    overlap: dict[int, float] = {}
    for t in turns:
        o = min(t1, t["end"]) - max(t0, t["start"])
        if o > 0:
            overlap[t["speaker"]] = overlap.get(t["speaker"], 0.0) + o
    if overlap:
        return max(overlap.items(), key=lambda kv: kv[1])[0]
    mid = (t0 + t1) / 2

    def dist(t: dict) -> float:
        return (
            t["start"] - mid
            if mid < t["start"]
            else (mid - t["end"] if mid > t["end"] else 0.0)
        )

    return min(turns, key=dist)["speaker"]


def best_agreement(conf: dict[tuple[int, int], float]) -> tuple[float, dict[int, int]]:
    """conf[(python_speaker, rust_speaker)] = seconds.  Best one-to-one matching of speaker names."""
    py = sorted({p for p, _ in conf})
    rs = sorted({r for _, r in conf})
    total = sum(conf.values()) or 1.0
    best, best_map = -1.0, {}
    for perm in itertools.permutations(rs, min(len(py), len(rs))):
        agree = sum(conf.get((p, r), 0.0) for p, r in zip(py, perm))
        if agree > best:
            best, best_map = agree, dict(zip(py, perm))
    return best / total, best_map


def main() -> None:
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    name, diar_path = sys.argv[1], Path(sys.argv[2])
    asr_path = Path(sys.argv[3]) if len(sys.argv) == 4 else None
    diar = json.loads(diar_path.read_text(encoding="utf-8"))
    turns = diar["turns"]
    lines = python_lines(name)

    print(f"meeting         : {name}")
    print(f"audio           : {diar['audio_seconds']:.0f} s")
    print(
        f"diarization time: {diar['elapsed_seconds']:.1f} s  ({diar['audio_seconds'] / max(diar['elapsed_seconds'], 1e-9):.1f}x real time)"
    )
    spk_time: dict[int, float] = {}
    for t in turns:
        spk_time[t["speaker"]] = spk_time.get(t["speaker"], 0.0) + t["end"] - t["start"]
    total = sum(spk_time.values()) or 1.0
    print(f"turns           : {len(turns)}  |  speakers: {len(spk_time)}")
    py_speakers = sorted({spk for _, spk, _ in lines})
    print(
        f"speakers        : detected {len(spk_time)} | Python pipeline (forced 2) {len(py_speakers)}"
    )
    print(
        "speaker split   : "
        + ", ".join(
            f"S{k + 1} {v / total * 100:.0f}% ({v:.0f}s)"
            for k, v in sorted(spk_time.items())
        )
    )

    conf: dict[tuple[int, int], float] = {}
    if asr_path:
        words = []  # (token, start, end)
        for seg in json.loads(asr_path.read_text(encoding="utf-8"))["segments"]:
            for w in seg.get("words", []):
                toks = tokens(w["text"])
                for k, tok in enumerate(
                    toks
                ):  # a word may hold several tokens (what's up -> "what's")
                    span = (w["end"] - w["start"]) / len(toks)
                    words.append(
                        (tok, w["start"] + k * span, w["start"] + (k + 1) * span)
                    )
        ref_tokens: list[str] = []
        ref_speaker: list[int] = []
        for _, spk, toks in lines:
            ref_tokens += toks
            ref_speaker += [spk] * len(toks)
        sm = difflib.SequenceMatcher(
            None, ref_tokens, [w[0] for w in words], autojunk=False
        )
        matched = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "equal":
                continue
            for i, j in zip(range(i1, i2), range(j1, j2)):
                _, s, e = words[j]
                rust = speaker_at(turns, s, e)
                if rust is None:
                    continue
                conf[(ref_speaker[i], rust)] = conf.get(
                    (ref_speaker[i], rust), 0.0
                ) + max(e - s, 0.05)
                matched += 1
        agree, mapping = best_agreement(conf)
        print(
            f"compared words  : {matched} of {len(ref_tokens)} python words (aligned to Rust ASR words)"
        )
        print(
            f"AGREEMENT       : {agree * 100:.1f}% of word time has the same speaker as the Python pipeline"
        )
        print(
            "speaker mapping : "
            + ", ".join(
                f"python S{p} = rust S{r + 1}" for p, r in sorted(mapping.items())
            )
        )
        # which direction do the errors go
        for (p, r), v in sorted(conf.items()):
            print(f"   python S{p} / rust S{r + 1}: {v:7.0f} s")
    else:
        for k, (start, spk, toks) in enumerate(lines):
            end = lines[k + 1][0] if k + 1 < len(lines) else start + 5
            end = min(
                end, start + max(1.0, len(toks) * 0.45)
            )  # a line rarely fills the whole gap to the next one
            t = start
            while t < end:
                rust = speaker_at(turns, t, min(t + 1.0, end))
                if rust is not None:
                    conf[(spk, rust)] = conf.get((spk, rust), 0.0) + min(1.0, end - t)
                t += 1.0
        agree, mapping = best_agreement(conf)
        print(
            f"AGREEMENT (rough, line intervals): {agree * 100:.1f}%  - pass the ASR json for word-level numbers"
        )


if __name__ == "__main__":
    main()
