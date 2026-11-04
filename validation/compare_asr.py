"""Compare the Rust ASR worker output with the Python pipeline's raw transcript.  Standard library only.

    python validation/compare_asr.py "<meeting name>" path/to/rust.asr.json

The Python transcript (transcripts/<name>.txt) is a BASELINE, not ground truth: this measures how
much the two engines agree.  Speakers, case, punctuation and fillers (uh/um) are ignored.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINE_RE = re.compile(r"^\[\d\d:\d\d:\d\d\] Speaker \d+: (.*)$")
FILLERS = {"uh", "um", "uhm", "umm", "erm", "uhh", "hmm", "mm"}


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower().replace("\u2019", "'")) if w not in FILLERS]


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    name, rust_json = sys.argv[1], Path(sys.argv[2])
    ref_path = ROOT / "transcripts" / f"{name}.txt"
    ref = []
    for line in ref_path.read_text(encoding="utf-8").splitlines():
        m = LINE_RE.match(line)
        if m:
            ref += words(m[1])
    data = json.loads(rust_json.read_text(encoding="utf-8"))
    hyp = []
    for seg in data["segments"]:
        hyp += words(seg["text"])

    d = edit_distance(ref, hyp)
    print(f"meeting        : {name}")
    print(f"model          : {data.get('model')}")
    print(f"audio          : {data['audio_seconds']:.0f} s")
    print(f"Rust ASR time  : {data['elapsed_seconds']:.1f} s  ({data['audio_seconds'] / data['elapsed_seconds']:.1f}x real time)")
    print(f"words          : python {len(ref)}  |  rust {len(hyp)}")
    print(f"disagreement   : {d / max(1, len(ref)) * 100:.1f}% (word edit distance / python words; lower = more similar)")
    with_words = sum(1 for s in data["segments"] if s.get("words"))
    print(f"segments       : {len(data['segments'])} ({with_words} with word timestamps)")


if __name__ == "__main__":
    main()
