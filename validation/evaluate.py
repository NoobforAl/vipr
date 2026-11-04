"""Score pipeline output against the human-verified gold clips.  Standard library only.

    python validation/evaluate.py                      # raw + refined (Python pipeline) vs gold
    python validation/evaluate.py --hyp-dir out/rust   # also score another engine: <dir>/<clip id>.txt
    python validation/evaluate.py --allow-unverified   # smoke test (gold still marked unverified)

Metrics (per clip and average)
  WER             word error rate: (substitutions + deletions + insertions) / words in gold.
                  Lower is better. Case, punctuation and fillers (uh/um) are ignored.
  Speaker acc.    of the words both texts agree on, the share given the same speaker
                  (speaker names are matched automatically, so Speaker 1/2 may be swapped).
"""

from __future__ import annotations

import argparse
import difflib
import itertools
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE_RE = re.compile(r"^\[(\d+):(\d\d)\]\s*(Speaker \d+|[^:]{1,30}):\s*(.*)$")
FILLERS = {"uh", "um", "uhm", "umm", "erm", "uhh", "hmm", "mm"}


def parse(path: Path) -> list[tuple[str, str]]:
    """-> list of (speaker, word) for every word, in order."""
    words: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        for w in re.findall(r"[a-z0-9']+", m[4].lower().replace("’", "'")):
            if w not in FILLERS:
                words.append((m[3].strip(), w))
    return words


def is_verified(path: Path) -> bool:
    first = path.read_text(encoding="utf-8").splitlines()[:3]
    return any(re.match(r"#\s*STATUS:\s*verified", line, re.I) for line in first)


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def speaker_accuracy(gold: list[tuple[str, str]], hyp: list[tuple[str, str]]) -> float | None:
    gw, hw = [w for _, w in gold], [w for _, w in hyp]
    pairs: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, gw, hw, autojunk=False).get_opcodes():
        if tag == "equal":
            pairs += [(gold[i1 + k][0], hyp[j1 + k][0]) for k in range(i2 - i1)]
    if not pairs:
        return None
    g_labels = sorted({g for g, _ in pairs})
    h_labels = sorted({h for _, h in pairs})
    best = 0
    # try every way of matching hypothesis speakers to gold speakers (few speakers => cheap)
    for perm in itertools.permutations(h_labels, min(len(h_labels), len(g_labels))) if len(h_labels) <= 6 else []:
        mapping = dict(zip(g_labels, perm))
        best = max(best, sum(1 for g, h in pairs if mapping.get(g) == h))
    return best / len(pairs)


def score(gold, hyp) -> tuple[float, float | None]:
    ref = [w for _, w in gold]
    wer = edit_distance(ref, [w for _, w in hyp]) / max(1, len(ref))
    return wer, speaker_accuracy(gold, hyp)


def pct(x: float | None) -> str:
    return "  n/a " if x is None else f"{x * 100:5.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(HERE / "clips"), help="folder with <id>.gold.txt etc.")
    ap.add_argument("--hyp-dir", help="extra engine output folder with <id>.txt files")
    ap.add_argument("--allow-unverified", action="store_true")
    args = ap.parse_args()

    clips = Path(args.dir)
    sources = {"raw": lambda cid: clips / f"{cid}.raw.txt",
               "refined": lambda cid: clips / f"{cid}.refined.txt"}
    if args.hyp_dir:
        sources["engine"] = lambda cid: Path(args.hyp_dir) / f"{cid}.txt"

    rows, skipped = [], []
    for gold_path in sorted(clips.glob("c*.gold.txt")):
        cid = gold_path.name.split(".")[0]
        if not (args.allow_unverified or is_verified(gold_path)):
            skipped.append(cid)
            continue
        gold = parse(gold_path)
        if not gold:
            skipped.append(cid)
            continue
        row = {"id": cid, "words": len(gold)}
        for name, locate in sources.items():
            p = locate(cid)
            row[name] = score(gold, parse(p)) if p.exists() else None
        rows.append(row)

    if skipped:
        print(f"Skipped (not verified yet): {', '.join(skipped)}\n")
    if not rows:
        sys.exit("No verified clips. Correct validation/clips/*.gold.txt and set '# STATUS: verified'.")

    names = list(sources)
    print(f"{'clip':<5}{'words':>6} | " + " | ".join(f"{n:^15}" for n in names))
    print(f"{'':<5}{'':>6} | " + " | ".join(f"{'WER':>7} {'spk':>7}" for _ in names))
    totals = {n: [0.0, 0.0, 0, 0.0, 0] for n in names}   # wer*words, words, spk sum, spk n
    for r in rows:
        cells = []
        for n in names:
            s = r[n]
            cells.append(f"{pct(s[0]):>7} {pct(s[1]):>7}" if s else f"{'-':>7} {'-':>7}")
            if s:
                t = totals[n]
                t[0] += s[0] * r["words"]; t[1] += r["words"]
                if s[1] is not None:
                    t[2] += s[1] * r["words"]; t[3] += r["words"]
        print(f"{r['id']:<5}{r['words']:>6} | " + " | ".join(cells))
    print("-" * (14 + 18 * len(names)))
    cells = []
    for n in names:
        t = totals[n]
        cells.append(f"{pct(t[0] / t[1] if t[1] else None):>7} {pct(t[2] / t[3] if t[3] else None):>7}")
    print(f"{'ALL':<5}{sum(r['words'] for r in rows):>6} | " + " | ".join(cells))
    if "raw" in totals and "refined" in totals and totals["raw"][1] and totals["refined"][1]:
        d = totals["refined"][0] / totals["refined"][1] - totals["raw"][0] / totals["raw"][1]
        print(f"\nLLM refine changes WER by {d * 100:+.1f} points ({'better' if d < 0 else 'worse' if d > 0 else 'same'}).")


if __name__ == "__main__":
    main()
