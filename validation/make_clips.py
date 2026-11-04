"""Pick a small, diverse validation set from the meetings we already transcribed.

Pure standard library (no GPU, no venv).  Run from the project root:
    python validation/make_clips.py

What it does
  1. scans transcripts/*.txt (+ refined/*.changes.log when present) in ~2.5 minute windows
  2. picks one window per "difficulty" category, preferring different meetings
  3. cuts the matching audio from audio/*.wav  -> validation/clips/<id>.wav
  4. writes, per clip: <id>.raw.txt / <id>.refined.txt (what the pipeline produced) and
     <id>.gold.txt (a DRAFT you correct by listening; marked unverified)

It never overwrites an existing *.gold.txt, and reuses validation/clips.json if it exists,
so it is safe to re-run (e.g. after the last meetings get refined).
"""

from __future__ import annotations

import json
import re
import wave
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "validation"
CLIPS_DIR = OUT / "clips"
TARGET_SECONDS = 150          # ~2.5 minutes per clip
MAX_SECONDS = 210
LINE_RE = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\] (Speaker \d+): (.*)$")
CHG_RE = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\] #(\d+) (TEXT|SPEAKER|REJECTED)")

CATEGORIES = {
    "turn-taking": "fast back-and-forth between the two speakers (diarization stress test)",
    "short-interjections": "many very short turns like 'okay', 'yeah' (speaker-label risk)",
    "jargon": "project / tool names and technical terms (glossary + LLM word-fix test)",
    "monologue": "one person talking for a long time (punctuation, long-text handling)",
    "sparse": "little speech spread over a long time (silence / hallucination test)",
    "llm-risk": "places where the LLM changed a lot or was rejected (checks the LLM guards)",
}


@dataclass
class Utt:
    idx: int
    start: float
    speaker: str
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())


def secs(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse_transcript(path: Path) -> list[Utt]:
    utts: list[Utt] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LINE_RE.match(line)
        if m:
            utts.append(Utt(len(utts), secs(*m.groups()[:3]), m[4], m[5]))
    return utts


def parse_changes(path: Path) -> list[tuple[int, str]]:
    if not path.exists():
        return []
    return [(int(m[4]), m[5]) for line in path.read_text(encoding="utf-8").splitlines()
            if (m := CHG_RE.match(line))]


def load_glossary() -> list[re.Pattern]:
    path = ROOT / "glossary.txt"
    terms: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            terms += [t.strip() for t in line.split("#", 1)[0].split(",") if t.strip()]
    return [re.compile(r"\b" + re.escape(t.lower()) + r"\b") for t in dict.fromkeys(terms)]


def fmt(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


# --------------------------------------------------------------------------- windows
def window_end_time(utts: list[Utt], e: int) -> float:
    return utts[e + 1].start - 0.2 if e + 1 < len(utts) else utts[e].start + 10


def windows_for(meeting: str, utts: list[Utt], changes: list[tuple[int, str]], glossary):
    kinds_by_idx: dict[int, list[str]] = {}
    for idx, kind in changes:
        kinds_by_idx.setdefault(idx, []).append(kind)
    for s in range(len(utts)):
        e = s
        while e + 1 < len(utts) and window_end_time(utts, e) - utts[s].start < TARGET_SECONDS:
            e += 1
        dur = window_end_time(utts, e) - utts[s].start
        if dur < TARGET_SECONDS * 0.8 or dur > MAX_SECONDS:
            continue
        chunk = utts[s:e + 1]
        words = sum(u.words for u in chunk)
        if words < 40:
            continue
        low = " ".join(u.text.lower() for u in chunk)
        kinds = [k for u in chunk for k in kinds_by_idx.get(u.idx, [])]
        yield {
            "meeting": meeting, "first": s, "last": e,
            "start": utts[s].start - 0.3, "end": window_end_time(utts, e), "dur": dur,
            "words": words,
            "wps": words / dur,
            "switches": sum(1 for a, b in zip(chunk, chunk[1:]) if a.speaker != b.speaker),
            "short": sum(1 for u in chunk if u.words <= 3),
            "jargon": sum(len(g.findall(low)) for g in glossary),
            "text_edits": kinds.count("TEXT"),
            "speaker_edits": kinds.count("SPEAKER"),
            "rejected": kinds.count("REJECTED"),
        }


SCORES = {
    "turn-taking": lambda w: w["switches"] / (w["dur"] / 60) if w["words"] >= 150 else 0,
    "short-interjections": lambda w: w["short"] if w["words"] >= 100 else 0,
    "jargon": lambda w: w["jargon"] * 3 + w["text_edits"],
    "monologue": lambda w: w["words"] if w["switches"] <= 3 else 0,
    "sparse": lambda w: (1 / w["wps"]) if w["words"] >= 40 else 0,
    "llm-risk": lambda w: w["rejected"] * 2 + w["speaker_edits"] * 4 + w["text_edits"] * 0.5,
}


def choose(all_windows: list[dict]) -> list[dict]:
    chosen: list[dict] = []
    used_meetings: dict[str, int] = {}
    for cat in CATEGORIES:
        ranked = sorted(all_windows, key=SCORES[cat], reverse=True)
        pick = None
        for w in ranked:
            if SCORES[cat](w) <= 0:
                break
            overlaps = any(c["meeting"] == w["meeting"] and not (w["end"] < c["start"] or w["start"] > c["end"])
                           for c in chosen)
            if overlaps:
                continue
            # prefer a meeting we haven't used yet, unless it is clearly worse
            if used_meetings.get(w["meeting"], 0) >= 1:
                alt = next((x for x in ranked if x["meeting"] not in used_meetings and SCORES[cat](x) >= 0.7 * SCORES[cat](w)
                            and not any(c["meeting"] == x["meeting"] and not (x["end"] < c["start"] or x["start"] > c["end"]) for c in chosen)),
                           None)
                if alt:
                    w = alt
            pick = w
            break
        if pick:
            pick = {**pick, "category": cat, "reason": CATEGORIES[cat]}
            chosen.append(pick)
            used_meetings[pick["meeting"]] = used_meetings.get(pick["meeting"], 0) + 1
    for n, c in enumerate(chosen, 1):
        c["id"] = f"c{n:02d}"
    return chosen


# --------------------------------------------------------------------------- outputs
def cut_audio(meeting: str, start: float, end: float, dst: Path) -> bool:
    src = ROOT / "audio" / f"{meeting}.wav"
    if not src.exists():
        return False
    with wave.open(str(src), "rb") as w:
        rate = w.getframerate()
        w.setpos(max(0, int(start * rate)))
        frames = w.readframes(int((end - start) * rate))
        params = w.getparams()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as out:
        out.setparams(params)
        out.writeframes(frames)
    return True


def lines_for(utts: list[Utt], first: int, last: int, t0: float) -> list[str]:
    return [f"[{fmt(u.start - t0)}] {u.speaker}: {u.text}" for u in utts[first:last + 1]]


GOLD_HEADER = """# STATUS: unverified
# Clip {id} | {category} | meeting {meeting} | {start} - {end}
# HOW TO VERIFY (listen to validation/clips/{id}.wav):
#  1. Fix every wrong word so the text matches what is said (fillers like uh/um may stay or go, both are ignored).
#  2. Fix the speaker on every line (Speaker 1 / Speaker 2 - labels only need to be consistent inside this file).
#  3. Punctuation and capital letters do not matter.
#  4. When finished, change the first line to:  # STATUS: verified
# Lines starting with '#' are ignored. One line per speaker turn:  [mm:ss] Speaker N: text
"""


def main() -> None:
    glossary = load_glossary()
    meetings = {p.stem: p for p in sorted((ROOT / "transcripts").glob("*.txt")) if not p.name.startswith("ALL_")}
    if not meetings:
        raise SystemExit("No transcripts found in transcripts/")

    clips_file = OUT / "clips.json"
    if clips_file.exists():
        chosen = json.loads(clips_file.read_text(encoding="utf-8"))
        print(f"reusing {clips_file.name} ({len(chosen)} clips)")
    else:
        all_windows = []
        for name, path in meetings.items():
            all_windows += windows_for(name, parse_transcript(path),
                                       parse_changes(ROOT / "refined" / f"{name}.changes.log"), glossary)
        chosen = choose(all_windows)
        OUT.mkdir(exist_ok=True)
        clips_file.write_text(json.dumps(chosen, indent=2), encoding="utf-8")

    total = 0.0
    for c in chosen:
        raw = parse_transcript(meetings[c["meeting"]])
        refined_path = ROOT / "refined" / f"{c['meeting']}.txt"
        refined = parse_transcript(refined_path) if refined_path.exists() else []
        t0 = c["start"] + 0.3
        base = CLIPS_DIR / c["id"]
        ok = cut_audio(c["meeting"], c["start"], c["end"], base.with_suffix(".wav"))
        raw_lines = lines_for(raw, c["first"], c["last"], t0)
        base.with_suffix(".raw.txt").write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
        if len(refined) == len(raw):
            base.with_suffix(".refined.txt").write_text(
                "\n".join(lines_for(refined, c["first"], c["last"], t0)) + "\n", encoding="utf-8")
        gold = base.with_suffix(".gold.txt")
        if not gold.exists():
            head = GOLD_HEADER.format(id=c["id"], category=c["category"], meeting=c["meeting"],
                                      start=fmt(c["start"]), end=fmt(c["end"]))
            gold.write_text(head + "\n".join(raw_lines) + "\n", encoding="utf-8")
        total += c["end"] - c["start"]
        print(f"{c['id']}  {c['category']:<20} {c['meeting']}  {fmt(c['start'])}-{fmt(c['end'])}  "
              f"{c['words']:>4} words  switches={c['switches']:<3} short={c['short']:<3} "
              f"jargon={c['jargon']:<2} llm(text/spk/rej)={c['text_edits']}/{c['speaker_edits']}/{c['rejected']}"
              f"{'' if ok else '  (audio missing)'}")
    print(f"\n{len(chosen)} clips, {total / 60:.1f} minutes of audio -> {OUT}")


if __name__ == "__main__":
    main()
