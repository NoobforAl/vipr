"""LLM post-processing of meeting transcripts (local, 4-bit, fits a 4 GB GPU).

Reads the raw transcripts written by transcribe_meetings.py, sends them to a local LLM in
small chunks and writes improved copies:
  * fixes words that were misheard but sound alike (using context + glossary.txt)
  * fixes punctuation / capitalisation, removes "uh"/"um"
  * fixes speaker labels that are clearly wrong (mostly short interjections like "okay")

Safety nets (small models love to over-edit):
  * every line is checked against the original; big rewrites are rejected
  * speaker flips are limited to short lines and capped per chunk
  * if the model breaks the output format the chunk is split and retried, then skipped
  * the raw transcript is never touched, and a .changes.log lists every edit

Usage:
    uv run refine_transcripts.py                 # transcripts/*.txt -> refined/*.txt
    uv run refine_transcripts.py transcripts/"2026-10-23 21-12-02.txt" --force
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
LINE_RE = re.compile(r"^\[(\d\d:\d\d:\d\d)\] (Speaker \d+): (.*)$")
OUT_RE = re.compile(r"^\s*(\d+)\s*\|\s*(Speaker \d+)\s*\|\s*(.*?)\s*$")
FILLERS = {"uh", "um", "uhm", "erm", "umm", "uhh"}
MAX_FLIP_WORDS = 12      # only short lines may change speaker
MAX_FLIP_FRACTION = 0.3  # if more than this share of a chunk flips, the model is guessing

SYSTEM_PROMPT = """You are a careful editor of automatic speech-recognition (ASR) transcripts of English \
meetings between two people (Speaker 1 and Speaker 2), usually about software development.

Your job, per line:
1. Fix words the ASR misheard when a similar-sounding word clearly fits the context better. \
Prefer spellings from the glossary. Only change a word when you are confident.
2. Add MISSING punctuation. Keep every comma, full stop and question mark that is already there unless \
it is plainly wrong - never strip punctuation from a line. Fix grammar only where it is clearly an ASR artefact.
3. Remove filler sounds such as "uh", "um". Keep everything else the speaker said, in their own words.
   Lines are pieces of continuous speech: a pause can split one sentence over several lines. Only capitalise \
the first word of a line if the previous line ended with . ? ! (or it is "I", a name, or a glossary term). \
Do not add a full stop to a line that is clearly an unfinished fragment.
4. SPEAKER LABELS: the diarizer sometimes gives a short line (e.g. "okay", "yeah", "right") to the wrong \
person. Change a label only if the flow of the conversation makes it clear (for example the same label \
would be answering its own question). When unsure, keep the label.

Never add information, never summarise, never merge or split lines, never translate, never answer \
questions in the text. If a line is already fine, return it unchanged.

Output format - exactly one line per input line, nothing else:
<id>|<Speaker N>|<corrected text>"""


# --------------------------------------------------------------------------- parsing
@dataclass
class Utt:
    time: str
    speaker: str
    text: str
    new_speaker: str = ""
    new_text: str = ""

    def __post_init__(self) -> None:
        self.new_speaker, self.new_text = self.speaker, self.text


def parse_transcript(text: str) -> tuple[list[str], list[Utt]]:
    header: list[str] = []
    utts: list[Utt] = []
    in_header = True
    for line in text.splitlines():
        if in_header:
            header.append(line)
            if line.startswith("=" * 10):
                in_header = False
            continue
        m = LINE_RE.match(line)
        if m:
            utts.append(Utt(m[1], m[2], m[3]))
    return header, utts


# --------------------------------------------------------------------------- guards
def norm_words(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in FILLERS]


def acceptable_edit(old: str, new: str) -> bool:
    """Reject rewrites: the corrected line must stay close to what was said."""
    wo, wn = norm_words(old), norm_words(new)
    if not wo:
        return False          # nothing but fillers: leave as is
    if not wn:
        return False          # model deleted the content
    ratio = difflib.SequenceMatcher(None, wo, wn).ratio()
    if not (ratio >= 0.6 and 0.6 <= len(wn) / len(wo) <= 1.5):
        return False
    # the model must not strip punctuation that was already there
    po, pn = punct_count(old), punct_count(new)
    return not (po >= 3 and pn < 0.6 * po)


def punct_count(s: str) -> int:
    return sum(c in ",.?!;:" for c in s)


def fix_continuation_case(prev: str, old: str, new: str, keep_caps: set[str]) -> str:
    """If this line continues an unfinished sentence, undo a capital the model added."""
    if not (prev and old and new):
        return new
    if prev.rstrip()[-1:] in ".?!" or not old[0].islower() or not new[0].isupper():
        return new
    first = re.sub(r"[^\w']", "", new.split()[0])
    if first == "I" or first.startswith("I'") or first.lower() in keep_caps:
        return new
    return new[0].lower() + new[1:]


# --------------------------------------------------------------------------- model
class Refiner:
    def __init__(self, model_id: str, glossary: list[str]) -> None:
        if not torch.cuda.is_available():
            sys.exit("Refining needs a CUDA GPU (4-bit quantisation). CUDA is not available.")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"[llm] loading {model_id} (4-bit)")
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quant, device_map={"": 0}, dtype=torch.float16
        )
        self.model.eval()
        self.glossary = glossary
        print(f"[llm] ready, VRAM used: {torch.cuda.memory_allocated() / 2**20:.0f} MB")

    @torch.inference_mode()
    def _chat(self, user: str, max_new: int) -> str:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        enc = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        out = self.model.generate(
            **enc, max_new_tokens=max_new, do_sample=False,
            temperature=None, top_p=None, top_k=None,
        )
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def _try(self, idx: list[int], utts: list[Utt], context: list[Utt]) -> dict[int, tuple[str, str]] | None:
        lines = [f"{i}|{utts[i].new_speaker}|{utts[i].new_text}" for i in idx]
        parts = []
        if self.glossary:
            parts.append("Glossary (preferred spellings): " + ", ".join(self.glossary))
        if context:
            parts.append("Previous lines (context only, do NOT output them):\n"
                         + "\n".join(f"{u.new_speaker}: {u.new_text}" for u in context))
        parts.append("Lines to fix:\n" + "\n".join(lines))
        expected_tokens = len(self.tok("\n".join(lines))["input_ids"])
        reply = self._chat("\n\n".join(parts), max_new=int(expected_tokens * 1.4) + 64)

        parsed: dict[int, tuple[str, str]] = {}
        for line in reply.splitlines():
            m = OUT_RE.match(line)
            if m:
                parsed[int(m[1])] = (m[2], m[3])
        if set(parsed) != set(idx):
            return None
        return parsed

    def refine_chunk(self, idx: list[int], utts: list[Utt], context: list[Utt]) -> dict[int, tuple[str, str]]:
        result = self._try(idx, utts, context)
        if result is not None:
            return result
        if len(idx) == 1:
            return {}                       # give up on this line, keep the original
        mid = len(idx) // 2                 # model broke the format -> split and retry
        left = self.refine_chunk(idx[:mid], utts, context)
        right = self.refine_chunk(idx[mid:], utts, context + [utts[i] for i in idx[:mid]][-4:])
        return {**left, **right}


# --------------------------------------------------------------------------- driver
def refine_utterances(refiner: Refiner, utts: list[Utt], chunk_lines: int, log: list[str]) -> dict[str, int]:
    labels = {u.speaker for u in utts}
    stats = {"text": 0, "speaker": 0, "rejected": 0, "skipped": 0}
    starts = range(0, len(utts), chunk_lines)
    for start in tqdm(starts, desc="  refining", unit="chunk", leave=False):
        idx = list(range(start, min(start + chunk_lines, len(utts))))
        context = utts[max(0, start - 4):start]
        result = refiner.refine_chunk(idx, utts, context)
        stats["skipped"] += len(idx) - len(result)

        flips = [i for i, (spk, _) in result.items() if spk in labels and spk != utts[i].speaker]
        allow_flips = len(flips) <= max(1, int(MAX_FLIP_FRACTION * len(idx)))

        keep_caps = {w.lower() for term in refiner.glossary for w in term.split()}
        for i in sorted(result):
            spk, text = result[i]
            u = utts[i]
            if i > 0:
                text = fix_continuation_case(utts[i - 1].new_text, u.text, text, keep_caps)
            if text != u.text:
                if acceptable_edit(u.text, text):
                    u.new_text = text
                    stats["text"] += 1
                    log.append(f"[{u.time}] #{i} TEXT\n   - {u.text}\n   + {text}")
                else:
                    stats["rejected"] += 1
                    log.append(f"[{u.time}] #{i} REJECTED rewrite\n   - {u.text}\n   ? {text}")
            if spk != u.speaker and spk in labels:
                if allow_flips and len(u.text.split()) <= MAX_FLIP_WORDS:
                    u.new_speaker = spk
                    stats["speaker"] += 1
                    log.append(f"[{u.time}] #{i} SPEAKER {u.speaker} -> {spk}: {u.new_text}")
                else:
                    stats["rejected"] += 1
    return stats


def write_refined(path: Path, header: list[str], utts: list[Utt], model_id: str) -> None:
    head = header[:-1] + [f"Refined : {model_id.split('/')[-1]}", header[-1]]
    lines = [*head, ""]
    for u in utts:
        lines += [f"[{u.time}] {u.new_speaker}: {u.new_text}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def refine_paths(paths: list[Path], out_dir: Path, glossary: list[str], model_id: str = MODEL_ID,
                 force: bool = False, chunk_lines: int = 20) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in paths if force or not (out_dir / p.name).exists()]
    outputs = [out_dir / p.name for p in paths]
    if todo:
        refiner = Refiner(model_id, glossary)
        for n, src in enumerate(todo, 1):
            print(f"[refine {n}/{len(todo)}] {src.name}")
            header, utts = parse_transcript(src.read_text(encoding="utf-8"))
            dst = out_dir / src.name
            if not utts:
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                continue
            t0, log = time.time(), []
            stats = refine_utterances(refiner, utts, chunk_lines, log)
            write_refined(dst, header, utts, model_id)
            dst.with_suffix(".changes.log").write_text("\n\n".join(log), encoding="utf-8")
            print(f"  saved {dst}  ({(time.time() - t0) / 60:.1f} min) "
                  f"text edits={stats['text']} speaker fixes={stats['speaker']} "
                  f"rejected={stats['rejected']} skipped={stats['skipped']}")
    else:
        print("Nothing to refine (use --force to redo).")

    done = [o for o in outputs if o.exists()]
    if done:
        combined = out_dir / "ALL_MEETINGS_refined.txt"
        combined.write_text(("\n\n" + "#" * 70 + "\n\n").join(
            o.read_text(encoding="utf-8") for o in sorted(done)), encoding="utf-8")
        print(f"Combined file: {combined}")
    return done


def main() -> None:
    from transcribe_meetings import ROOT, load_glossary  # also loads .env

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", default=[str(ROOT / "transcripts")], help="transcript .txt files or folders")
    ap.add_argument("-o", "--output-dir", default=str(ROOT / "refined"))
    ap.add_argument("--model", default=MODEL_ID, help=f"HF causal LM (default: {MODEL_ID})")
    ap.add_argument("--glossary", default=str(ROOT / "glossary.txt"))
    ap.add_argument("--chunk-lines", type=int, default=20, help="lines per LLM call (default 20)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    paths: list[Path] = []
    for p in map(Path, args.inputs):
        if p.is_dir():
            paths += [f for f in sorted(p.glob("*.txt")) if not f.name.startswith("ALL_MEETINGS")]
        elif p.is_file():
            paths.append(p)
        else:
            sys.exit(f"Not found: {p}")
    if not paths:
        sys.exit("No transcripts found. Run transcribe_meetings.py first.")
    refine_paths(paths, Path(args.output_dir), load_glossary(Path(args.glossary)),
                 args.model, args.force, args.chunk_lines)


if __name__ == "__main__":
    main()
