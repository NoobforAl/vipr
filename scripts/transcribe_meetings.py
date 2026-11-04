"""Meeting transcriber: video -> audio -> speaker diarization + speech-to-text -> .txt

Pipeline (all local, CUDA by default):
  1. ffmpeg (bundled via imageio-ffmpeg) extracts 16 kHz mono WAV from every video.
  2. faster-whisper (large-v3-turbo) transcribes with word timestamps.   [GPU]
  3. pyannote speaker-diarization-community-1 finds who speaks when.      [GPU]
  4. Words are assigned to speakers and merged into a conversation.

Whisper and the diarizer run one after the other (never at the same time) so the
whole thing fits in a 4 GB GPU.

Usage:
    uv run transcribe_meetings.py                       # every video in this folder
    uv run transcribe_meetings.py path/to/folder_or_file.mp4 --language en
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# torch must be imported BEFORE faster_whisper/ctranslate2 on Windows: it registers the
# CUDA/cuDNN DLL directory that ctranslate2 needs.
import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from imageio_ffmpeg import get_ffmpeg_exe
from tqdm import tqdm

VIDEO_EXTS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".m4a",
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".ts",
}
SAMPLE_RATE = 16_000
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


# --------------------------------------------------------------------------- helpers
def load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines); real environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip("\"'"))


HERE = Path(__file__).resolve().parent
# The scripts live in <project>/scripts; the data (videos, transcripts/, refined/, audio/,
# glossary.txt, .env) lives one level up. Works from either location.
ROOT = next((p for p in (HERE, HERE.parent) if (p / "glossary.txt").exists()), HERE)
load_dotenv(ROOT / ".env")
load_dotenv(HERE / ".env")


@dataclass
class Word:
    start: float
    end: float
    text: str
    speaker: str | None = None


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def load_glossary(path: Path, max_terms: int = 120) -> list[str]:
    """Read terms from a text file; '#' starts a comment. Whisper's prompt window is small
    (~224 tokens), so the list is capped."""
    if not path.is_file():
        return []
    terms: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0]
        terms += [t.strip() for t in line.split(",") if t.strip()]
    terms = list(dict.fromkeys(terms))  # de-duplicate, keep order
    if len(terms) > max_terms:
        print(f"WARNING: glossary has {len(terms)} terms, using the first {max_terms}")
    return terms[:max_terms]


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        isinstance(exc, torch.cuda.OutOfMemoryError)
        or "out of memory" in msg
        or "cuda failed" in msg
    )


# --------------------------------------------------------------------------- 1. audio
def extract_audio(video: Path, wav: Path) -> None:
    """Extract 16 kHz mono 16-bit WAV (what both Whisper and pyannote want)."""
    if wav.exists() and wav.stat().st_size > 0:
        return
    wav.parent.mkdir(parents=True, exist_ok=True)
    tmp = wav.with_suffix(".part.wav")
    cmd = [
        get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(wav)


# --------------------------------------------------------------------------- 2. speech-to-text
def transcribe(wav: Path, args: argparse.Namespace) -> tuple[list[Word], str]:
    device = args.device
    compute_type = args.compute_type
    if device == "cpu" and compute_type == "int8_float16":
        compute_type = "int8"

    print(f"  [whisper] loading {args.model} on {device} ({compute_type})")
    model = WhisperModel(args.model, device=device, compute_type=compute_type)
    try:
        # Pass a float32 array (not a path) so faster-whisper never touches PyAV.
        audio, _ = sf.read(str(wav), dtype="float32")
        glossary_kwargs = {}
        if args.glossary_terms:
            terms = ", ".join(args.glossary_terms)
            # hotwords bias every 30 s window; initial_prompt primes the first one.
            glossary_kwargs = {
                "hotwords": terms,
                "initial_prompt": f"Glossary: {terms}.",
            }
            print(f"  [whisper] glossary: {len(args.glossary_terms)} terms")
        segments, info = model.transcribe(
            audio,
            **glossary_kwargs,
            language=args.language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,  # skips silence, avoids hallucinations
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,  # avoids repetition loops on long audio
        )
        print(
            f"  [whisper] language={info.language} (p={info.language_probability:.2f}), "
            f"duration={fmt_time(info.duration)}"
        )
        words: list[Word] = []
        with tqdm(
            total=round(info.duration), unit="s", desc="  transcribing", leave=False
        ) as bar:
            for seg in segments:
                for w in seg.words or []:
                    words.append(Word(w.start, w.end, w.word))
                bar.n = min(round(seg.end), bar.total)
                bar.refresh()
        return words, info.language
    finally:
        del model
        free_cuda()


# --------------------------------------------------------------------------- 3. diarization
def check_diarization_access(token: str | None) -> None:
    """Make sure the gated pyannote model is reachable before spending time on Whisper."""
    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(DIARIZATION_MODEL, "config.yaml", token=token)
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            f"Cannot access {DIARIZATION_MODEL} ({type(exc).__name__}).\n"
            f"  1. Accept the model terms at https://huggingface.co/{DIARIZATION_MODEL}\n"
            "  2. Create a read token at https://huggingface.co/settings/tokens\n"
            "  3. Set it:  $env:HF_TOKEN = 'hf_...'   (or pass --hf-token)"
        )


def load_diarization_pipeline(token: str | None):
    from pyannote.audio import Pipeline  # imported lazily: it is slow to import

    return Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)


def diarize(
    wav: Path, args: argparse.Namespace, token: str | None
) -> list[tuple[float, float, str]]:
    audio, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    # Pre-loaded waveform => pyannote never needs torchcodec/ffmpeg shared libs.
    inputs = {"waveform": torch.from_numpy(audio.T.copy()), "sample_rate": sr}
    del audio

    kwargs = {}
    if args.num_speakers:
        kwargs["num_speakers"] = args.num_speakers
    if args.min_speakers:
        kwargs["min_speakers"] = args.min_speakers
    if args.max_speakers:
        kwargs["max_speakers"] = args.max_speakers

    pipeline = load_diarization_pipeline(token)
    if pipeline is None:
        sys.exit(
            f"Could not load {DIARIZATION_MODEL}.\n"
            "  1. Accept the model terms at https://huggingface.co/"
            + DIARIZATION_MODEL
            + "\n"
            "  2. Create a read token at https://huggingface.co/settings/tokens\n"
            "  3. Set it:  $env:HF_TOKEN = 'hf_...'   (or pass --hf-token)"
        )

    devices = [args.diar_device] + (["cpu"] if args.diar_device != "cpu" else [])
    for dev in devices:
        try:
            print(f"  [pyannote] diarizing on {dev}")
            pipeline.to(torch.device(dev))
            out = pipeline(inputs, **kwargs)
            break
        except Exception as exc:
            if dev != "cpu" and is_oom(exc):
                print("  [pyannote] GPU out of memory -> retrying on CPU (slower)")
                free_cuda()
                continue
            raise
    del pipeline
    free_cuda()

    # pyannote 4.x: `exclusive_speaker_diarization` has no overlapping turns, which is
    # exactly what you want when aligning with ASR. Older versions return an Annotation.
    annotation = getattr(out, "exclusive_speaker_diarization", None) or getattr(
        out, "speaker_diarization", out
    )
    return [
        (t.start, t.end, spk) for t, _, spk in annotation.itertracks(yield_label=True)
    ]


# --------------------------------------------------------------------------- 4. merge
def assign_speakers(words: list[Word], turns: list[tuple[float, float, str]]) -> None:
    """Give each word the speaker whose turn contains (or is nearest to) its midpoint."""
    if not turns:
        for w in words:
            w.speaker = "SPEAKER_00"
        return
    turns = sorted(turns)
    starts = np.array([t[0] for t in turns])
    ends = np.array([t[1] for t in turns])
    for w in words:
        mid = (w.start + w.end) / 2
        inside = np.where((starts <= mid) & (mid <= ends))[0]
        if len(inside):
            idx = inside[0]
        else:  # word fell in a gap -> nearest turn
            idx = int(np.argmin(np.minimum(np.abs(starts - mid), np.abs(ends - mid))))
        w.speaker = turns[idx][2]


def build_conversation(
    words: list[Word], max_gap: float = 2.0
) -> list[tuple[float, float, str, str]]:
    """Group consecutive words of one speaker into utterances."""
    convo: list[list] = []  # [start, end, speaker, text]
    for w in words:
        if convo and convo[-1][2] == w.speaker and w.start - convo[-1][1] <= max_gap:
            convo[-1][1] = w.end
            convo[-1][3] += w.text
        else:
            convo.append([w.start, w.end, w.speaker, w.text])
    return [(s, e, spk, txt.strip()) for s, e, spk, txt in convo if txt.strip()]


def write_transcript(path: Path, video: Path, language: str, convo) -> None:
    # Friendly names in order of first appearance: Speaker 1, Speaker 2, ...
    names: dict[str, str] = {}
    for _, _, spk, _ in convo:
        names.setdefault(spk, f"Speaker {len(names) + 1}")

    duration = convo[-1][1] if convo else 0
    lines = [
        f"Meeting : {video.stem}",
        f"Source  : {video.name}",
        f"Language: {language}",
        f"Length  : {fmt_time(duration)}",
        f"Speakers: {len(names)}",
        "=" * 70,
        "",
    ]
    for start, _, spk, text in convo:
        lines.append(f"[{fmt_time(start)}] {names[spk]}: {text}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- main
def find_inputs(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for p in map(Path, paths):
        if p.is_file():
            found.append(p)
        elif p.is_dir():
            found += [f for f in sorted(p.iterdir()) if f.suffix.lower() in VIDEO_EXTS]
        else:
            sys.exit(f"Not found: {p}")
    return found


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "inputs",
        nargs="*",
        default=[str(ROOT)],
        help="video files and/or folders (default: the project folder)",
    )
    ap.add_argument(
        "-o",
        "--output-dir",
        default=str(ROOT / "transcripts"),
        help="where .txt files go (default: <project>/transcripts)",
    )
    ap.add_argument(
        "--audio-dir",
        default=str(ROOT / "audio"),
        help="where extracted WAVs are cached (default: <project>/audio)",
    )
    ap.add_argument(
        "--model",
        default="large-v3-turbo",
        help="faster-whisper model (e.g. large-v3-turbo, large-v3, medium, small)",
    )
    ap.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="device for Whisper (default: cuda)",
    )
    ap.add_argument(
        "--diar-device",
        default=None,
        choices=["cuda", "cpu"],
        help="device for diarization (default: same as --device)",
    )
    ap.add_argument(
        "--compute-type",
        default="int8_float16",
        help="int8_float16 (4 GB GPUs) | float16 | int8",
    )
    ap.add_argument(
        "--language",
        default="en",
        help="language code, e.g. en, ar, fr; 'auto' to detect (default: en)",
    )
    ap.add_argument(
        "--glossary",
        default=str(ROOT / "glossary.txt"),
        help="text file of names/jargon (one per line or comma-separated) that Whisper should "
        "prefer when a word sounds ambiguous (default: glossary.txt if it exists)",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=2,
        help="exact number of speakers (default: 2; 0 = detect)",
    )
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        help="Hugging Face token (or set HF_TOKEN)",
    )
    ap.add_argument(
        "--no-refine",
        action="store_true",
        help="skip the LLM step that fixes misheard words / speaker labels (see refine_transcripts.py)",
    )
    ap.add_argument(
        "--refined-dir",
        default=str(ROOT / "refined"),
        help="where the LLM-improved transcripts go (default: <project>/refined)",
    )
    ap.add_argument(
        "--llm",
        default=None,
        help="override the refine LLM (default: Qwen/Qwen3-4B-Instruct-2507)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-process meetings that already have a transcript",
    )
    ap.add_argument(
        "--keep-audio", action="store_true", default=True, help=argparse.SUPPRESS
    )
    ap.add_argument(
        "--delete-audio",
        action="store_true",
        help="delete the extracted WAV after each meeting",
    )
    args = ap.parse_args()

    if args.language == "auto":
        args.language = None
    args.glossary_terms = load_glossary(Path(args.glossary))
    check_diarization_access(args.hf_token)  # fail fast, before any slow work

    args.diar_device = args.diar_device or args.device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU (this will be slow).")
        args.device = args.diar_device = "cpu"

    # Never treat our own output folders as inputs
    out_dir, audio_dir = Path(args.output_dir), Path(args.audio_dir)
    videos = [
        v
        for v in find_inputs(args.inputs)
        if audio_dir.resolve() not in v.resolve().parents
    ]
    if not videos:
        sys.exit("No video/audio files found.")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{len(videos)} meeting(s) to process\n")

    done: list[Path] = []
    for i, video in enumerate(videos, 1):
        txt = out_dir / f"{video.stem}.txt"
        print(f"[{i}/{len(videos)}] {video.name}")
        if txt.exists() and not args.force:
            print("  already done, skipping (use --force to redo)\n")
            done.append(txt)
            continue
        t0 = time.time()
        wav = audio_dir / f"{video.stem}.wav"
        try:
            print("  [audio] extracting...")
            extract_audio(video, wav)
            words, language = transcribe(wav, args)
            if not words:
                print("  no speech detected")
                txt.write_text(
                    f"Meeting: {video.stem}\n\n(no speech detected)\n", encoding="utf-8"
                )
                continue
            turns = diarize(wav, args, args.hf_token)
            assign_speakers(words, turns)
            write_transcript(txt, video, language, build_conversation(words))
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
            print(f"  FAILED: {type(exc).__name__}: {exc}\n")
            continue
        if args.delete_audio:
            wav.unlink(missing_ok=True)
        done.append(txt)
        print(f"  saved {txt}  ({(time.time() - t0) / 60:.1f} min)\n")

    # One combined file with every meeting, in chronological (filename) order
    if done:
        combined = out_dir / "ALL_MEETINGS.txt"
        parts = [p.read_text(encoding="utf-8") for p in sorted(done)]
        combined.write_text(("\n\n" + "#" * 70 + "\n\n").join(parts), encoding="utf-8")
        print(f"Combined file: {combined}")

    # LLM pass: runs last, after Whisper and the diarizer have released the GPU
    if done and not args.no_refine:
        free_cuda()
        import refine_transcripts

        refine_transcripts.refine_paths(
            sorted(done),
            Path(args.refined_dir),
            args.glossary_terms,
            args.llm or refine_transcripts.MODEL_ID,
            args.force,
        )


if __name__ == "__main__":
    main()
