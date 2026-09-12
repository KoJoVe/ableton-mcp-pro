"""
MIDI Gen AI Bridge: Ableton MCP clip notes -> AI MIDI generation -> notes back.

Uses the [midigenai](https://github.com/nicholasbien/midigenai) package directly;
the model is downloaded from HuggingFace on first use and cached locally.

Reads JSON config from stdin:
  {
    "notes": [{"pitch": 62, "start_time": 0.0, "duration": 0.5, "velocity": 100}, ...],
    "tempo_bpm": 80.0,
    "max_new_tokens": 256, "temperature": 1.2, "top_k": 50,
    "prompt_end_beat": 32.0,           # filter output to notes after this beat
    "pitch_range": [60, 96],           # optional pitch filter
    "version": "v2-pilot",             # pinned model subfolder
    "revision": "71347f...",           # pinned immutable HF revision
    "repo_id": "nicholasbien/midigenai" # pinned model repository
  }

Writes JSON to stdout:
  {"prompt_tokens": int, "generated_tokens": int, "tempo_bpm": float,
   "notes": [{"pitch", "start_time", "duration", "velocity"}, ...]}

Switching to a new model release requires a security review, an immutable
revision, and updated SHA-256 constants in this file.

Run with whatever Python env has the `[ai]` extras installed:
  python -m pip install -r pylock.ai.toml
  python tools/midigenai_bridge.py < cfg.json
"""

import json
import hashlib
import os
import sys
import tempfile
from pathlib import Path

TPQ = 480

PINNED_MODEL_REPO = "nicholasbien/midigenai"
PINNED_MODEL_VERSION = "v2-pilot"
PINNED_MODEL_REVISION = "71347f047227c835002a0b7e05c47b08af0b0984"
PINNED_CHECKPOINT_SHA256 = "e697c407f40bf58a7cc8f7d6a417be9819d3bdbd33c2104f9ac43e509d5f86d5"
PINNED_TOKENIZER_SHA256 = "13744f902aded08d3ec03378565a9b8e20c56fbf9eb45a6c067c77df9de3bd03"


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# Lazy global so repeated calls in the same process don't reload the model
_GENERATOR = None


def _get_generator(repo_id, version, revision):
    """Download a pinned model, verify it, and load it with restricted unpickling."""
    global _GENERATOR
    requested = (
        repo_id or PINNED_MODEL_REPO,
        version or PINNED_MODEL_VERSION,
        revision or PINNED_MODEL_REVISION,
    )
    pinned = (PINNED_MODEL_REPO, PINNED_MODEL_VERSION, PINNED_MODEL_REVISION)
    custom_source = requested != pinned
    if custom_source and not _env_flag("ABLETON_MCP_ALLOW_CUSTOM_MODEL_SOURCE"):
        raise ValueError(
            "Custom model sources are disabled. Set "
            "ABLETON_MCP_ALLOW_CUSTOM_MODEL_SOURCE=1 only after auditing the source."
        )

    cache_key = requested
    if _GENERATOR is None or _GENERATOR[0] != cache_key:
        from midigenai import V2Generator, download_v2_files
        checkpoint, tokenizer = download_v2_files(
            repo_id=requested[0], version=requested[1], revision=requested[2]
        )
        if not custom_source:
            actual_checkpoint_hash = _sha256(checkpoint)
            actual_tokenizer_hash = _sha256(tokenizer)
            if actual_checkpoint_hash != PINNED_CHECKPOINT_SHA256:
                raise RuntimeError("Pinned model checkpoint SHA-256 mismatch")
            if actual_tokenizer_hash != PINNED_TOKENIZER_SHA256:
                raise RuntimeError("Pinned tokenizer SHA-256 mismatch")

        # The dependency currently asks torch.load for unrestricted pickle loading.
        # Override that one call so downloaded checkpoints can only use the restricted
        # weights-only unpickler, then immediately restore torch.load.
        import torch
        original_torch_load = torch.load

        def restricted_torch_load(*args, **kwargs):
            kwargs["weights_only"] = True
            return original_torch_load(*args, **kwargs)

        try:
            torch.load = restricted_torch_load
            gen = V2Generator(checkpoint_path=checkpoint, tokenizer_path=tokenizer)
        finally:
            torch.load = original_torch_load
        _GENERATOR = (cache_key, gen)
    return _GENERATOR[1]


def notes_to_score(notes, tempo_bpm):
    from symusic import Score, Track, Note, Tempo
    score = Score(TPQ)
    score.tempos = [Tempo(time=0, qpm=tempo_bpm)]

    by_program = {}
    for n in notes:
        prog = int(n.get("program", 0))
        by_program.setdefault(prog, []).append(n)

    for prog, ns in by_program.items():
        track = Track(program=prog, is_drum=False, name=f"prog{prog}")
        for n in ns:
            track.notes.append(Note(
                time=int(round(float(n["start_time"]) * TPQ)),
                duration=max(1, int(round(float(n["duration"]) * TPQ))),
                pitch=int(n["pitch"]),
                velocity=int(n["velocity"]),
            ))
        score.tracks.append(track)
    return score


def score_to_notes(score, prompt_end_beat=None, pitch_range=None):
    notes = []
    tpq = score.ticks_per_quarter
    for track in score.tracks:
        for n in track.notes:
            start_beat = float(n.start) / tpq
            dur_beat = float(n.duration) / tpq
            pitch = int(n.pitch)
            if prompt_end_beat is not None and start_beat < prompt_end_beat - 1e-6:
                continue
            if pitch_range is not None and not (pitch_range[0] <= pitch <= pitch_range[1]):
                continue
            notes.append({
                "pitch": pitch,
                "start_time": round(start_beat - (prompt_end_beat or 0), 4),
                "duration": round(dur_beat, 4),
                "velocity": int(n.velocity),
                "mute": False,
            })
    notes.sort(key=lambda x: (x["start_time"], x["pitch"]))
    return notes


def main():
    cfg = json.loads(sys.stdin.read())

    notes = cfg["notes"]
    tempo = float(cfg.get("tempo_bpm", 120.0))
    max_new_tokens = int(cfg.get("max_new_tokens", 256))
    temperature = float(cfg.get("temperature", 1.2))
    top_k = int(cfg.get("top_k", 50))
    prompt_end_beat = cfg.get("prompt_end_beat")
    if prompt_end_beat is not None:
        prompt_end_beat = float(prompt_end_beat)
    pitch_range = cfg.get("pitch_range")
    # Defaults resolve to audited, immutable constants above. Caller-selected model
    # sources are rejected unless ABLETON_MCP_ALLOW_CUSTOM_MODEL_SOURCE=1.
    version = cfg.get("version")
    repo_id = cfg.get("repo_id")
    revision = cfg.get("revision")

    score = notes_to_score(notes, tempo)
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        in_path = f.name
    score.dump_midi(in_path)

    gen = _get_generator(repo_id, version, revision)
    prompt_ids = gen.encode_midi_file(in_path)

    out_path = in_path.replace(".mid", "_out.mid")
    new_ids = gen.generate_to_midi(
        prompt_ids, out_path,
        tempo_bpm=tempo,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    from symusic import Score
    out_score = Score(out_path)
    out_notes = score_to_notes(out_score, prompt_end_beat=prompt_end_beat, pitch_range=pitch_range)

    Path(in_path).unlink(missing_ok=True)
    Path(out_path).unlink(missing_ok=True)

    json.dump({
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(new_ids),
        "tempo_bpm": tempo,
        "notes": out_notes,
    }, sys.stdout)


if __name__ == "__main__":
    main()
