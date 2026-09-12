# tools/

Helper scripts that compose with the Ableton MCP server but run as their own
processes. The agent (or you) drives them via shell, no MCP server changes.

## midigenai_bridge.py — AI MIDI continuation

CLI that turns Ableton clip notes into an AI-generated continuation. Wraps
the [`midigenai`](https://github.com/nicholasbien/midigenai) package
(currently a 25M custom transformer, event-based tokenization). The model
is auto-downloaded from
[huggingface.co/nicholasbien/midigenai](https://huggingface.co/nicholasbien/midigenai)
on first use and cached at `~/.cache/huggingface/`.

### Model security

This fork pins the default model to an immutable Hugging Face revision and
verifies SHA-256 hashes for both the checkpoint and tokenizer before loading.
It forces PyTorch's restricted `weights_only=True` unpickler even though the
upstream dependency currently requests unrestricted loading.

Caller-selected repositories, versions, or revisions are rejected by default.
Do not set `ABLETON_MCP_ALLOW_CUSTOM_MODEL_SOURCE=1` unless you have audited the
source; restricted loading reduces code-execution risk but does not make an
arbitrary checkpoint harmless against denial-of-service or parser flaws.

### Setup

Use the platform-specific `pylock.ai.toml` with a current pip release, or
regenerate that lockfile for your Python/platform combination before install.

That installs `midigenai`, `torch`, `miditok`, `symusic`, and
`huggingface_hub`. No separate repo clone, no extra venv.

### What it does

```
JSON notes (stdin)
    ↓
build a temporary .mid file
    ↓
encode with the v2 MidiTok tokenizer
    ↓
sample N tokens from the v2 transformer (downloaded from HF on first call)
    ↓
decode back to MIDI, drop everything ≤ prompt_end_beat
    ↓
JSON notes (stdout)
```

### Usage

```bash
echo '{
  "notes": [
    {"pitch": 74, "start_time": 0,  "duration": 1, "velocity": 95},
    {"pitch": 78, "start_time": 1,  "duration": 1, "velocity": 95},
    {"pitch": 81, "start_time": 2,  "duration": 2, "velocity": 100}
  ],
  "tempo_bpm": 80,
  "max_new_tokens": 256,
  "temperature": 1.2,
  "top_k": 50,
  "prompt_end_beat": 4.0,
  "pitch_range": [60, 92]
}' | python tools/midigenai_bridge.py
```

Returns:

```json
{
  "prompt_tokens": 33,
  "generated_tokens": 256,
  "tempo_bpm": 80,
  "notes": [{"pitch": 78, "start_time": 0.0, "duration": 0.5, "velocity": 95, "mute": false}, ...]
}
```

Note start times are returned **relative to `prompt_end_beat`** so the result
drops cleanly into a fresh clip starting at beat 0.

### Knobs

| key | meaning |
|---|---|
| `notes` | seed notes — Ableton MCP format (`pitch`/`start_time`/`duration`/`velocity`) |
| `tempo_bpm` | tempo of seed *and* output (the model is tempo-agnostic; this is just decoding) |
| `max_new_tokens` | how much continuation to sample. ~4 tokens/note → 256 ≈ 60 notes |
| `temperature` | 0.5–1.5. Lower sticks closer to the prompt; higher diverges more |
| `top_k` | nucleus sampling K |
| `prompt_end_beat` | drop output notes that start before this beat (= filter out the prompt itself) |
| `pitch_range` | optional `[min, max]` MIDI pitch filter — useful when you only want, say, the lead range |
| `version` | Pinned model subfolder. Overrides are disabled by default. |
| `revision` | Pinned immutable Hugging Face commit. Overrides are disabled by default. |
| `repo_id` | Pinned Hugging Face repository. Overrides are disabled by default. |

### Switching to a new model release

Treat every model update like a code update. Review the source, choose an
immutable Hugging Face commit, calculate SHA-256 hashes for the checkpoint and
tokenizer, update the pinned constants in `midigenai_bridge.py`, regenerate the
AI lockfile, and rerun the security tests. Do not use a floating branch or tag.

### Wiring it into the Ableton workflow

The agent typically:

1. `mcp__AbletonMCP__get_clip_notes` → pull a clip's notes as the seed
2. Build a JSON payload with those notes + tempo + knobs
3. Run `tools/midigenai_bridge.py` via `Bash`, capture stdout
4. (optional) post-filter: snap to scale, clip durations, drop anything outside the song length
5. `mcp__AbletonMCP__create_clip` + `mcp__AbletonMCP__add_notes_to_clip` on a target track to drop the result

We deliberately did **not** add MCP tool wrappers around this. The bridge is a
plain CLI; the agent drives it via shell. Reasons:

- No MCP server restart needed when the bridge changes
- The bridge can be used standalone outside the agent (`echo … | python …`)
- The MCP server stays lean — `[ai]` is opt-in, not required

### Prompt design

v2 was trained on single-track polyphonic MIDI (heavy on piano via Lakh +
MAESTRO + POP909 + GiantMIDI). Best results come from:

- A monophonic or lightly-polyphonic **melodic seed** (4–8 bars, ~10–15 notes)
- Single instrument (omit `program` from notes, or all on the same program)
- Avoid feeding several stacked tracks (pad chords + bass + lead) — the model
  is happiest continuing a coherent musical phrase, not multi-part arrangements

### Performance

First call ≈ 2–4s on CPU (download + model load + ~250 tokens). Subsequent
calls in the same Python process reuse the loaded generator, so they run at
~70 notes/s. The bridge keeps the generator alive in a module-level cache.
