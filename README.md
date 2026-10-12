# wisper: local push-to-talk dictation for Windows that learns how you speak

Hold **Ctrl+Shift**, talk, release: the text is typed where your cursor is (the clipboard is untouched). Esc or any
other key during the hold cancels, so Ctrl+Shift+T and other shortcuts still work. A small pill at the bottom of the
screen shows Listening / Transcribing, with a short soft beep when listening starts. Everything runs on your PC:
one speech model on the GPU and no AI cleanup model, so the result is fast and uses little memory.

```
mic (kept open while you're active, closes after 10 min idle; 0.4 s pre-roll so the first word is never cut)
  -> while you talk: every natural pause (0.3 s) after 5+ s of new speech commits that piece:
       Whisper large-v3 turbo (GPU, your vocabulary as a hint) -> noise guard -> code cleanup (~1 ms)
  -> at short pauses the remaining tail is transcribed ahead of time; if you release without saying more it's reused
  -> on release: only the last few seconds are left -> typed
```

**Cleanup is plain code, no model** (`service/cleanup.py`): hesitations (um, uh), stutters ("can you, can you"),
filler asides ("Like, where..." / ", you know,"), your known mishearings (`replacements.json`), exact spelling of your
names, CamelCase names said as words ("test flight" -> TestFlight, "open AI" -> OpenAI), a sound-alike fixer for your
names ("info tick" -> Infotik; ordinary English words are never touched), punctuation and capitals.

**It learns your words.** `learn/build_vocab.py` collects candidate terms from your project and skill folder names
(and a Graphify `graph.json` knowledge graph if you have one), then keeps the ones you have actually said
at least 3 times that aren't ordinary English. Every night `learn/learn.py` rebuilds that list, re-transcribes the
day's clips with the slower, more accurate Whisper large-v3, and turns mishearings that keep recurring into fixes.

## Speed
Release -> text, measured on real recordings with an RTX 5070 Laptop GPU:
- usually near-instant (the text was already transcribed during a pause before you let go);
- about 0.2-0.5 s when you release right on your last word (that last piece is transcribed after release).

Memory: one process, ~0.6 GB RAM and ~2.6 GB of GPU memory.

## Setup (Windows, NVIDIA GPU)
1. Python 3.12, then `pip install -r requirements.txt`.
2. CUDA libraries for faster-whisper (no CUDA toolkit install needed). Put `cublas64_12.dll`, `cublasLt64_12.dll`
   and `cudart64_12.dll` in `bin/cuda/`, for example from NVIDIA's wheels:
   ```
   pip download nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 --no-deps --only-binary=:all: -d dl
   ```
   then unzip the `.whl` files (they are zip archives) and copy those three DLLs. RTX 50-series GPUs need cuBLAS 12.8
   or newer: with 12.4 every pass was 2.3x slower.
3. The model downloads on first start (~1.6 GB). After that, set `HF_HUB_OFFLINE=1` or leave it: the service already
   runs offline once the model is cached.
4. `personal/*.example.json` are used until you create your own `personal/vocab.json` and
   `personal/replacements.json` (copy the examples and add your names). Put machine-specific settings, such as
   `vocab_folders` and `vocab_graph`, in `config.local.json`; it overrides `config.json` and is not committed.
5. `powershell -File tools\install_autostart.ps1` starts it hidden at login (Task Scheduler, priority 4) and
   registers the nightly learner. Or run it by hand: `pythonw -I service\wisper.py`.

## Files
| Path | What |
|---|---|
| `config.json` | port, model, thresholds (restart the service after editing); `config.local.json` overrides it |
| `personal/vocab.json` | `names` (always spelled exactly so) and `context` (hints). Hot-reloaded |
| `personal/replacements.json` | mishearing -> what you meant. Seeded + learned nightly; hot-reloaded |
| `personal/vocab_auto.json` | generated nightly: terms from your folders/graph that you actually say |
| `service/wisper.py` | the service (engine, HTTP API, logging) |
| `service/cleanup.py` | the code-only cleanup and the name fixer |
| `service/dictate.py` | native hotkey, mic, chunking, pill, beep, typing |
| `learn/learn.py` | nightly learner: rebuilds the vocabulary, then re-checks new clips with Whisper large-v3 (beam 5) |
| `learn/build_vocab.py` | the vocabulary miner (~2 s) |
| `data/wisper.db` | every dictation: raw and final text and timings (the learning data); clips in `data/audio` (90 days) |
| `lab/` | benchmarks and experiments (speed, accuracy, chunking, beep) |

## Running
- Health: `curl http://127.0.0.1:47821/health`
- Nightly learning: task `wisper-learn` at 03:30 (or when the PC is next on). Manual: `python -I learn\learn.py`.
- Teach it a fix right now: add `["what it hears", "What you meant", "manual"]` to `personal/replacements.json`.
- Any app that speaks the OpenAI transcription API can use `http://127.0.0.1:47821/v1/audio/transcriptions`.
- `"start_beep": false` in the config turns the beep off.

## Things that were tested and rejected
- An LLM cleanup pass (Qwen 3.5 4B): ~0.8 GB more memory and ~0.9 s per use, and it mostly changed punctuation.
- Shorter encoder windows: 2x faster passes, but 27-53% of words garbled. Whisper needs its full 30 s window.
- Trimming the vocabulary hint or skipping voice detection: under 25 ms per pass to gain.
- Smaller chunks (3 s), the previous chunk's words as context, reusing an unfinished early pass on release
  (`lab/chunk_test.py`, all switchable per session): no measurable gain over the defaults, and 3 s chunks alone
  changed more words.

## Known limits
- Windows only (keyboard hook, SendInput typing, Task Scheduler).
- The name fixer only knows names you have said at least 3 times or listed in `vocab.json`.
- Fillers are only removed where punctuation marks them as an aside, so "I feel like" and "you know how" stay.
