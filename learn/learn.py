"""Nightly self-improvement for wisper.

1. Rebuild the auto vocabulary from your Graphify graph + everything you have said (build_vocab.py).
2. For each new dictation (wisper.db) with a saved clip, re-transcribe it with a slow, accurate
teacher (Whisper large-v3, beam 5) and compare with what the fast live model heard. Word-level differences
that keep recurring become learned mishearing fixes (personal/replacements.json); names the teacher keeps
hearing become vocabulary hints (personal/vocab.json "context"). The service hot-reloads both files.

usage: python learn.py
"""
import datetime, difflib, glob, json, os, re, sqlite3, sys, time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = {**json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8")),  # + your own overrides:
       **(json.load(open(os.path.join(ROOT, "config.local.json"), encoding="utf-8"))
          if os.path.exists(os.path.join(ROOT, "config.local.json")) else {})}
PERSONAL = os.path.join(ROOT, "personal")
CUDA_DIR = os.path.join(ROOT, "bin", "cuda")
os.add_dll_directory(CUDA_DIR); os.environ["PATH"] = CUDA_DIR + os.pathsep + os.environ["PATH"]
sys.path.insert(0, os.path.join(ROOT, "service")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MIN_COUNT, MIN_DAYS = 3, 2       # a pair must recur this often before it is trusted
TEACHER = "large-v3"


def log(*a):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(os.path.join(ROOT, "logs", "learn.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def db():
    con = sqlite3.connect(os.path.join(ROOT, CFG["log_db"]))
    con.execute("""create table if not exists teacher (dictation_id integer primary key, audio_file text,
                   teacher_text text, ts text default (datetime('now','localtime')))""")
    con.execute("""create table if not exists pairs (heard text, meant text, count int, days text,
                   status text default 'candidate', primary key (heard, meant))""")
    return con


def pending(con):
    """Dictations not yet checked by the teacher that still have their saved clip."""
    rows = con.execute("""select d.id, d.asr_text, d.audio_file
                          from dictations d left join teacher t on t.dictation_id = d.id
                          where t.dictation_id is null and d.asr_text != '' and coalesce(d.source,'live') != 'test'
                          and d.audio_file is not null""").fetchall()
    return [(i, asr, f) for i, asr, f in rows if os.path.exists(f)]


def prune_audio(days=90):
    cutoff = time.time() - days * 86400
    for f in glob.glob(os.path.join(ROOT, "data", "audio", "*", "*.ogg")):
        if os.path.getmtime(f) < cutoff:
            os.remove(f)


def teacher_model():
    from faster_whisper import WhisperModel
    try:  # GPU only if there is room next to the live service, otherwise CPU (slower, fine at night)
        import subprocess
        free = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.split()[0])
    except Exception:
        free = 0
    if free > 4200:
        log(f"teacher on GPU ({free} MiB free)")
        return WhisperModel(TEACHER, device="cuda", compute_type="float16")
    log(f"teacher on CPU ({free} MiB GPU free)")
    return WhisperModel(TEACHER, device="cpu", compute_type="int8", cpu_threads=min(4, max(2, (os.cpu_count() or 4) // 4)))


def tokens(t):
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'.+-]*[A-Za-z0-9]|[A-Za-z0-9]", t)


def namey(word, position_ok):
    return (any(c.isupper() for c in word[1:]) or any(c.isdigit() for c in word)
            or (word[0].isupper() and position_ok))


def diff_pairs(live, teacher):
    """Yield (heard, meant) phrase pairs where the teacher's version contains a name-like word."""
    a, b = tokens(live), tokens(teacher)
    sm = difflib.SequenceMatcher(None, [w.lower() for w in a], [w.lower() for w in b], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "replace" or not (1 <= i2 - i1 <= 3 and 1 <= j2 - j1 <= 3):
            continue
        meant = b[j1:j2]
        if any(namey(w, j1 + k > 0) for k, w in enumerate(meant)):
            heard = " ".join(a[i1:i2]).lower()
            if heard.replace(" ", "") != "".join(meant).lower():
                yield heard, " ".join(meant)


def main():
    for f in ("vocab.json", "replacements.json"):      # first run: start from the shipped examples
        mine, example = os.path.join(PERSONAL, f), os.path.join(PERSONAL, f.replace(".json", ".example.json"))
        if not os.path.exists(mine) and os.path.exists(example):
            import shutil
            shutil.copy(example, mine)
    con = db()
    prune_audio()
    try:
        import build_vocab
        build_vocab.main()
    except Exception as e:
        log("vocabulary rebuild failed:", e)
    todo = pending(con)
    log(f"{len(todo)} new dictations with audio to learn from")
    if not todo:
        return
    model = teacher_model()
    from cleanup import Personal
    prompt = Personal().asr_prompt  # same hint the live model gets
    today = datetime.date.today().isoformat()
    name_counts = Counter()
    for n, (i, asr, f) in enumerate(todo, 1):
        while con.execute("select count(*) from dictations where source='live' and "
                          "ts >= datetime('now','localtime','-2 minutes')").fetchone()[0]:
            time.sleep(30)  # you're dictating: stay out of the way
        segs, _ = model.transcribe(f, language=CFG["language"], beam_size=5, vad_filter=True, initial_prompt=prompt,
                                   condition_on_previous_text=False)
        teacher = " ".join(s.text.strip() for s in segs).strip()
        con.execute("insert or replace into teacher (dictation_id, audio_file, teacher_text) values (?,?,?)",
                    (i, f, teacher))
        for heard, meant in diff_pairs(asr, teacher):
            row = con.execute("select count, days from pairs where heard=? and meant=?", (heard, meant)).fetchone()
            days = set(json.loads(row[1])) if row else set()
            days.add(today)
            con.execute("insert or replace into pairs (heard, meant, count, days, status) values (?,?,?,?, "
                        "coalesce((select status from pairs where heard=? and meant=?),'candidate'))",
                        (heard, meant, (row[0] if row else 0) + 1, json.dumps(sorted(days)), heard, meant))
        toks = tokens(teacher)
        for k, w in enumerate(toks):
            if k > 0 and namey(w, True) and len(w) > 2:
                name_counts[w] += 1
        con.commit()
        if n % 25 == 0:
            log(f"  {n}/{len(todo)}")
    promote(con, name_counts)


def promote(con, name_counts):
    rp = os.path.join(PERSONAL, "replacements.json")
    rep = json.load(open(rp, encoding="utf-8"))
    known = {p[0].lower() for p in rep["pairs"]}
    # words the teacher also writes in lower case are ordinary English, never "fix" them
    teacher_words = Counter(w for (t,) in con.execute("select teacher_text from teacher") for w in tokens(t))
    added = []
    for heard, meant, count, days in con.execute(
            "select heard, meant, count, days from pairs where status='candidate'").fetchall():
        if count < MIN_COUNT or len(json.loads(days)) < MIN_DAYS or heard in known:
            continue
        # conflicting targets for the same sound -> not safe
        others = con.execute("select sum(count) from pairs where heard=? and meant!=?", (heard, meant)).fetchone()[0] or 0
        if others >= count / 2 or (len(heard.split()) == 1 and teacher_words.get(heard, 0) >= 2):
            con.execute("update pairs set status='rejected' where heard=? and meant=?", (heard, meant))
            continue
        rep["pairs"].append([heard, meant, f"learned {datetime.date.today()} x{count}"])
        con.execute("update pairs set status='learned' where heard=? and meant=?", (heard, meant))
        added.append(f"{heard!r} -> {meant!r} (seen {count}x)")
    if added:
        json.dump(rep, open(rp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    vp = os.path.join(PERSONAL, "vocab.json")
    vocab = json.load(open(vp, encoding="utf-8"))
    have = {w.lower() for w in vocab["names"] + vocab["context"]}
    new_terms = [w for w, c in name_counts.most_common() if c >= MIN_COUNT and w.lower() not in have
                 and teacher_words.get(w.lower(), 0) == 0][:10]
    if new_terms:
        vocab["context"] += new_terms
        json.dump(vocab, open(vp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    con.commit()

    report = [f"# wisper learning report {datetime.datetime.now():%Y-%m-%d %H:%M}", "",
              f"New mishearing fixes ({len(added)}):"] + [f"- {a}" for a in added] + \
             ["", f"New vocabulary hints ({len(new_terms)}): " + ", ".join(new_terms), "", "Top pending candidates:"]
    for heard, meant, count, days in con.execute("select heard, meant, count, days from pairs where status='candidate' "
                                                 "order by count desc limit 15"):
        report.append(f"- {heard!r} -> {meant!r}  x{count} on {len(json.loads(days))} day(s)")
    stats = con.execute("""select count(*), avg(total_ms) from dictations
                           where ts >= datetime('now','localtime','-1 day') and coalesce(source,'live')='live'""").fetchone()
    report += ["", f"Last 24h: {stats[0]} dictations, avg {int(stats[1] or 0)} ms from release to typed text"]
    open(os.path.join(ROOT, "data", "learn-report.md"), "w", encoding="utf-8").write("\n".join(report) + "\n")
    log(f"learned {len(added)} fixes, {len(new_terms)} vocab hints")


if __name__ == "__main__":
    main()
