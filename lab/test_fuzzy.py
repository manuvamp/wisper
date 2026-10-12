"""Every word the sound-alike fixer would change across all historical transcripts."""
import sys, sqlite3, collections, re
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\wisper\service")
import cleanup
P = cleanup.Personal()
texts = [r[0] for r in sqlite3.connect(r"D:\wisper\data\openwhispr-history.db").execute(
    "select text from transcriptions where status='completed'")]
texts += [r[0] for r in sqlite3.connect(r"D:\wisper\data\wisper.db").execute("select asr_text from dictations where asr_text!=''")]
changes = collections.Counter()
for t in texts:
    a = re.split(r"(\W+)", t)
    b = re.split(r"(\W+)", P.fuzzy.fix(t))
    if a != b:
        import difflib
        sm = difflib.SequenceMatcher(None, a, b)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op != "equal":
                changes[("".join(a[i1:i2]), "".join(b[j1:j2]))] += 1
print(len(texts), "transcripts;", sum(changes.values()), "fixes;", len(changes), "distinct")
for (x, y), n in changes.most_common(120):
    print(f"{n:4d}  {x!r} -> {y!r}")
