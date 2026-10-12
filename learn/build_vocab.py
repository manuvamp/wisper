"""Build your personal vocabulary from your own data (no AI): names/tools/assets you actually say.

Candidates come from what you work with:
  - a Graphify memory graph, if you have one (document headings + project names; config.json "vocab_graph")
  - folder names of your skills and projects (config.json "vocab_folders", globs)
They are kept only if you have actually said them in your dictation history (wisper's own, plus an imported
OpenWhispr history in data/openwhispr-history.db if present), and only if they are not ordinary English (an ordinary word never needs "correcting").
Output: personal/vocab_auto.json  -> used for the Whisper hint prompt and the sound-alike name fixer.

usage: python build_vocab.py
"""
import collections, glob, json, os, re, sqlite3, sys
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service"))
from cleanup import is_common  # noqa: E402
from wordfreq import zipf_frequency  # noqa: E402

CFG = {**json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8")),  # + your own overrides:
       **(json.load(open(os.path.join(ROOT, "config.local.json"), encoding="utf-8"))
          if os.path.exists(os.path.join(ROOT, "config.local.json")) else {})}
GRAPH = CFG.get("vocab_graph", "")                     # optional Graphify graph.json of your notes/code
FOLDERS = CFG.get("vocab_folders", [r"~\.claude\skills\*"])   # globs; folder names become candidates
MIN_SAID = 3
LOWER_COUNTS = {}
CURATED_LOWER = []


def candidates():
    """term -> preferred spelling, from the things you work with."""
    labels = []
    try:
        g = json.load(open(GRAPH, encoding="utf-8")) if GRAPH else {"nodes": []}
        for n in g["nodes"]:
            if n.get("file_type") in ("document", "concept", "module"):
                labels.append(n.get("label", ""))
            if n.get("project"):
                labels.append(n["project"].replace("-", " "))
    except Exception as e:
        print("graph not readable:", e)
    for pat in map(os.path.expanduser, FOLDERS):
        labels += [os.path.basename(p).replace("-", " ").replace("_", " ") for p in glob.glob(pat) if os.path.isdir(p)]
    spell = collections.defaultdict(collections.Counter)
    for lab in labels:
        toks = re.findall(r"[A-Za-z][A-Za-z0-9]+(?:\.[a-z]{2,4})?", lab)
        for t in toks:
            if 3 <= len(t) <= 24:
                spell[t.lower()][t] += 1
        for a, b in zip(toks, toks[1:]):                       # "Nano Banana", "Social HQ"
            if a[0].isupper() and b[0].isupper():
                spell[f"{a} {b}".lower()][f"{a} {b}"] += 1
    return {k: c.most_common(1)[0][0] for k, c in spell.items()}


def corpus(lower=True):
    texts = []
    for db, q in ((os.path.join(ROOT, "data", "openwhispr-history.db"), "select text from transcriptions where status='completed'"),
                  (os.path.join(ROOT, "data", "wisper.db"), "select asr_text from dictations where coalesce(source,'live')!='test'")):
        if os.path.exists(db):
            texts += [r[0] or "" for r in sqlite3.connect(db).execute(q)]
    joined = "\n".join(texts)
    return joined.lower() if lower else joined


def nice(term, spelled, cased):
    """Display spelling: ALLCAPS acronyms (GLM), CamelCase as you/your docs write it (TikTok, HyperFrames),
    ordinary jargon stays lowercase (repos, mockups), invented names get a capital (Orano)."""
    words = []
    for w in term.split():
        forms = cased.get(w, collections.Counter())
        total = sum(forms.values()) + LOWER_COUNTS.get(w, 0)
        caps = [f for f in forms if f.isupper()]
        camel = [f for f in forms if any(c.isupper() for c in f[1:]) and not f.isupper()]
        if caps and len(w) <= 5 and sum(forms[f] for f in caps) >= 0.5 * total:
            words.append(max(caps, key=forms.get))                        # GLM, GPT, EC2
        elif camel and sum(forms[f] for f in camel) >= 0.3 * total:
            words.append(max(camel, key=forms.get))                       # TikTok, HyperFrames
        elif not spelled.isupper() and any(c.isupper() for c in spelled[1:]) and " " not in term:
            words.append(spelled)                                         # CamelCase from your docs
        elif zipf_frequency(w, "en") >= 1.0:
            words.append(w)                                               # jargon: repos, mockups
        else:
            words.append(w.capitalize())                                  # invented name: Orano
    return " ".join(words)


def main():
    global CURATED_LOWER
    from cleanup import Personal
    v = json.load(open(Personal._path(None, "vocab.json"), encoding="utf-8"))
    CURATED_LOWER = [c for c in v["names"] + v["context"] if c.islower()]
    cand = candidates()
    text = corpus()
    raw = corpus(lower=False)
    global LOWER_COUNTS
    LOWER_COUNTS = collections.Counter(w for w in re.findall(r"[a-z][a-z0-9]+", raw) if w.islower())
    cased = collections.defaultdict(collections.Counter)
    for w in re.findall(r"[A-Za-z][A-Za-z0-9]+", raw):
        if not w.islower():
            cased[w.lower()][w] += 1
    words = collections.Counter(re.findall(r"[a-z0-9][a-z0-9.']*[a-z0-9]|[a-z0-9]", text))
    pairs = collections.Counter(f"{m.group(1)} {m.group(2)}" for m in            # "nano banana": counted once
                                re.finditer(r"(?<![a-z0-9])([a-z0-9]+) (?=([a-z0-9]+)(?![a-z0-9]))", text))
    out = []
    for term, spelled in cand.items():
        if " " in term:
            n = pairs.get(term, 0)
            if (n < MIN_SAID or all(is_common(w) for w in term.split())       # "You Can", "To Be": not names
                    or any(zipf_frequency(w, "en") >= 4.5 for w in term.split())):  # "The ORANO", "Guardiora Is"
                continue
        else:
            n = words.get(term, 0)
            if n < MIN_SAID or is_common(term):
                continue
        disp = nice(term, spelled, cased)
        if any(term.startswith(c) for c in CURATED_LOWER):   # "subagents" follows your "subagent" (lowercase)
            disp = term
        out.append((n, disp))
    out.sort(reverse=True)
    terms = [t for _, t in out]
    json.dump({"_doc": "Generated by learn/build_vocab.py from your Graphify graph, skills and projects, filtered to "
                       "words you have actually said. Don't edit; put manual terms in vocab.json.",
               "terms": terms, "counts": {t: n for n, t in out}},
              open(os.path.join(ROOT, "personal", "vocab_auto.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{len(cand)} candidates from your graph/skills/projects -> {len(terms)} you actually say")
    print(", ".join(f"{t} ({n})" for n, t in out[:80]))


if __name__ == "__main__":
    main()
