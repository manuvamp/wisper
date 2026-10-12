"""Code-only transcript cleanup (no AI model). ~1 ms per dictation.

Order matters:
  1. hesitation sounds (um, uh, er, hmm)            "it, uh, works"            -> "it works"
  2. stutters / repeated words and phrases           "can you, can you check"   -> "can you check"
  3. filler words set off by punctuation             "Like, where is it?"       -> "Where is it?"
                                                     "the logo, you know, moves" -> "the logo moves"
  4. your mishearings + exact spelling of names      "info tick" -> "Infotik", "discord" -> "Discord"
  5. punctuation/capitalisation tidy-up              ", don't run" -> "Don't run", "i think" -> "I think"
Fillers are only removed where punctuation marks them as an aside, so "I feel like", "you know how" and
"like a circle" (meaningful uses) are left alone.
"""
import functools, glob, json, os, re, logging

log = logging.getLogger("wisper")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- "is this ordinary English?" (Whisper's vocabulary)
@functools.lru_cache(maxsize=1)
def _tokenizer():
    from tokenizers import Tokenizer
    hub = os.path.expanduser(r"~\.cache\huggingface\hub")
    path = (glob.glob(os.path.join(hub, "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo", "snapshots", "*",
                                   "tokenizer.json")) or glob.glob(os.path.join(hub, "*whisper*", "snapshots", "*",
                                                                                "tokenizer.json")))[0]
    return Tokenizer.from_file(path)


try:
    from wordfreq import zipf_frequency
except ImportError:
    zipf_frequency = None


@functools.lru_cache(maxsize=50000)
def is_common(word):
    """Ordinary English word ("reverted", "info", "tick")? Those are never "corrected" into a name.
    wordfreq frequency (names like Orano/Infotik score 0) + Whisper's vocabulary as a fallback."""
    w = word.lower()
    if zipf_frequency and zipf_frequency(w, "en") >= 2.5:
        return True
    return len(_tokenizer().encode(" " + w, add_special_tokens=False).ids) == 1


def sound_key(word):
    """Rough phonetic key: 'Guardiora'/'Gardiora'/'Guardiorah' -> same key."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return ""
    for a, b in (("ph", "f"), ("ck", "k"), ("qu", "kw"), ("x", "ks"), ("wh", "w"), ("gh", "g"), ("c", "k"),
                 ("q", "k"), ("z", "s"), ("v", "f"), ("y", "i"), ("dg", "j")):
        w = w.replace(a, b)
    first, rest = w[0], re.sub(r"[aeiouh]", "", w[1:])
    return first + re.sub(r"(.)\1+", r"\1", rest)


def _function_word(w):
    """a, is, as, are, to, on, I ... never part of a split-up name."""
    return len(w) <= 2 or (zipf_frequency is not None and zipf_frequency(w.lower(), "en") >= 5.0)


class Fuzzy:
    """Fix words that *sound like* one of your names but came out misspelled or split ("guardio ra", "infotick")."""

    def __init__(self, names):
        from rapidfuzz import fuzz
        self.ratio = fuzz.ratio
        self.single = {}
        for n in names:
            if " " not in n and len(n) >= 4:
                self.single.setdefault(n.lower(), n)
        self.by_key = {}
        for low, n in self.single.items():
            self.by_key.setdefault(sound_key(low), []).append(n)

    def best(self, word, joined=False):
        low = word.lower()
        if low in self.single:   # exact: only change casing for real names (orano -> Orano), not jargon (repos)
            n = self.single[low]
            return n if n != n.lower() else None
        if len(low) < 4:
            return None
        for suf in ("s", "es", "'s"):                          # plural/possessive of a name: keep as said
            if low.endswith(suf) and low[:-len(suf)] in self.single:
                return None
        best, score = None, 0.0
        for n in self.by_key.get(sound_key(low), []):        # same sound key: lenient spelling threshold
            s = self.ratio(low, n.lower())
            if s >= (80 if not joined else 85) and s > score:
                best, score = n, s
        if best is None:                                     # different key: must be a very close spelling
            for nl, n in self.single.items():
                if abs(len(nl) - len(low)) <= 2:
                    s = self.ratio(low, nl)
                    if s >= (90 if not joined else 93) and s > score:
                        best, score = n, s
        return best

    def fix(self, text):
        toks = re.split(r"(\W+)", text)                       # words at even indexes, separators at odd
        i = 0
        while i < len(toks):
            w = toks[i]
            if w and w[0].isalpha():
                # two words that are really one name: "info tick" -> Infotik, "hyper frames" -> HyperFrames
                nxt = toks[i + 2] if i + 2 < len(toks) else ""
                if (nxt[:1].isalpha() and toks[i + 1] == " " and w.lower() not in self.single
                        and nxt.lower() not in self.single and not _function_word(w) and not _function_word(nxt)):
                    j = w + nxt
                    m = self.best(j, joined=True) if len(j) >= 6 else None
                    if m and abs(len(m) - len(j)) <= 1 and not (
                            is_common(w) and is_common(nxt) and self.ratio(j.lower(), m.lower()) < 95):
                        toks[i:i + 3] = [m]
                        i += 1
                        continue
                if not is_common(w):
                    m = self.best(w)
                    if m:
                        toks[i] = m
            i += 1
        return "".join(toks)


SPLIT_STOP = {"in", "on", "up", "at", "to", "of", "it", "is", "do", "go", "me"}


class Personal:
    """personal/vocab.json + personal/replacements.json, hot-reloaded when either file changes."""
    FILES = ("vocab.json", "replacements.json", "vocab_auto.json")

    def __init__(self):
        self.stamp = None
        self.reload()

    def _path(self, f):
        """personal/<f>, or the shipped personal/<name>.example.json until you make your own copy."""
        p = os.path.join(ROOT, "personal", f)
        example = p.replace(".json", ".example.json")
        return p if os.path.exists(p) or not os.path.exists(example) else example

    def _stamp(self):
        return tuple(os.path.getmtime(self._path(f)) if os.path.exists(self._path(f)) else 0 for f in self.FILES)

    def maybe_reload(self):
        if self._stamp() != self.stamp:
            self.reload()

    def reload(self):
        self.stamp = self._stamp()
        vocab = json.load(open(self._path("vocab.json"), encoding="utf-8"))
        self.names, self.context = vocab["names"], vocab["context"]
        auto = []
        if os.path.exists(self._path("vocab_auto.json")):
            auto = json.load(open(self._path("vocab_auto.json"), encoding="utf-8"))["terms"]
        known = {n.lower() for n in self.names + self.context}
        self.auto = [t for t in auto if t.lower() not in known]
        pairs = json.load(open(self._path("replacements.json"), encoding="utf-8"))["pairs"]
        subs = sorted(((src, dst) for src, dst, *_ in pairs), key=lambda p: -len(p[0]))  # longest first
        subs += [(n, n) for n in self.names]                                             # canonical casing
        # CamelCase names said as separate words: "test flight" -> TestFlight, "chat GPT" -> ChatGPT
        for n in self.names + [t for t in self.auto if re.search(r"[a-z][A-Z]", t)]:
            parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", n)
            if len(parts) >= 2 and "".join(parts) == n.replace(" ", "") and all(len(p) >= 2 for p in parts) \
                    and not any(p.lower() in SPLIT_STOP for p in parts):   # "Linked In": "logged in and linked in"
                subs.append((" ".join(parts), n))
        self.subs = [(re.compile(r"(?<![\w'])" + re.escape(s).replace(r"\ ", r"[\s-]*") + r"(?![\w'])", re.I), d)
                     for s, d in subs]
        self.fuzzy = Fuzzy(self.names + self.auto)
        # Whisper prompt: well-formed text (capitalised, punctuated) so output keeps that style; ~224 token cap.
        terms = self.names[:40] + self.context[:10] + self.auto[:25]
        self.asr_prompt = "Notes on " + ", ".join(terms) + "."
        # leaves room for the previous chunk's words (later chunks of a long dictation)
        self.asr_prompt_short = "Notes on " + ", ".join(self.names[:25] + self.auto[:10]) + "."
        log.info("personal data loaded: %d names, %d auto terms, %d replacements", len(self.names), len(self.auto),
                 len(pairs))


SEP = r"(?:[,.…;:]*|\s*[-—]+)\s+"                       # between repeats: "the, the" "the... the" "the - the"
HESITATION = re.compile(r"(?<![\w'])(?:u+h+m*|u+m+|e+r+m+|e+r+|hm+|mm+|ehm)(?![\w'])[,.…]?", re.I)
REPEAT_PHRASE = re.compile(r"\b((?:\w+[\s,]+){1,3}?\w+)(?:" + SEP + r"\1\b)+", re.I)
REPEAT_WORD = re.compile(r"\b(\w+)(?:" + SEP + r"\1\b)+", re.I)
KEEP_DOUBLE = {"had", "that", "is", "very", "really", "no", "bye", "ha", "go", "yeah", "okay", "ok", "hello", "hey"}
FILLER = r"(?:you know|like|basically|kind of|sort of|i guess|you see)"
FILLER_ASIDE = re.compile(r",\s*" + FILLER + r"\s*(?=[,.?!])", re.I)                 # "x, you know, y" -> "x, y"
FILLER_START = re.compile(r"(^|[.?!]\s+)" + FILLER + r",\s+", re.I)                  # "Like, where" -> "Where"
SENT_START = re.compile(r"(^|[.?!]\s+)([a-z])")


def clean(text, personal):
    t = " ".join(text.split())
    if not t:
        return ""
    t = HESITATION.sub("", t)
    t = REPEAT_PHRASE.sub(lambda m: m.group(1), t)
    t = REPEAT_WORD.sub(lambda m: m.group(0) if m.group(1).lower() in KEEP_DOUBLE and
                        len(re.findall(r"\w+", m.group(0))) == 2 else m.group(1), t)
    t = FILLER_ASIDE.sub("", t)
    t = FILLER_START.sub(r"\1", t)
    for rx, dst in personal.subs:
        t = rx.sub(dst, t)
    t = personal.fuzzy.fix(t)
    # punctuation and capitalisation
    t = re.sub(r"\s+([,.;:!?…])", r"\1", t)              # no space before punctuation
    t = re.sub(r"([,;:])(?=[^\s\d])", r"\1 ", t)          # one space after
    t = re.sub(r"([,;:])\s*([,;:])", r"\1", t)            # ", ," -> ","
    t = re.sub(r"[,;:]\s*([.!?])", r"\1", t)              # ",." -> "."
    t = re.sub(r"(^|[.!?]\s+)[,;:]\s*", r"\1", t)         # stray comma at a sentence start
    t = re.sub(r"\bi\b(?!['.-]\w)", "I", t)               # i -> I (not "i.e.", not "i-th")
    t = re.sub(r"\bi'(m|ll|d|ve)\b", lambda m: "I'" + m.group(1), t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t).strip(" ,;:")
    t = SENT_START.sub(lambda m: m.group(1) + m.group(2).upper(), t)
    return t


def looks_like_hallucination(text, speech_s):
    """Whisper's classic noise outputs: other scripts, 'Thank you.' from silence, endless repeats."""
    if not text.strip():
        return True
    letters = [c for c in text if not c.isspace()]
    if letters and sum(bool(NON_LATIN.match(c)) for c in letters) / len(letters) > 0.3:
        return True
    norm = re.sub(r"[^\w\s]", "", text.lower()).strip()
    if norm in PHANTOMS and speech_s < 2.0:
        return True
    words = norm.split()
    return len(words) >= 8 and len(set(words)) / len(words) < 0.2


NON_LATIN = re.compile("[^\\x00-\\u024F\\u2000-\\u206F\\u20AC\\u2122\\s\\d]")
PHANTOMS = {"thank you", "thanks for watching", "thank you for watching", "you", "bye", "music",
            "applause", "subtitles by the amaraorg community", "please subscribe"}
