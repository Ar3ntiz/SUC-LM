"""Cross-check of the C++ byte-level BPE tokenizer (cpp/src/tokenizer.cpp, tool suc-tokenize) against HF transformers'
Qwen3 tokenizer: exact token ids on WikiText-2 test lines and a hand-written / randomized edge-case corpus, decode
equality and the decode(encode(x)) == x round trip.

Skipped unless suc-tokenize is built into $SUC_CPP_BUILD (default ~/suc-local/cpp-build) and the Qwen3-0.6B
tokenizer.json is in the HF cache. Runs offline (HF_HUB_OFFLINE=1).
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import unicodedata
from pathlib import Path

import pytest

BUILD = Path(os.environ.get("SUC_CPP_BUILD", Path.home() / "suc-local" / "cpp-build"))
TOOL = BUILD / "suc-tokenize"
MODEL, REV = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"
TOKENIZER_SHA = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
HUB = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
TOKENIZER_JSON = HUB / "models--Qwen--Qwen3-0.6B" / "snapshots" / REV / "tokenizer.json"
# Qwen3.8-Flash-Next (qwen4_exp): the Qwen3.5 Split regex (letter runs include \p{M}), 248k vocab. HF cache (the Mac
# Studio has the snapshot) or $SUC_QWEN38_TOKENIZER; the reference is HF `tokenizers` on the same file.
QWEN38_REV = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
QWEN38_JSON = Path(os.environ.get("SUC_QWEN38_TOKENIZER") or
                   HUB / "models--Qwen--Qwen3.8-Flash-Next" / "snapshots" / QWEN38_REV / "tokenizer.json")

pytestmark = [
    pytest.mark.skipif(not TOOL.exists(), reason=f"suc-tokenize not built in {BUILD} (run cpp/build.sh)"),
    pytest.mark.skipif(not TOKENIZER_JSON.exists(), reason=f"{TOKENIZER_JSON} not in the HF cache"),
]

EDGE_CASES = [
    "",
    " ",
    "Hello world",
    "Hello, world! How are you?",
    # accents: NFC and NFD forms (the NFC normalizer must make them identical)
    "café naïve résumé Ångström",
    unicodedata.normalize("NFD", "café naïve résumé Ångström"),
    "é ä ộ ṩ ṩ ṩ q̣̇",
    "Å Ω ̈́ क़ שּׁ ᾀ",   # singletons / excluded compositions
    unicodedata.normalize("NFD", "Tiếng Việt có dấu: Hà Nội, Đà Nẵng"),
    # CJK, Korean (precomposed and conjoining jamo), Japanese
    "中文分词测试，北京欢迎你。",
    "日本語のテキスト、カタカナとひらがな。",
    "한국어 텍스트 각 한글",
    unicodedata.normalize("NFD", "한국어 텍스트"),
    # emoji: ZWJ sequences, skin tones, flags, keycaps, variation selectors
    "👨‍👩‍👧‍👦 👍🏽 👩🏿‍💻 🏳️‍🌈 🇮🇹🇯🇵 1️⃣ ❤️ ☺︎",
    "emoji😀in😀text😀",
    # Arabic, Hebrew, Hindi with combining marks, Thai
    "مرحبا بالعالم، كيف حالك؟ ١٢٣",
    "שָׁלוֹם עוֹלָם",
    "नमस्ते दुनिया, क़ ख़ ग़",
    unicodedata.normalize("NFD", "क़ ख़ ग़ ज़ ड़ ढ़ फ़ य़"),
    "สวัสดีครับ",
    # whitespace
    "a  b   c    d",
    "\t\tindented\twith\ttabs",
    "line1\r\nline2\r\n\r\nline3",
    "trailing spaces   ",
    "   leading spaces",
    "multiple\n\n\n\nnewlines\n",
    "\n",
    "\n\n",
    " \n \n ",
    "mixed \t \n\t  \r\n  end",
    " nbsp  em　ideographic sep\u0085nel",
    "\x0b\x0c\x1c\x1d\x1e\x1f",
    # contractions
    "I'M you'RE they'll he'd she's it'S we've don't CAN'T Y'ALL",
    "'s 't 're 've 'm 'll 'd 'S 'T 'RE 'VE 'M 'LL 'D 'x",
    "''s ' s rock'n'roll l'été",
    # digits
    "1234567890 3.14159 1,000,000 -42 +7 0x1F",
    "٠١٢٣٤٥٦٧٨٩ ①②③ ½ Ⅻ",
    # code
    "def f(x):\n    if x > 0:\n        return x ** 2\n    return -x\n",
    "int main() {\n\tprintf(\"%d\\n\", a[i]);\n\treturn 0;\n}\n",
    "    \n        \n",
    "x = {'a': [1, 2], \"b\": None}  # comment",
    "<div class=\"x\">&amp; &lt;tag&gt;</div>",
    "https://example.com/path?q=1&r=2#frag",
    "$$$ ### @@@ !!! ??? ... --- ___ *** ~~~",
    # special / added tokens embedded in text
    "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are helpful.<|im_end|>",
    "text<|endoftext|>more text",
    "<think>\nreasoning\n</think>\n\nanswer",
    "<tool_call>{\"name\": \"f\"}</tool_call>",
    "<|im_start|><|im_end|><|endoftext|>",
    "<|im_start|  <|im_end <|endoftext|>> <<|im_end|>>",
    " <|im_start|> spaced <|im_end|> ",
    "<|fim_prefix|>def<|fim_suffix|>\n<|fim_middle|>",
    # long runs
    "a" * 300,
    " " * 100 + "x",
    "-" * 200,
    "\n" * 50,
    "ab" * 150,
    "😀" * 40,
]


def _random_corpus(n: int = 2000, seed: int = 0) -> list[str]:
    """Random strings over pools that stress the regex classes and NFC (combining marks in random order)."""
    rng = random.Random(seed)
    pools = [
        "abcXYZ'sStTlLdDmMrReEvV ",
        " \t\n\r\x0b\x0c  　\u0085",
        "0123456789٠١٢①½Ⅻ",
        ".,;:!?-_()[]{}<>\"'`~@#$%^&*+=|\\/",
        "़゙्ְּّ̧̨〪̣̖̀́̂̃̈̊ͅ",
        "aeiouAEIOUnNcCsSkKgGhHΑαАаאاकก각가あ中",
        "😀👍🏽‍️\U0001f1ee\U0001f1f9⃣",
        "éÅÅΩẛẠǕक़שּׁᾀ̈́",
    ]
    out = []
    for _ in range(n):
        k = rng.randint(1, 3)
        chars = "".join(rng.choice(pools) for _ in range(k))
        s = "".join(rng.choice(chars) for _ in range(rng.randint(1, 60)))
        if rng.random() < 0.2:
            s = s.replace(rng.choice(s), rng.choice(["<|im_start|>", "<|im_end|>", "<think>", "\n"]), 1)
        out.append(s)
    # uniformly random code points (assigned or not, all planes): class tables and NFC data
    for _ in range(300):
        cps = [rng.randrange(0x110000) for _ in range(rng.randint(1, 30))]
        out.append("".join(chr(c) for c in cps if not 0xD800 <= c < 0xE000))
    # every canonically decomposable BMP character in NFD form, in chunks
    dec = [unicodedata.normalize("NFD", chr(c)) for c in range(0xC0, 0x10000)
           if not 0xD800 <= c < 0xE000 and unicodedata.decomposition(chr(c))[:1] not in ("", "<")]
    out += [" ".join(dec[i:i + 50]) for i in range(0, len(dec), 50)]
    return out


def _run(args: list[str], tokenizer: Path | None = None) -> subprocess.CompletedProcess:
    tokenizer = tokenizer or TOKENIZER_JSON
    r = subprocess.run([str(TOOL), "--tokenizer", str(tokenizer), *args], capture_output=True, text=True,
                       encoding="utf-8")
    assert r.returncode == 0, r.stderr
    return r


def _cpp_encode(texts: list[str], tmp_path: Path, add_special: bool = False,
                tokenizer: Path | None = None) -> tuple[list[list[int]], str]:
    p = tmp_path / "in.json"
    p.write_text(json.dumps(texts, ensure_ascii=True))
    r = _run(["--encode-json", str(p)] + (["--add-special-tokens"] if add_special else []), tokenizer=tokenizer)
    lines = r.stdout.split("\n")[:-1]
    assert len(lines) == len(texts)
    return [[int(x) for x in ln.split(",")] if ln else [] for ln in lines], r.stderr


def _cpp_decode(ids: list[list[int]], tmp_path: Path, skip_special: bool = False,
                tokenizer: Path | None = None) -> list[str]:
    p = tmp_path / "ids.json"
    p.write_text(json.dumps(ids))
    r = _run(["--decode-json", str(p)] + (["--skip-special"] if skip_special else []), tokenizer=tokenizer)
    out = [json.loads(ln) for ln in r.stdout.split("\n")[:-1]]
    assert len(out) == len(ids)
    return out


@pytest.fixture(scope="module")
def hf():
    os.environ["HF_HUB_OFFLINE"] = "1"
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(MODEL, revision=REV)


@pytest.fixture(scope="module")
def wikitext_lines() -> list[str]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    pq = pytest.importorskip("pyarrow.parquet")
    from huggingface_hub import hf_hub_download

    from suc.evaluation.data import WIKITEXT_REPO
    try:
        path = hf_hub_download(repo_id=WIKITEXT_REPO, repo_type="dataset",
                               filename="wikitext-2-raw-v1/test-00000-of-00001.parquet")
    except Exception as e:  # noqa: BLE001 - offline and not cached
        pytest.skip(f"WikiText-2 test split not cached: {e}")
    rows = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    return [r for r in rows if r]   # raw rows, not stripped (each keeps its trailing "\n")


def _compare(hf, texts: list[str], tmp_path: Path, label: str, tokenizer: Path | None = None) -> None:
    ours, err = _cpp_encode(texts, tmp_path, tokenizer=tokenizer)
    ref = [hf.encode(t, add_special_tokens=False) for t in texts]
    bad = [i for i, (a, b) in enumerate(zip(ours, ref)) if a != b]
    ours_sp, _ = _cpp_encode(texts, tmp_path, add_special=True, tokenizer=tokenizer)
    ref_sp = [hf.encode(t, add_special_tokens=True) for t in texts]
    bad_sp = [i for i, (a, b) in enumerate(zip(ours_sp, ref_sp)) if a != b]
    # decode: HF ids through the C++ decoder vs transformers.decode, with and without special tokens
    dec = _cpp_decode(ref, tmp_path, tokenizer=tokenizer)
    bad_dec = [i for i, t in enumerate(dec) if t != hf.decode(ref[i], skip_special_tokens=False)]
    dec_skip = _cpp_decode(ref, tmp_path, skip_special=True, tokenizer=tokenizer)
    bad_skip = [i for i, t in enumerate(dec_skip) if t != hf.decode(ref[i], skip_special_tokens=True)]
    # round trip of our own ids: decode(encode(x)) == NFC(x) (== x for NFC input)
    rt = _cpp_decode(ours, tmp_path, tokenizer=tokenizer)
    bad_rt = [i for i, t in enumerate(rt) if t != unicodedata.normalize("NFC", texts[i])
              and t != hf.decode(ref[i])]
    nfc_lines = [i for i, t in enumerate(texts) if unicodedata.normalize("NFC", t) == t]
    bad_rt_exact = [i for i in nfc_lines if rt[i] != texts[i]]
    print(f"\n[{label}] lines={len(texts)} tokens={sum(map(len, ref))} encode_mismatch={len(bad)} "
          f"encode_add_special_mismatch={len(bad_sp)} decode_mismatch={len(bad_dec)} "
          f"decode_skip_special_mismatch={len(bad_skip)} roundtrip_mismatch={len(bad_rt)} "
          f"roundtrip_exact_nfc_lines={len(nfc_lines)} roundtrip_exact_mismatch={len(bad_rt_exact)}")
    print(f"[{label}] {err.strip().splitlines()[-1]}")
    for i in (bad + bad_sp + bad_dec + bad_skip + bad_rt + bad_rt_exact)[:5]:
        print(f"  mismatch line {i}: {texts[i]!r}\n    ours={ours[i]}\n    ref ={ref[i]}")
    assert not (bad or bad_sp or bad_dec or bad_skip or bad_rt or bad_rt_exact)


def test_wikitext2_test_lines(hf, wikitext_lines, tmp_path):
    assert len(wikitext_lines) >= 1000
    _compare(hf, wikitext_lines, tmp_path, "wikitext2-test")


def test_wikitext2_document(hf, wikitext_lines, tmp_path):
    """The whole split as one string (the perplexity convention of suc.evaluation.data)."""
    text = "\n\n".join(wikitext_lines[:2000])
    ours, err = _cpp_encode([text], tmp_path)
    assert ours[0] == hf.encode(text, add_special_tokens=False)
    print(f"\n[wikitext2-document] {len(ours[0])} tokens; {err.strip().splitlines()[-1]}")


def test_edge_cases(hf, tmp_path):
    _compare(hf, EDGE_CASES, tmp_path, "edge-cases")


def test_random_corpus(hf, tmp_path):
    _compare(hf, _random_corpus(), tmp_path, "random+nfd")


def test_encode_lines_mode(hf, wikitext_lines, tmp_path):
    lines = [ln.rstrip("\n") for ln in wikitext_lines[:500]]
    p = tmp_path / "lines.txt"
    p.write_bytes("\n".join(lines).encode())
    out = _run(["--encode-lines", str(p)]).stdout.split("\n")[:-1]
    ours = [[int(x) for x in ln.split(",")] if ln else [] for ln in out]
    assert ours == [hf.encode(t, add_special_tokens=False) for t in lines]


def test_decode_random_ids(hf, tmp_path):
    """Random id sequences (partial UTF-8 byte tokens, special tokens in between) decode as transformers does."""
    rng = random.Random(1)
    special = [hf.convert_tokens_to_ids(t) for t in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>")]
    byte_ids = [hf.convert_tokens_to_ids(t) for t in ("ä", "¸", "Ń", "ð", "Ł", "ĺ", "Ģ", "Ġ", "Ċ")]
    seqs = []
    for _ in range(300):
        seqs.append([rng.choice([rng.randrange(151643), rng.choice(special), rng.choice(byte_ids),
                                 rng.choice(byte_ids)]) for _ in range(rng.randint(0, 12))])
    for skip in (False, True):
        ours = _cpp_decode(seqs, tmp_path, skip_special=skip)
        bad = [i for i, s in enumerate(seqs) if ours[i] != hf.decode(s, skip_special_tokens=skip)]
        assert not bad, [(seqs[i], ours[i], hf.decode(seqs[i], skip_special_tokens=skip)) for i in bad[:3]]


def test_added_token_flags(tmp_path):
    """lstrip / rstrip / single_word / normalized added tokens, against HF tokenizers on a modified Qwen3 file."""
    tk = pytest.importorskip("tokenizers")
    t = tk.Tokenizer.from_file(str(TOKENIZER_JSON))
    t.add_tokens([tk.AddedToken("<L>", lstrip=True, normalized=False),
                  tk.AddedToken("<R>", rstrip=True, normalized=False),
                  tk.AddedToken("<LR>", lstrip=True, rstrip=True, normalized=False)])
    t.add_special_tokens([tk.AddedToken("<SP>", lstrip=True, normalized=False)])
    t.add_tokens([tk.AddedToken("word", single_word=True, normalized=False),
                  tk.AddedToken("caf\u00e9!", normalized=True)])
    path = tmp_path / "tokenizer.json"
    t.save(str(path))
    texts = ["a <L> b", "a   <L>b", "<L>", "  <L>  ", "x<R>   y", "x <R>\n\n y", "p  <LR>  q", "<L><R><LR>",
             "a <SP>  b", "<R> <L>", "word words sword word. (word) wordy aword word", "\u00e0word word\u00e0",
             "cafe\u0301! caf\u00e9! xcaf\u00e9!y", "<|im_start|> <L> <|im_end|>", " \u3000<L>\u00a0<R>\u2003z"]
    ref = [t.encode(x, add_special_tokens=False).ids for x in texts]
    ours, _ = _cpp_encode_with(texts, tmp_path, path)
    assert ours == ref
    p = tmp_path / "ids.json"
    p.write_text(json.dumps(ref))
    for skip in (False, True):
        out = [json.loads(ln) for ln in _run(["--decode-json", str(p)] + (["--skip-special"] if skip else []),
                                             tokenizer=path).stdout.split("\n")[:-1]]
        assert out == [t.decode(r, skip_special_tokens=skip) for r in ref]


def _cpp_encode_with(texts: list[str], tmp_path: Path, tokenizer: Path) -> tuple[list[list[int]], str]:
    p = tmp_path / "in2.json"
    p.write_text(json.dumps(texts, ensure_ascii=True))
    r = _run(["--encode-json", str(p)], tokenizer=tokenizer)
    lines = r.stdout.split("\n")[:-1]
    return [[int(x) for x in ln.split(",")] if ln else [] for ln in lines], r.stderr


def test_text_and_decode_cli(hf):
    text = "<|im_start|>user\nCiao, come stai? 😀<|im_end|>"
    r = _run(["--text", text, "--json"])
    ids = json.loads(r.stdout)["ids"]
    assert ids == hf.encode(text, add_special_tokens=False)
    r = _run(["--decode", ",".join(map(str, ids)), "--json"])
    assert json.loads(r.stdout)["text"] == text
    r = _run(["--decode", ",".join(map(str, ids)), "--skip-special"])
    assert r.stdout == hf.decode(ids, skip_special_tokens=True) + "\n"


def test_unicode_tables_up_to_date():
    """cpp/src/unicode_tables.inc matches what the generator derives from the installed tokenizers."""
    pytest.importorskip("tokenizers")
    gen = Path(__file__).resolve().parents[2] / "cpp" / "tools" / "gen_unicode_tables.py"
    r = subprocess.run([os.sys.executable, str(gen), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_suc1_tokenizer_id_resolution(tmp_path):
    """--suc1 resolves the header tokenizer_id through the HF cache and checks the sha256."""
    import mlx.core as mx

    from suc.io.suc1 import write_suc1

    def make(tid: str) -> Path:
        p = tmp_path / f"m{len(list(tmp_path.iterdir()))}.suc1"
        write_suc1(p, architecture={"model_type": "qwen3"}, tokenizer_id=tid,
                   tensors={"w": mx.zeros((4, 4), dtype=mx.float32)})
        return p

    good = make(f"{MODEL}@{REV};tokenizer.json:sha256:{TOKENIZER_SHA}")
    r = subprocess.run([str(TOOL), "--suc1", str(good), "--text", "hi there"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert str(TOKENIZER_JSON) in r.stderr
    bad = make(f"{MODEL}@{REV};tokenizer.json:sha256:{'0' * 64}")
    r = subprocess.run([str(TOOL), "--suc1", str(bad), "--text", "hi"], capture_output=True, text=True)
    assert r.returncode != 0 and "sha256 mismatch" in r.stderr
    missing = make(f"Nobody/none@{'a' * 40}")
    r = subprocess.run([str(TOOL), "--suc1", str(missing), "--text", "hi"], capture_output=True, text=True)
    assert r.returncode != 0 and "not found" in r.stderr


# ------------------------------------------------------------------ Qwen3.8-Flash-Next (qwen4_exp) tokenizer

class _HFTokenizers:
    """HF `tokenizers` on a tokenizer.json with the transformers encode / decode call signature used by _compare."""

    def __init__(self, path: Path):
        tk = pytest.importorskip("tokenizers")
        self.t = tk.Tokenizer.from_file(str(path))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.t.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self.t.decode(ids, skip_special_tokens=skip_special_tokens)


qwen38 = pytest.mark.skipif(not QWEN38_JSON.exists(), reason=f"{QWEN38_JSON} not available")


@pytest.fixture(scope="module")
def hf38():
    return _HFTokenizers(QWEN38_JSON)


@qwen38
def test_qwen38_regex_is_the_qwen35_variant():
    pre = json.loads(QWEN38_JSON.read_text())["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    assert "[\\p{L}\\p{M}]+" in pre and "[^\\s\\p{L}\\p{M}\\p{N}]+" in pre


@qwen38
def test_qwen38_wikitext2_and_edge_cases(hf38, wikitext_lines, tmp_path):
    _compare(hf38, wikitext_lines, tmp_path, "qwen3.8 wikitext2-test", tokenizer=QWEN38_JSON)
    _compare(hf38, EDGE_CASES, tmp_path, "qwen3.8 edge-cases", tokenizer=QWEN38_JSON)


@qwen38
def test_qwen38_random_corpus(hf38, tmp_path):
    marks = ["".join(random.Random(i).choice("\u0300\u0301\u0308\u0323\u093c\u094d\u05b0\u05bc\u0e31\u0e34\u20dd")
                     for _ in range(3)) for i in range(50)]
    extra = [f"{m}word {m} x{m}y {m}{m}\n{m}" for m in marks] + ["\u0301abc", "a\u0301\u0302b", " \u0301", "\n\u0301x"]
    _compare(hf38, _random_corpus() + extra, tmp_path, "qwen3.8 random+nfd+marks", tokenizer=QWEN38_JSON)


# ------------------------------------------------------------------ DeepSeek-V4-Flash (deepseek_v4) tokenizer

def _dsv4_json() -> Path:
    env = os.environ.get("SUC_DSV4_TOKENIZER")
    if env:
        return Path(env)
    snaps = sorted((HUB / "models--deepseek-ai--DeepSeek-V4-Flash" / "snapshots").glob("*/tokenizer.json"))
    return snaps[-1] if snaps else HUB / "models--deepseek-ai--DeepSeek-V4-Flash" / "missing"


DSV4_JSON = _dsv4_json()
dsv4 = pytest.mark.skipif(not DSV4_JSON.exists(), reason=f"{DSV4_JSON} not available")
DSV4_SPECIAL = ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "<think>",
                "</think>", "<｜tool▁calls▁begin｜>", "<dsml:"]


@pytest.fixture(scope="module")
def hfds():
    return _HFTokenizers(DSV4_JSON)


@dsv4
def test_dsv4_pre_tokenizer_is_the_three_split_sequence():
    pre = json.loads(DSV4_JSON.read_text())["pre_tokenizer"]
    assert [p["type"] for p in pre["pretokenizers"]] == ["Split", "Split", "Split", "ByteLevel"]


@dsv4
def test_dsv4_wikitext2_and_edge_cases(hfds, wikitext_lines, tmp_path):
    _compare(hfds, wikitext_lines, tmp_path, "dsv4 wikitext2-test", tokenizer=DSV4_JSON)
    _compare(hfds, EDGE_CASES, tmp_path, "dsv4 edge-cases", tokenizer=DSV4_JSON)


@dsv4
def test_dsv4_random_corpus(hfds, tmp_path):
    rng = random.Random(5)
    pools = ["0123456789", "日本語のテキストカタカナ中文字符一龥ぁゟ゠ヿ", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
             "abcXYZ", " \t\n\r", "«»¿¡§¶†‡•…‰′″‹›€£¥©®™°±×÷→⇒∀∂∑√∞≈≠≤≥", "़्̀́̂"]
    extra = []
    for _ in range(1500):
        chars = "".join(rng.choice(pools) for _ in range(rng.randint(1, 3)))
        s = "".join(rng.choice(chars) for _ in range(rng.randint(1, 50)))
        if rng.random() < 0.3:
            s = s.replace(rng.choice(s), rng.choice(DSV4_SPECIAL), 1)
        extra.append(s)
    extra += ["1234567", "a1234b", "x.y.z", "#include <stdio.h>", "foo.bar(baz)", "'hello' \"world\"", "-->\n\n",
              "  \n  \n", "价格: ¥1,234.56", "<｜User｜>你好<｜Assistant｜></think>", "e.g., i.e.; etc."]
    _compare(hfds, _random_corpus() + extra, tmp_path, "dsv4 random", tokenizer=DSV4_JSON)


# ------------------------------------------------------------------ Qwen-family regex variants (digit groups)

QWEN3_RX = ("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
            "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")
DIGIT_TEXTS = ["1", "12", "123", "1234", "1234567", "a1234b", "3.14159", "x12 345 6789y", "٠١٢٣٤", "①②③④⑤",
               "½12Ⅻ3", "2026-10-05T07:12:38Z", "  1234  \n5678\n", "price: 1,234,567.89$", "'s123'll45"]


@pytest.mark.parametrize("variant", [("\\p{N}{1,3}", False), ("\\p{N}{1,2}", False), ("\\p{N}{1,3}", True)])
def test_qwen_regex_digit_group_variants(variant, tmp_path):
    """The Qwen3 Split regex with \\p{N}{1,d} (GLM-5.x: d = 3), also combined with the Qwen3.5 \\p{M} letter runs, on
    the Qwen3 vocabulary: ids equal HF tokenizers (Oniguruma) on digit-heavy strings, edge cases and the random corpus."""
    tk = pytest.importorskip("tokenizers")
    num, marks = variant
    rx = QWEN3_RX.replace("|\\p{N}|", f"|{num}|")
    if marks:
        rx = rx.replace("]?\\p{L}+|", "]?[\\p{L}\\p{M}]+|").replace("[^\\s\\p{L}\\p{N}]+", "[^\\s\\p{L}\\p{M}\\p{N}]+")
    j = json.loads(TOKENIZER_JSON.read_text())
    j["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = rx
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(j))
    t = tk.Tokenizer.from_file(str(path))
    texts = DIGIT_TEXTS + EDGE_CASES + _random_corpus(600, seed=7)
    ours, _ = _cpp_encode_with(texts, tmp_path, path)
    ref = [t.encode(x, add_special_tokens=False).ids for x in texts]
    bad = [(texts[i], a, b) for i, (a, b) in enumerate(zip(ours, ref)) if a != b]
    assert not bad, bad[:3]


def test_unsupported_split_regex_fails_loudly(tmp_path):
    j = json.loads(TOKENIZER_JSON.read_text())
    j["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = QWEN3_RX.replace("|\\p{N}|", "|\\p{N}+|")
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(j))
    r = subprocess.run([str(TOOL), "--tokenizer", str(path), "--text", "a"], capture_output=True, text=True)
    assert r.returncode != 0 and "Split regex" in r.stderr


# ------------------------------------------------------------------ GLM-5.3-Flash (glm5_next) tokenizer

def _glm_json() -> Path:
    env = os.environ.get("SUC_GLM_TOKENIZER")
    if env:
        return Path(env)
    snaps = sorted((HUB / "models--zai-org--GLM-5.3-Flash" / "snapshots").glob("*/tokenizer.json"))
    return snaps[-1] if snaps else HUB / "models--zai-org--GLM-5.3-Flash" / "missing"


GLM_JSON = _glm_json()
glm = pytest.mark.skipif(not GLM_JSON.exists(), reason=f"{GLM_JSON} not available")
GLM_SPECIAL = ["[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<think>", "</think>", "<|endoftext|>",
               "<tool_call>", "<arg_key>", "/nothink"]


@pytest.fixture(scope="module")
def hfglm():
    return _HFTokenizers(GLM_JSON)


@glm
def test_glm_regex_is_the_digit_group_variant():
    pre = json.loads(GLM_JSON.read_text())["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    assert pre == QWEN3_RX.replace("|\\p{N}|", "|\\p{N}{1,3}|")


@glm
def test_glm_wikitext2_and_edge_cases(hfglm, wikitext_lines, tmp_path):
    _compare(hfglm, wikitext_lines, tmp_path, "glm5.3 wikitext2-test", tokenizer=GLM_JSON)
    _compare(hfglm, EDGE_CASES + DIGIT_TEXTS, tmp_path, "glm5.3 edge-cases", tokenizer=GLM_JSON)


@glm
def test_glm_random_corpus(hfglm, tmp_path):
    rng = random.Random(11)
    extra = []
    for s in _random_corpus(800, seed=3):
        if s and rng.random() < 0.3:
            s = s.replace(rng.choice(s), rng.choice(GLM_SPECIAL), 1)
        extra.append(s)
    extra += ["[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Ciao, come stai?<|assistant|><think>",
              "<|assistant|><think>a</think>b", "x/nothink y", "1234567890 0.000001 1e-10"]
    _compare(hfglm, extra, tmp_path, "glm5.3 random", tokenizer=GLM_JSON)
