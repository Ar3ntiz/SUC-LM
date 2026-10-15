#!/usr/bin/env python3
"""Build experiments/eval/suite_hard.json: the `hard` suite of `suc eval` (30 cases, 5 per subset), from pinned public
datasets with automatic grading, chosen to DISCRIMINATE between compressed models (the core suite saturates at 29-30/30);
with --per-subset 20 the enlarged `hard20` suite experiments/eval/suite_hard20.json (120 cases, 20 per subset).

Subsets (source files pinned to a dataset commit, downloaded with huggingface_hub; licenses in the suite file):
  GSM8K test, MATH-500, MMLU-Pro test, ARC-Challenge test, BIG-Bench Hard (5 tasks), and LongCtx: synthetic multi-hop
  needles in PG19-test books at ~8k / 16k / 32k tokens.
Disjoint from calibration (suc/data/sources.py): calibration uses GSM8K *train* only, C4 validation, codeparrot,
Dolly, PubMedQA unlabeled; none of these test splits nor PG19 is a calibration or validation source.

Selection is deterministic (no RNG except the seeded needle names / depths) and deliberate: per subset a rule that picks
items spanning difficulty and topic, written into the suite file's "selection" block together with the chosen ids.
Grading reuses the ds4-style extractors (suc.evaluation.answers): integer (GSM8K, MATH integer answers, BBH
multistep arithmetic, LongCtx) and multiple choice (MMLU-Pro 10 options, ARC, BBH choice tasks).
Every case: prompt_style ds4 (final "Answer: <...>" line), think false (thinking off; all three frontier chat
templates support it), per-subset max_tokens budgets for a visible chain of thought.

    .venv/bin/python scripts/build_hard_suite.py [--out experiments/eval/suite_hard.json]
    .venv/bin/python scripts/build_hard_suite.py --per-subset 20      # -> experiments/eval/suite_hard20.json

Enlarged suite (--per-subset N > 5): the SAME rules, applied in rounds. Every rule defines strata (step counts, MATH
level/subject, MMLU-Pro category/source, ARC exam source, BBH task, LongCtx design/length) and a preference order within
a stratum; round 0 takes the first item of every stratum (= exactly the 30-case suite, same ids, same prompts), round k
the next one, strata that run out are skipped, until N items. Key-letter preferences become "least-used key letter
so far" (identical to "not used yet" in round 0). Case order: subset blocks as in the 30-case suite, round-major
within a block, LongCtx last (so a time cap of `suc eval`, SUC_EVAL_DEADLINE, drops whole LongCtx rounds first).

Token sizing of the LongCtx prompts uses the Qwen3 tokenizer (Qwen/Qwen3-0.6B, HF cache); other tokenizers differ by
a few percent, every system of one model sees the same ids (rendered once by suc eval).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parents[1]
PINS = {
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "MIT"),
    "math500": ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "MIT (MATH, Hendrycks et al.)"),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be", "MIT"),
    "arc": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "CC-BY-SA-4.0"),
    "bbh": ("lukaemon/bbh", "982bb89fd79532a8ac676a61fc42eb1aeec63f99", "MIT (BIG-Bench Hard, Suzgun et al.)"),
    "pg19": ("emozilla/pg19-test", "c5e39bf32e33f9111323aa68d7d9000d22722035", "Apache-2.0 (PG19, public-domain books)"),
}
BUDGET = {"GSM8K": 1024, "MATH": 2048, "MMLU-Pro": 1536, "ARC-Challenge": 768, "BBH": 1536, "LongCtx": 512}
TOKENIZER = "Qwen/Qwen3-0.6B"


def _get(key: str, filename: str) -> Path:
    repo, rev, _ = PINS[key]
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=rev))


def _parquet(key: str, filename: str) -> list[dict]:
    return pq.read_table(_get(key, filename)).to_pylist()


def _case(cid: str, source: str, domain: str, kind: str, question: str, answer: str, choices=None) -> dict:
    c = {"id": cid, "source": source, "domain": domain, "kind": kind, "question": question.strip(),
         "answer": answer, "prompt_style": "ds4", "think": False, "max_tokens": BUDGET[source]}
    if choices is not None:
        c["choices"] = list(choices)
    return c


def _rounds(strata: list[list], n: int) -> list:
    """Round-robin over strata (each already in preference order): round k takes the k-th item of every stratum that
    still has one, in stratum order, until n items. Round 0 = one item per stratum."""
    out, k = [], 0
    while len(out) < n and any(k < len(s) for s in strata):
        out += [s[k] for s in strata if k < len(s)][: n - len(out)]
        k += 1
    return out


# ------------------------------------------------------------------------------------------------ GSM8K

_NUM = re.compile(r"\d+(?:\.\d+)?")


def _nums(s: str) -> set[str]:
    return {n.rstrip("0").rstrip(".") if "." in n else n for n in _NUM.findall(s.replace(",", ""))}


def gsm8k(n: int = 5) -> tuple[list[dict], str]:
    rows = _parquet("gsm8k", "main/test-00000-of-00001.parquet")
    pool = []
    for i, r in enumerate(rows):
        sol, ans = r["answer"].split("####")
        steps = sol.count("<<")
        distract = _nums(r["question"]) - _nums(sol)          # numbers stated but never used by the reference solution
        pool.append((i, steps, len(distract), ans.strip().replace(",", ""), r["question"]))
    pool = [p for p in pool if p[2] >= 1 and re.fullmatch(r"\d+", p[3])]
    levels = sorted({p[1] for p in pool}, reverse=True)[:5]
    out = []
    strata = [sorted([p for p in pool if p[1] == s], key=lambda p: (-p[2], p[0])) for s in levels]
    for i, steps, nd, ans, q in _rounds(strata, n):
        out.append(_case(f"gsm8k/test-{i}", "GSM8K", f"{steps} steps, {nd} distractor number(s)", "integer", q, ans))
    rule = ("GSM8K test (1319). Steps = '<<...>>' calculator annotations of the reference solution; distractors = "
            "numbers stated in the question that the reference solution never uses. Pool: >= 1 distractor, integer "
            "answer. One item for each of the 5 largest step counts (multi-step with distractors, spanning the top of "
            "the difficulty range); within a step count the most distractors, then the lowest row index.")
    if n > 5:
        rule += (f" Enlarged ({n}): rounds over the same 5 step counts, next item in the same order, step counts "
                 "with no item left skipped.")
    return out, rule


# ------------------------------------------------------------------------------------------------ MATH-500

MATH_STRATA = [(5, "Number Theory"), (5, "Counting & Probability"), (5, "Intermediate Algebra"), (4, "Geometry"),
               (3, "Algebra")]


def math500(n: int = 5) -> tuple[list[dict], str]:
    rows = [json.loads(x) for x in _get("math500", "test.jsonl").read_text().splitlines() if x.strip()]
    out, strata = [], []
    for lvl, subj in MATH_STRATA:
        pool = [r for r in rows if r["level"] == lvl and r["subject"] == subj and re.fullmatch(r"\d+", r["answer"].strip())]
        strata.append([(lvl, subj, r) for r in sorted(pool, key=lambda r: (len(r["solution"]), r["unique_id"]),
                                                      reverse=True)])
    for lvl, subj, r in _rounds(strata, n):
        out.append(_case(f"math500/{r['unique_id'].removeprefix('test/').removesuffix('.json')}", "MATH",
                         f"level {lvl} {subj}", "integer", r["problem"], r["answer"].strip()))
    rule = ("MATH-500 (500), integer answers only (graded exactly by the integer extractor; \\boxed{} accepted). One "
            "item per stratum " + ", ".join(f"L{l} {s}" for l, s in MATH_STRATA) + " (levels 3-5 across number theory, "
            "probability, algebra, geometry); within a stratum the longest reference solution (most derivation steps).")
    if n > 5:
        rule += (f" Enlarged ({n}): rounds over the same strata, next-longest reference solution; pool sizes "
                 + ", ".join(f"L{l} {s} {len(st)}" for (l, s), st in zip(MATH_STRATA, strata)) + ".")
    return out, rule


# ------------------------------------------------------------------------------------------------ MMLU-Pro

MMLU_STRATA = [("law", "ori_mmlu-professional_law"), ("physics", "scibench"), ("chemistry", "scibench"),
               ("engineering", "stemez"), ("math", "theoremQA")]


def mmlu_pro(n: int = 5) -> tuple[list[dict], str]:
    rows = _parquet("mmlu_pro", "data/test-00000-of-00001.parquet")
    out, used = [], {}
    pools = [[r for r in rows if r["category"] == cat and r["src"].startswith(src) and len(r["options"]) == 10
              and r["answer_index"] >= 4 and len(r["question"]) <= 1200] for cat, src in MMLU_STRATA]
    while len(out) < n and any(pools):
        for (cat, src), pool in zip(MMLU_STRATA, pools):
            if not pool or len(out) == n:
                continue
            # round 0: used.get() is 0 / 1, the same order as "key letter not used yet" of the 30-case rule
            pool.sort(key=lambda r: (used.get(r["answer"], 0), -len(r["question"]), r["question_id"]))
            r = pool.pop(0)
            used[r["answer"]] = used.get(r["answer"], 0) + 1
            out.append(_case(f"mmlu_pro/{r['question_id']}", "MMLU-Pro", f"{cat} ({r['src']})", "choice",
                             r["question"], r["answer"], r["options"]))
    rule = ("MMLU-Pro test (12032). Strata: " + ", ".join(f"{c}/{s}" for c, s in MMLU_STRATA) + " (the hardest "
            "categories / the sources added by MMLU-Pro: scibench, stemez, theoremQA). Pool: all 10 options present, "
            "key among E-J (beyond the 4 MMLU positions), question <= 1200 chars; within a stratum the longest "
            "question, preferring a key letter not used yet.")
    if n > 5:
        rule += (f" Enlarged ({n}): rounds over the same strata, next item in the same order with 'not used yet' "
                 "generalized to 'least used so far' (key letters spread over E-J).")
    return out, rule


# ------------------------------------------------------------------------------------------------ ARC-Challenge

def arc(n: int = 5) -> tuple[list[dict], str]:
    rows = _parquet("arc", "ARC-Challenge/test-00000-of-00001.parquet")
    by_src: dict[str, list[dict]] = {}
    for r in rows:
        by_src.setdefault(r["id"].split("_")[0], []).append(r)
    srcs = sorted(by_src, key=lambda s: (-len(by_src[s]), s))[:5]
    out, used, pools = [], {}, []
    for s in srcs:
        pool = []
        for r in by_src[s]:
            labels = list(r["choices"]["label"])
            if len(labels) < 4 or len(r["question"]) > 700 or any(re.search(r"\d \d", t) for t in r["choices"]["text"]):
                continue
            key = "ABCDE"[labels.index(r["answerKey"])]
            if key == "A":
                continue
            pool.append((r, key))
        pools.append(pool)
    while len(out) < n and any(pools):
        for s, pool in zip(srcs, pools):
            if not pool or len(out) == n:
                continue
            # round 0: used.get() is 0 / 1, the same order as "key letter not used yet" of the 30-case rule
            pool.sort(key=lambda p: (used.get(p[1], 0), -len(p[0]["question"]), p[0]["id"]))
            r, key = pool.pop(0)
            used[key] = used.get(key, 0) + 1
            out.append(_case(f"arc_challenge/{r['id']}", "ARC-Challenge", s, "choice", r["question"], key,
                             r["choices"]["text"]))
    rule = ("ARC-Challenge test (1172; the questions retrieval and co-occurrence baselines miss). One item from each of "
            "the 5 largest exam sources (id prefix: " + ", ".join(srcs) + "), >= 4 options, key not the first option, "
            "question <= 700 chars, no garbled option ('83 331'); within a source the longest question (multi-clause scenarios need the most "
            "reasoning), preferring a key letter not used yet. Numeric labels mapped to A-E.")
    if n > 5:
        rule += (f" Enlarged ({n}): rounds over the same 5 sources, next item in the same order with 'not used yet' "
                 "generalized to 'least used so far' (keys spread over B-E).")
    return out, rule


# ------------------------------------------------------------------------------------------------ BIG-Bench Hard

def _bbh(task: str) -> list[dict]:
    return _parquet("bbh", f"{task}/test-00000-of-00001.parquet")


def _split_options(text: str) -> tuple[str, list[str], list[str]]:
    """'... Options:\\n(A) x\\n(B) y' -> (stem, choices, labels) ; '- Yes\\n- No' bullets -> labels A, B."""
    stem, _, opts = text.partition("\nOptions:\n")
    lab, ch = [], []
    for line in opts.strip().splitlines():
        m = re.match(r"\(([A-Z])\)\s*(.*)", line.strip())
        if m:
            lab.append(m.group(1))
            ch.append(m.group(2))
        elif line.strip().startswith("- "):
            lab.append("ABCDEFGHIJ"[len(lab)])
            ch.append(line.strip()[2:])
    return stem, ch, lab


def bbh(n: int = 5) -> tuple[list[dict], str]:
    rows = [(i, r) for i, r in enumerate(_bbh("multistep_arithmetic_two")) if re.fullmatch(r"\d+", r["target"])]
    strata = [[_case(f"bbh/multistep_arithmetic_two-{i}", "BBH", "multistep_arithmetic_two", "integer",
                     "Evaluate the arithmetic expression: " + r["input"].rstrip("= ").strip(), r["target"])
               for i, r in sorted(rows, key=lambda x: (int(x[1]["target"]), -x[0]), reverse=True)]]
    for task in ("logical_deduction_seven_objects", "tracking_shuffled_objects_seven_objects", "date_understanding",
                 "causal_judgement"):
        pool = []
        for i, r in enumerate(_bbh(task)):
            stem, ch, lab = _split_options(r["input"])
            tgt = r["target"].strip("() ")
            key = tgt if tgt in lab else ("ABCDEFGHIJ"[ch.index(tgt)] if tgt in ch else None)
            if key is None or key == "A" or len(r["input"]) > 2000:
                continue
            pool.append((-len(r["input"]), i, stem, ch, key))
        strata.append([_case(f"bbh/{task}-{i}", "BBH", task, "choice", stem, key, ch)
                       for _, i, stem, ch, key in sorted(pool)])
    out = _rounds(strata, n)
    rule = ("BIG-Bench Hard: one item from each of multistep_arithmetic_two (positive target only: the integer "
            "extractor reads unsigned integers; the largest target), logical_deduction_seven_objects, "
            "tracking_shuffled_objects_seven_objects, date_understanding, causal_judgement (Yes/No as choices A/B). "
            "Choice tasks: key not (A), input <= 2000 chars, the longest input (most constraints / swaps / date "
            "steps / longest vignette); options moved from the input into the choice list.")
    if n > 5:
        rule += (f" Enlarged ({n}): rounds over the same 5 tasks, next item in the same order (next-largest target; "
                 "next-longest input). causal_judgement keeps the 'key not (A)' rule, so all its items have key B "
                 "(No).")
    return out, rule


# ------------------------------------------------------------------------------------------------ LongCtx

FIRST = ["Benedek", "Ilse", "Teodor", "Marisol", "Ansgar", "Philippa", "Corvin", "Liesl", "Osric", "Yelena",
         "Halvard", "Saoirse", "Ludovic", "Annika", "Bartholomew", "Zofia"]
LAST = ["Sorrell", "Varnholt", "Quennell", "Abernathy", "Lindqvist", "Okonkwo-Hale", "Pemberton", "Ruskova",
        "Daventry", "Marchetti", "Holloway", "Engstrom", "Castellane", "Whitlow", "Barrowmere", "Ferrand"]
PLACES = ["Quillmere", "Kestrel Point", "Ashbourne Ferry", "Vellacourt", "Thornwick", "Marrowgate", "Halcyon Reach",
          "Dunmoral", "Saltreach", "Brindlemoor", "Osterholt", "Fennick Cross"]
SHIPS = ["Morrow Lantern", "Grey Petrel", "Constant Hope", "Saint Ivo", "Northern Ember", "Quiet Heron"]

# (target tokens of the whole user turn, design); designs listed in the suite's selection block
LC_PLAN = [(8000, "2hop"), (8000, "3hop"), (16000, "2hop_distract"), (16000, "sum_2x2hop"), (31500, "3hop_reverse_distract")]


def _facts(design: str, rnd: random.Random) -> tuple[list[str], str, int, str]:
    """(fact sentences in insertion order, question, integer answer, short description)."""
    people = [f"{a} {b}" for a, b in zip(rnd.sample(FIRST, 4), rnd.sample(LAST, 4))]
    places = rnd.sample(PLACES, 6)
    ships = rnd.sample(SHIPS, 3)
    codes = rnd.sample(range(1000, 9999), 6)

    def moved(p, t):
        return f"In the spring of that year the archivist {p} quietly moved the family ledgers to the town of {t}."

    def vault(t, n):
        return f"Every clerk in {t} knew that the ledger vault of the town was numbered {n}."

    if design == "2hop":
        return ([moved(people[0], places[0]), vault(places[0], codes[0])],
                f"What is the number of the ledger vault in the town to which the archivist {people[0]} moved the "
                "family ledgers?", codes[0], "2-hop: person -> town -> vault number")
    if design == "3hop":
        f = [f"Captain {people[0]} had been given command of the ship {ships[0]}.",
             f"For the whole winter the {ships[0]} lay berthed at the harbour of {places[0]}.",
             f"The harbour master of {places[0]} stamped every document with permit number {codes[0]}."]
        return (f, f"What permit number does the harbour master of the harbour where the ship commanded by Captain "
                   f"{people[0]} was berthed stamp on documents?", codes[0], "3-hop: captain -> ship -> harbour -> permit")
    if design == "2hop_distract":
        f = [moved(people[0], places[0]), vault(places[0], codes[0]), moved(people[1], places[1]),
             vault(places[1], codes[1]), vault(places[2], codes[2])]
        rnd.shuffle(f)
        return (f, f"What is the number of the ledger vault in the town to which the archivist {people[0]} moved the "
                   "family ledgers?", codes[0], "2-hop with 2 distractor chains (other archivist, unrelated town)")
    if design == "sum_2x2hop":
        f = [moved(people[0], places[0]), vault(places[0], codes[0]), moved(people[1], places[1]),
             vault(places[1], codes[1]), moved(people[2], places[2]), vault(places[2], codes[2])]
        rnd.shuffle(f)
        return (f, f"What is the sum of the ledger vault numbers of the two towns to which the archivists {people[0]} "
                   f"and {people[1]} moved the family ledgers?", codes[0] + codes[1],
                "two 2-hop chains + addition, 1 distractor chain")
    if design == "3hop_reverse_distract":
        f = [f"The harbour master of {places[0]} stamped every document with permit number {codes[0]}.",
             f"The harbour master of {places[1]} stamped every document with permit number {codes[1]}.",
             f"For the whole winter the {ships[0]} lay berthed at the harbour of {places[0]}.",
             f"For the whole winter the {ships[1]} lay berthed at the harbour of {places[1]}.",
             f"Captain {people[1]} had been given command of the ship {ships[1]}.",
             f"Captain {people[0]} had been given command of the ship {ships[0]}."]
        return (f, f"What permit number does the harbour master of the harbour where the ship commanded by Captain "
                   f"{people[0]} was berthed stamp on documents?", codes[0],
                "3-hop, facts in reverse chain order, a parallel distractor chain")
    raise ValueError(design)


def _sentence_cut(text: str, pos: int) -> int:
    j = text.find(". ", pos)
    return len(text) if j < 0 else j + 2


def longctx(n: int = 5) -> tuple[list[dict], str]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    books = pq.read_table(_get("pg19", "data/test-00000-of-00001-29a571947c0b5ccc.parquet"),
                          columns=["short_book_title", "text"]).to_pylist()
    books = [b for b in books if len(b["text"]) > 200_000]
    out, details = [], []
    plan = [(r, t, d) for r in range(-(-n // len(LC_PLAN))) for t, d in LC_PLAN][:n]   # rounds over the 5 designs
    for k, (rnd_i, target, design) in enumerate(plan):
        rnd = random.Random(1000 + k)
        facts, q, ans, desc = _facts(design, rnd)
        book = books[3 + k]                        # one book per case, never the first ones
        text = " ".join(book["text"][20_000:].split())   # skip the Gutenberg header / front matter, flatten lines
        head = ("Read the following document carefully. A few sentences in it state facts needed to answer the "
                "question after it; use only facts stated in the document.\n\n<document>\n")
        tail = f"\n</document>\n\nQuestion: {q}"
        depths = sorted(rnd.uniform(0.08, 0.92) for _ in facts)
        chars = int(target * 4.2)
        for _ in range(4):                          # fit the token count of the whole user turn to the target
            hay = text[:chars]
            doc, prev = [], 0
            for d, f in zip(depths, facts):
                cut = _sentence_cut(hay, int(d * len(hay)))
                doc += [hay[prev:cut], f + " "]
                prev = cut
            doc.append(hay[prev:])
            question = head + "".join(doc) + tail
            ntok = len(tok.encode(question))
            chars = int(chars * target / ntok)
        suffix = f"-r{rnd_i}" if rnd_i else ""      # round 0 keeps the ids of the 30-case suite
        out.append(_case(f"longctx/{design}-{target // 1000}k{suffix}", "LongCtx", f"~{ntok} Qwen3 tokens, {desc}", "integer",
                         question, str(ans)))
        details.append({"id": out[-1]["id"], "book": book["short_book_title"], "qwen3_tokens": ntok,
                        "fact_depths": [round(d, 3) for d in depths], "design": desc})
    rule = ("Synthetic multi-hop needles (invented names, 4-digit codes, seeded random.Random(1000 + case)) inserted "
            "at sentence boundaries of PG19-test books (one book per case, text after the first 20k characters, "
            "whitespace flattened) at seeded depths in [0.08, 0.92]; prompt sized to ~8k / 8k / 16k / 16k / 31.5k "
            "Qwen3 tokens (other tokenizers +-10 %). Designs grow in difficulty: 2-hop, 3-hop, 2-hop with "
            "distractor chains, two 2-hop chains + addition, 3-hop in reverse chain order with a parallel distractor "
            "chain. Integer answers.")
    if n > 5:
        rule += (f" Enlarged ({n}): rounds over the same 5 designs and lengths (case k = 5 * round + design index, "
                 "seed random.Random(1000 + k), book 3 + k: fresh names, codes, depths and book per case; ids of rounds "
                 ">= 1 end in -r<round>).")
    return out, rule + " Details: " + json.dumps(details)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-subset", type=int, default=5,
                    help="items per subset (5: the 30-case suite_hard.json; 20: the enlarged suite_hard20.json)")
    ap.add_argument("--out", help="default experiments/eval/suite_hard.json (5) / suite_hard<N>.json")
    a = ap.parse_args()
    n = a.per_subset
    if n < 5:
        raise SystemExit("--per-subset must be >= 5 (the 30-case suite is round 0)")
    a.out = a.out or str(REPO / "experiments" / "eval" / ("suite_hard.json" if n == 5 else f"suite_hard{n}.json"))
    cases, selection = [], {}
    for name, fn in (("GSM8K", gsm8k), ("MATH", math500), ("MMLU-Pro", mmlu_pro), ("ARC-Challenge", arc),
                     ("BBH", bbh), ("LongCtx", longctx)):
        got, rule = fn(n)
        assert len(got) == n, (name, len(got))
        cases += got
        selection[name] = {"rule": rule, "ids": [c["id"] for c in got], "max_tokens": BUDGET[name]}
    suite = {
        "suite": "suc_hard", "version": 1,
        "description": ("SUC hard suite: 30 automatically graded items, 5 per subset (GSM8K, MATH-500, MMLU-Pro, "
                        "ARC-Challenge, BIG-Bench Hard, long-context multi-hop needles at ~8k-32k tokens), selected "
                        "deterministically to discriminate between compressed models where the core suite saturates. "
                        "Thinking off (every case think=false; the DeepSeek-V4, GLM-5.3 and Qwen3.8 chat templates all "
                        "support a non-thinking mode), ds4-style 'Answer:' final line, greedy. 5 items per subset: a "
                        "per-question comparison of paired systems, not a benchmark score. Built by "
                        "scripts/build_hard_suite.py."),
        "attribution": "Items from the pinned public datasets below; prompt format and answer extraction follow "
                       "ds4-eval (MIT, The ds4.c authors).",
        "licenses": {k: f"{v[0]}@{v[1]}: {v[2]}" for k, v in PINS.items()},
        "disjointness": "No item comes from a calibration / validation source of suc.data.sources (GSM8K is used "
                        "there as train split only; MATH, MMLU-Pro, ARC, BBH, PG19 are not used).",
        "selection": selection,
        "system": None,
        "cases": cases,
    }
    if n != 5:
        suite["suite"] = f"suc_hard{n}"
        suite["description"] = (
            f"SUC hard suite, enlarged: {6 * n} automatically graded items, {n} per subset (GSM8K, MATH-500, MMLU-Pro, "
            "ARC-Challenge, BIG-Bench Hard, long-context multi-hop needles at ~8k-32k tokens), selected "
            "deterministically by the rules of the 30-case suite_hard.json applied in rounds (round 0 = its 30 "
            "items, same ids and prompts). Thinking off (every case think=false), ds4-style 'Answer:' final line, "
            f"greedy. {n} items per subset: still a paired per-question comparison of systems; read differences with "
            "the McNemar counts of scripts/hard_suite_report.py. Built by scripts/build_hard_suite.py --per-subset "
            f"{n}.")
        suite["subset_of"] = {"experiments/eval/suite_hard.json": "its 30 case ids and prompts are round 0 of every "
                                                                 "subset here (byte-identical case records)"}
    Path(a.out).write_text(json.dumps(suite, indent=1, ensure_ascii=False) + "\n")
    from suc.evaluation.suite import load_suite

    s = load_suite(a.out)
    print(a.out, len(s["cases"]), "cases, sha256", s["sha256"])
    for k, v in selection.items():
        print(k, v["ids"])


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(REPO))
    main()
