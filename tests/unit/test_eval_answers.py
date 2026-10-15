"""Answer extractors of `suc eval` (suc.evaluation.answers). The cases are the ds4-eval extractor self-tests
(ds4_eval.c run_extractor_self_tests, MIT, The ds4.c authors) plus SUC additions (\\boxed{}, raw-prompt fallbacks)."""
import pytest

from suc.evaluation.answers import extract_answer, grade, line_spec_matches, visible_text

TEN = list("ABCDEFGHIJ")


@pytest.mark.parametrize("text,expected", [
    ("</think>So answer is 0.716 H+/O2. That corresponds to option F.\nThus final answer: F.</think>The visible "
     "explanation repeats the calculation.\nAnswer: F", "F"),
    ("</think>Answer: F\nThis answer is final; option H is a tempting distractor.", "F"),
    ("</think>The answer is F. This answer is final; option H is tempting.", "F"),
    ("</think>Answer: I think it is C", "C"),
    ("</think>Answer: I'll go with C.", "C"),
    ("</think>Answer: A careful reading shows C.", "C"),
    ("</think>Answer: It is not B, the answer is D", "D"),
    ("</think>Answer: rules out C, leaving D", "D"),
    ("</think>Answer: It isn't B, so D", "D"),
    ("</think>Answer: D, not B", "D"),
])
def test_choice_extractor_ds4_cases(text, expected):
    assert extract_answer("choice", text, nchoices=10) == expected


def test_choice_genuine_standalone_i():
    assert extract_answer("choice", "</think>Answer: I.", nchoices=10) == "I"


def test_choice_out_of_range_pronoun_harmless_with_four_choices():
    assert extract_answer("choice", "</think>Answer: I think it is D", nchoices=4) == "D"


def test_choice_no_letter():
    assert extract_answer("choice", "nothing here 123", nchoices=4) == "?"


@pytest.mark.parametrize("text,expected", [
    ("</think>I first thought the answer was 80.\nFinal answer: 082", "82"),
    ("</think>The answer is 082. This answer comes from AIME 2025.", "82"),
    ("</think>Answer: m+n = 256+37 = 293", "293"),
    ("</think>Answer: a+b+c = 12+25+25 = 62", "62"),
    ("</think>Answer: 2*7 + 3*6 + 5*4 + 7*5 = 81", "81"),
    ("</think>Answer: 82\nThe value 2025 is just the year.", "82"),
])
def test_integer_extractor_ds4_cases(text, expected):
    assert extract_answer("integer", text) == expected


@pytest.mark.parametrize("text,expected", [
    # Qwen3 style without an Answer: line (raw QA prompts): the last \boxed{} wins
    ("<think>17*23 = 391?</think>\n\n$$17 \\times 23 = 391$$\n**Risposta finale:** $\\boxed{391}$ (check 17*20)", "391"),
    ("Il treno percorre $\\boxed{200}$ chilometri.", "200"),
    ("**Risposta finale:** 29 mele.", "29"),
    ("Thus 1,024 is the result.", "1024"),                 # thousands separator inside the last number
    ("no digits at all", "?"),
])
def test_integer_extractor_suc_additions(text, expected):
    assert extract_answer("integer", text) == expected


def test_integer_boxed_on_the_answer_line_wins():
    # seen on Qwen3-0.6B (suc eval smoke run): the first integer of the line is part of the restated question
    assert extract_answer("integer", "**Final Answer:** 15% of 240 is \\boxed{36}.") == "36"


def test_integer_strict_marker_beats_boxed():
    assert extract_answer("integer", "\\boxed{12}\nAnswer: 13") == "13"


def test_lines_extractor_and_matching():
    text = ("</think>I think the answer should be line 10, because CWE-122 may apply.\n**Answer:** 10</think>"
            "The primary write is at line 10.\nAnswer: 10")
    assert extract_answer("lines", text) == "10"
    assert extract_answer("lines", "Answer: lines 9 and 15") == "9,15"
    assert extract_answer("lines", "Answer: 20 - 22") == "20-22"
    assert line_spec_matches("9-10", "10") and line_spec_matches("17-20", "18,19")
    assert not line_spec_matches("9-10", "10,11") and not line_spec_matches("11", "?")
    assert line_spec_matches("0", "0")


def test_visible_text_after_first_think_close():
    assert visible_text("<think>a</think>b</think>c") == "b</think>c"
    assert visible_text("plain") == "plain"


def test_grade():
    assert grade({"kind": "integer", "answer": "082"}, "Answer: 82") == ("82", True)
    assert grade({"kind": "choice", "answer": "B", "choices": ["x", "y", "z", "w"]}, "Answer: C") == ("C", False)
    assert grade({"kind": "open"}, "anything") == (None, None)
