"""Chat rendering for tokenizers without a chat_template (DeepSeek-V4 ships encoding/encoding_*.py instead)."""
from suc.data.sources import SourceSpec, render


class _NoTemplateTokenizer:
    chat_template = None

    def __init__(self, path):
        self.name_or_path = str(path)

    def apply_chat_template(self, *a, **k):
        raise ValueError("Cannot use chat template functions because tokenizer.chat_template is not set")


def _spec():
    return SourceSpec(name="t", domain="instruction", repo_id="x/y", filename="f", revision="r",
                      format="jsonl", text_fields=("q", "a"), template="chat")


def test_chat_uses_model_encoding_module(tmp_path):
    enc = tmp_path / "encoding"
    enc.mkdir()
    (enc / "encoding_dsv4.py").write_text(
        "def encode_messages(messages, thinking_mode, add_default_bos_token=True, **kw):\n"
        "    return '<B>' + '|'.join(f\"{m['role']}:{m['content']}\" for m in messages) + ':' + thinking_mode\n")
    out = render(_spec(), {"q": "hi", "a": "yo"}, _NoTemplateTokenizer(tmp_path))
    assert out == "<B>user:hi|assistant:yo:chat"


def test_chat_without_template_or_encoding_raises(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="chat_template"):
        render(_spec(), {"q": "hi", "a": "yo"}, _NoTemplateTokenizer(tmp_path))
