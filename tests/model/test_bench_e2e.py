import json

import pytest
import yaml

from suc.cli.bench import run
from suc.provenance import REQUIRED_METADATA_KEYS


@pytest.mark.model
def test_bench_end_to_end_on_dev_model(tmp_path):
    cfg = {
        "name": "e2e_dev",
        "output_dir": str(tmp_path / "results"),
        "model": {"id": "Qwen/Qwen3-0.6B"},
        "dataset": {"seq_len": 512, "max_tokens": 2048},
        "representations": [{"kind": "bf16"}, {"kind": "mlx_affine", "bits": 4}],
        "bench": {"prompt_lengths": [32], "decode_tokens": 8, "warmup": 0, "repeats": 1},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))

    result = json.loads(run(path).read_text())

    assert set(REQUIRED_METADATA_KEYS) <= set(result["metadata"])
    assert len(result["metadata"]["model_revision"]) == 40  # resolved commit, not "main"
    bf16, int4 = result["representations"]
    assert bf16["identity_check"]["exact"]
    assert bf16["size"]["bpp"] == pytest.approx(16.0)
    assert int4["size"]["bpp"] == pytest.approx(4.5, abs=0.01)
    assert int4["perplexity"]["ppl"] > bf16["perplexity"]["ppl"]
    assert int4["kl_vs_reference"]["kl_mean"] > 0.0
