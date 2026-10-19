#!/usr/bin/env python3
"""Paired C++ KL against existing BF16 references, on their exact validation rows.

No original checkpoint is loaded. All rows are scored, excluding the final
position as in the compression report. Results are qualification, never tuning.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import mlx.core as mx
import numpy as np
import frontier_measure as fm
from efficiency_compare_m1 import local_model, SOURCE, FLAGS
from suc.analysis.sensitivity import logit_divergence
from suc.evaluation.reconstruct import aggregate_kl
from suc.evaluation.streamed import RefLogits

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=['q8', 'q27', 'q30', 'q38', 'ds', 'glm'], required=True)
    ap.add_argument('--chunk', type=int, help='Explicit matched chunk, e.g. a service profile; both binaries must honor it')
    ap.add_argument('--profile', default='', help='Separate evidence name for a distinct service profile')
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--tcq-reuse', action='store_true')
    ap.add_argument('--auto-default', action='store_true', help='Verify automatic candidate dispatch, without a reuse override')
    ap.add_argument('--rows', type=int, help='Explicit smoke subset; never presented as full qualification')
    a = ap.parse_args()
    model = local_model(a.model)
    run = SOURCE / model.relative_to(SOURCE).parts[0]
    manifest = json.loads((run/'reference/manifest.json').read_text())
    ref_path = run/'reference/ref_logits.safetensors'
    assert sha(ref_path) == manifest['outputs']['ref_logits.safetensors']['sha256']
    data = run/'calibrate/data.safetensors'
    assert sha(data) == manifest['inputs']['calibrate']['sha256']
    cfg = json.loads((run/'reference/reference.json').read_text())
    tokens = mx.load(str(data))['validation'][:cfg['kl_sequences']]
    ref = RefLogits(ref_path)
    assert tuple(tokens.shape) == tuple(ref.shape[:2])
    assert tokens.shape[1] == cfg['kl_seq_len']
    if a.rows is not None:
        assert 0 < a.rows <= tokens.shape[0]
        tokens = tokens[:a.rows]
    token_hash = hashlib.sha256(np.array(tokens).astype('<i4').tobytes()).hexdigest()
    results = {}; hashes = {}; env = dict(os.environ)
    for key in FLAGS:
        env.pop(key, None)
    env['SUC_NGRAM_THREADS'] = '1'
    # Historical CLI treats --chunk 64 as an automatic sentinel (512 on these
    # rows). Use an unambiguous explicit value and assert the effective value.
    # GLM's historical CLI also clamps oversized chunks to the prompt length.
    # A full validation row is unambiguous on all three families.
    chunk = a.chunk or int(tokens.shape[1])
    for label, build in [('baseline', a.baseline.expanduser()), ('candidate', a.candidate.expanduser())]:
        parts = []; records = []; hashes[label] = []
        current_env = dict(env)
        current_env['SUC_TCQ_REUSE'] = '1' if label == 'candidate' and a.tcq_reuse else '0'
        if label == 'candidate' and a.auto_default:
            current_env.pop('SUC_TCQ_REUSE', None)
        for i in range(tokens.shape[0]):
            row = [int(x) for x in np.array(tokens[i])]
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); prompt = root/'tokens.txt'; output = root/'logits.npy'
                prompt.write_text(','.join(map(str, row)))
                cmd = [str(build/'suc1-generate'), str(model), '--tokens-file', str(prompt),
                       '--steps', '1', '--prefill', 'batch', '--chunk', str(chunk),
                       '--logits-out', str(output), '--json', str(root/'run.json'), '--quiet']
                start = time.monotonic()
                proc = subprocess.run(cmd, env=current_env, capture_output=True, text=True)
                if proc.returncode:
                    raise RuntimeError(proc.stderr[-3000:])
                logits = np.load(output, mmap_mode='r')
                assert logits.shape[0] >= len(row) and logits.shape[1] == ref.shape[-1]
                # Hash raw C++ values; BF16 cast matches the established reference protocol.
                hashes[label].append(hashlib.sha256(logits[:len(row)].tobytes()).hexdigest())
                r = ref.rows(i, i+1)
                got = mx.array(np.asarray(logits[:len(row)]))[None].astype(r.dtype)
                score = logit_divergence(r[:, :-1], got[:, :-1], tokens[i:i+1, 1:], return_tokens=True)
                parts.append((score, len(row)-1))
                record = json.loads((root/'run.json').read_text())
                assert record['chunk'] == chunk, (label, record['chunk'], chunk)
                records.append({'row': i, 'wall_s': time.monotonic()-start, 'run': record})
                print('REFERENCE_KL', a.model, label, i+1, '/', tokens.shape[0],
                      'mean', score['kl_mean'], flush=True)
                del logits, got, r
                mx.clear_cache()
        results[label] = {'kl': aggregate_kl(parts), 'records': records,
                          'binary_commit': (build/'GIT_COMMIT').read_text().strip()}
    os.environ['SUC_BENCH_BUILD_COMMIT'] = results['candidate']['binary_commit']
    name = 'eff_reference_kl_'+('auto_smoke_' if a.auto_default and a.rows else '')+(a.profile+'_' if a.profile else '')+a.model
    fm._write(name, str(model), token_hash,
              {'reference_sha256': manifest['outputs']['ref_logits.safetensors']['sha256'],
               'reference_manifest': manifest, 'chunk': chunk, 'tcq_reuse': a.tcq_reuse,
               'sequences': int(tokens.shape[0]), 'sequence_length': int(tokens.shape[1]),
               'smoke_subset': a.rows is not None, 'automatic_default': a.auto_default,
               'precision': 'BF16 logits, canonical float32 divergence; global token p95'},
              {'systems': results, 'raw_logit_hashes': hashes,
               'all_rows_bit_identical': hashes['baseline'] == hashes['candidate']})

if __name__ == '__main__':
    main()
