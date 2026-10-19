"""Experimental efficiency paths: actual Metal graph and HTTP cache regressions."""
import json
import os
import subprocess
from pathlib import Path

import httpx
import numpy as np
import pytest
from tests.unit.test_cpp_generate import ARCH, _tiny_file
from tests.unit.test_cpp_server import _start, _stop, _tokenizer

BUILD=Path(os.environ.get("SUC_CPP_BUILD",Path.home()/"suc-local/cpp-build"))
pytestmark=pytest.mark.skipif(not (BUILD/"suc1-server").exists(),reason="Metal test build required")

@pytest.fixture(scope="module")
def fixture_model(tmp_path_factory):
    root=tmp_path_factory.mktemp("efficiency")
    arch=dict(ARCH,num_attention_heads=2,num_key_value_heads=1,head_dim=64)
    model=_tiny_file(root/"tiny.suc1",False,arch=arch)
    return root,model,_tokenizer(root/"tokenizer.json",arch["vocab_size"])

@pytest.mark.parametrize("rows",[3,64])
def test_query_tile_logits(fixture_model,tmp_path,rows):
    _,model,tok=fixture_model
    outputs=[]
    for enabled in ["0","1"]:
        path=tmp_path/(enabled+".npy")
        cmd=[str(BUILD/"suc1-generate"),str(model),"--tokenizer",str(tok),"--tokens",",".join(str(i%200+1) for i in range(rows)),"--steps","2","--prefill","batch","--logits-out",str(path),"--quiet"]
        result=subprocess.run(cmd,env={**os.environ,"SUC_PREFILL_QUERY_TILE":enabled},capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        outputs.append(np.load(path))
    assert np.array_equal(*outputs),np.max(np.abs(outputs[0]-outputs[1]))

def test_live_prefix_http(fixture_model,tmp_path):
    _,model,tok=fixture_model
    process,log,s=_start(model,tok,tmp_path,"--sync","--prefill","token")
    def ask(prompt):
        response=httpx.post(s["url"]+"/v1/completions",json={"prompt":prompt,"max_tokens":1,"temperature":0},timeout=60)
        assert response.status_code==200,response.text
        return response.json()
    try:
        first=ask([5,17,300]);assert first["usage"]["prompt_tokens_details"]["cached_tokens"]==0
        reused=ask([5,17,300,7]);assert reused["usage"]["prompt_tokens_details"]["cached_tokens"]==3
        assert reused["usage"]["prompt_tokens"]==4
        divergent=ask([6,17,300]);assert divergent["usage"]["prompt_tokens_details"]["cached_tokens"]==0
        fresh=ask([5,17,300,7]);assert fresh["usage"]["prompt_tokens_details"]["cached_tokens"]==0
        assert reused["choices"]==fresh["choices"]
        shorter=ask([5,17]);assert shorter["usage"]["prompt_tokens_details"]["cached_tokens"]==0
    finally:_stop(process,log)

def test_lookup_verification_metadata(fixture_model,tmp_path):
    _,model,tok=fixture_model
    output=tmp_path/"lookup.json"
    result=subprocess.run([str(BUILD/"suc1-generate"),str(model),"--tokens","5,17,300,5,17,300","--steps","16","--lookup-window","4","--json",str(output),"--quiet"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    data=json.loads(output.read_text());assert data["lookup_verification"]["window"]==4
    assert data["lookup_verification"]["batches"]>0
    assert len(data["generated"])==16

# Synthetic fixtures, not additional real candidate models.
from tests.unit.test_cpp_tcq import ds4
from tests.unit.test_cpp_qwen4 import q4_files, _cpp as q4_run

@pytest.mark.xfail(strict=True,reason="Rejected half prototype: measured KL exceeds the predeclared quality budget")
def test_tcq_half_prefill_quality(ds4,tmp_path,monkeypatch):
    d=ds4["d"];arrays=[]
    monkeypatch.setenv("SUC_TILE_BT","8")
    for mode in ["0","1"]:
        monkeypatch.setenv("SUC_TCQ_PREFILL_HALF",mode)
        path=tmp_path/(mode+".npy")
        d._cpp(ds4["native"],tmp_path,"--prefill","batch","--chunk","16","--logits-out",str(path),tokens=d.LONG_PROMPT,steps=1)
        arrays.append(np.load(path).astype(np.float64))
    def logprob(x):
        x=x-x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
    a,b=map(logprob,arrays);kl=(np.exp(a)*(a-b)).sum(-1)
    print("TCQ half KL",float(kl.mean()),float(kl.max()))
    assert kl.mean()<.005 and kl.max()<.1

@pytest.mark.parametrize("variant",["raw","row4"])
def test_ngram_workers_identical(q4_files,tmp_path,monkeypatch,variant):
    arrays=[];tokens=[]
    for threads in ["1","4"]:
        monkeypatch.setenv("SUC_NGRAM_THREADS",threads)
        path=tmp_path/(threads+".npy")
        generated,record=q4_run(q4_files[variant],tmp_path,"--prefill","token","--logits-out",str(path),steps=8)
        assert record["ngram_cpu_workers"]==int(threads)
        assert record["ngram_head_tasks"]>=4
        arrays.append(np.load(path));tokens.append(generated)
    assert tokens[0]==tokens[1]
    assert np.array_equal(*arrays)


def test_tcq_reuse_bit_identical(ds4,tmp_path,monkeypatch):
    d=ds4["d"];arrays=[];tokens=[]
    monkeypatch.setenv("SUC_TILE_BT","16")
    monkeypatch.setenv("SUC_TCQ_PREFILL_HALF","0")
    for mode in ["0","1"]:
        monkeypatch.setenv("SUC_TCQ_REUSE",mode)
        path=tmp_path/("reuse"+mode+".npy")
        generated,_,_=d._cpp(ds4["native"],tmp_path,"--prefill","batch","--chunk","64","--logits-out",str(path),tokens=d.LONG_PROMPT,steps=4)
        arrays.append(np.load(path));tokens.append(generated)
    assert tokens[0]==tokens[1]
    assert np.array_equal(*arrays),float(np.max(np.abs(arrays[0]-arrays[1])))
