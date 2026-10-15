// Full next-token logits of fixed token rows through llama.cpp (libllama), for the KL of GGUF files on exactly the
// token ids of a `suc compress` run (scripts/gguf_llama.py kl; the llama.cpp counterpart of ds4_score.c rows).
//
//   llama_score MODEL.gguf TOKENS.i32 N L OUT.bf16 [--n-gpu-layers K]
//
// TOKENS.i32: N rows of L little-endian int32 token ids. Every row is decoded from an empty context in ONE batch of L
// tokens (logits for every position); OUT.bf16 receives, per row, the logits of positions 0 .. L-2 (those predicting
// tokens 1 .. L-1) as bfloat16 (round to nearest even), shape (N, L-1, n_vocab). One JSON line per row on stdout:
// {"row", "tokens", "nll_sum", "top1_hits", "seconds", "vocab"}; NLL from the fp32 logits (log-sum-exp in double).
#include "llama.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void die(const char *m) { fprintf(stderr, "llama_score: %s\n", m); exit(1); }

static uint16_t to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);   // NaN stays NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static void quiet_log(enum ggml_log_level level, const char *text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

int main(int argc, char **argv) {
    if (argc < 6) die("usage: llama_score MODEL.gguf TOKENS.i32 N L OUT.bf16 [--n-gpu-layers K]");
    const char *model_path = argv[1], *tok_path = argv[2], *out_path = argv[5];
    const int n_rows = atoi(argv[3]), L = atoi(argv[4]);
    int ngl = 999;
    for (int i = 6; i < argc; i++)
        if (!strcmp(argv[i], "--n-gpu-layers") && i + 1 < argc) ngl = atoi(argv[++i]);
    if (n_rows < 1 || L < 2) die("N >= 1 and L >= 2 required");

    std::vector<int32_t> toks((size_t)n_rows * L);
    FILE *f = fopen(tok_path, "rb");
    if (!f || fread(toks.data(), 4, toks.size(), f) != toks.size()) die("cannot read the token file");
    fclose(f);

    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) die("cannot load the model");
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = L;
    cp.n_batch = L;
    cp.n_ubatch = L;
    cp.n_seq_max = 1;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) die("cannot create the context");

    FILE *out = fopen(out_path, "wb");
    if (!out) die("cannot open the output file");
    std::vector<uint16_t> row_bf16((size_t)(L - 1) * n_vocab);
    llama_batch batch = llama_batch_init(L, 0, 1);
    for (int r = 0; r < n_rows; r++) {
        auto t0 = std::chrono::steady_clock::now();
        llama_memory_clear(llama_get_memory(ctx), true);
        const int32_t *row = toks.data() + (size_t)r * L;
        batch.n_tokens = L;
        for (int i = 0; i < L; i++) {
            if (row[i] < 0 || row[i] >= n_vocab) die("token id outside the GGUF vocabulary");
            batch.token[i] = row[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        if (llama_decode(ctx, batch) != 0) die("llama_decode failed");
        double nll = 0.0;
        long hits = 0;
        for (int i = 0; i < L - 1; i++) {
            const float *lg = llama_get_logits_ith(ctx, i);
            if (!lg) die("no logits");
            float mx = lg[0];
            int arg = 0;
            for (int v = 1; v < n_vocab; v++)
                if (lg[v] > mx) { mx = lg[v]; arg = v; }
            double s = 0.0;
            for (int v = 0; v < n_vocab; v++) s += exp((double)lg[v] - mx);
            nll += (mx + log(s)) - lg[row[i + 1]];
            hits += arg == row[i + 1];
            uint16_t *dst = row_bf16.data() + (size_t)i * n_vocab;
            for (int v = 0; v < n_vocab; v++) dst[v] = to_bf16(lg[v]);
        }
        if (fwrite(row_bf16.data(), 2, row_bf16.size(), out) != row_bf16.size()) die("write failed");
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("{\"row\": %d, \"tokens\": %d, \"nll_sum\": %.9f, \"top1_hits\": %ld, \"seconds\": %.4f, \"vocab\": %d}\n",
               r, L - 1, nll, hits, sec, n_vocab);
        fflush(stdout);
    }
    fclose(out);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
