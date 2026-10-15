/* ds4_score: teacher-forced scoring and greedy generation of community GGUF files through the ds4 engine
 * (https://github.com/antirez/ds4), for the "Community GGUF comparison" of docs/compression_report.md.
 *
 * Built against an existing ds4 checkout (its core objects), see scripts/gguf/build_ds4_score.sh:
 *
 *   ds4_score rows MODEL.gguf TOKENS.i32 N_ROWS ROW_LEN OUT.jsonl [options]
 *       TOKENS.i32 = N_ROWS x ROW_LEN little-endian int32 token ids (the exact ids of suc compress: HF tokenizer).
 *       Every row starts from an empty context (non-overlapping windows, as suc.evaluation.streamed): the first token
 *       is synced, then the row is decoded token by token, and after every input position p < ROW_LEN - 1 the full
 *       next-token logits are read: NLL of token p + 1 (log-sum-exp in double) and argmax hit are accumulated.
 *       One JSON line per finished row is appended to OUT.jsonl (resumable: rows already there are skipped).
 *       --logits-out FILE  also stores the logits of positions 0 .. ROW_LEN - 2 of every row as bfloat16
 *                          (round to nearest even, as mlx astype), row-major [row][pos][vocab], written in place
 *                          (pwrite) so a resumed run fills the missing rows; scored against the reference logits
 *                          by scripts/gguf_compare.py with suc's own KL code.
 *       --batch B          B rows decoded together through ds4_sessions_eval_batch (native Metal batching for
 *                          DeepSeek-V4 / GLM without SSD streaming; other cases fall back to sequential sessions).
 *       --max-rows N       stop after N rows (counting rows already done).
 *   ds4_score serve MODEL.gguf [options]
 *       greedy generation server on stdin / stdout, one request per line:
 *         "GEN <max_new> <n_stop> <stop ids...> <n_prompt> <prompt ids...>"
 *       answer: "OK <stopped 0|1> <prefill_s> <n> <tokens...> <decode ms per token...>"  (stop token not included)
 *   ds4_score tokenize MODEL.gguf TEXT_FILE [options]  ds4's own tokenization of a text (vocabulary check)
 *
 * Common options: --ctx N (default 4096 + 64), --ssd-streaming, --ssd-streaming-cache-experts N|<n>GB.
 */
#include "ds4.h"
#include "ds4_ssd.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define MAX_BATCH 16

static void die(const char *msg) {
    fprintf(stderr, "ds4_score: %s\n", msg);
    exit(1);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);   /* NaN stays NaN */
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

typedef struct {
    int ctx;
    bool ssd;
    uint32_t cache_experts;
    uint64_t cache_bytes;
    int batch;
    int max_rows;
    const char *logits_out;
} options;

static int parse_opts(int argc, char **argv, int first, options *o) {
    for (int i = first; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--ctx") && i + 1 < argc) o->ctx = atoi(argv[++i]);
        else if (!strcmp(a, "--ssd-streaming")) o->ssd = true;
        else if (!strcmp(a, "--ssd-streaming-cache-experts") && i + 1 < argc) {
            if (!ds4_parse_streaming_cache_experts_arg(argv[++i], &o->cache_experts, &o->cache_bytes))
                die("bad --ssd-streaming-cache-experts");
        } else if (!strcmp(a, "--batch") && i + 1 < argc) o->batch = atoi(argv[++i]);
        else if (!strcmp(a, "--max-rows") && i + 1 < argc) o->max_rows = atoi(argv[++i]);
        else if (!strcmp(a, "--logits-out") && i + 1 < argc) o->logits_out = argv[++i];
        else {
            fprintf(stderr, "ds4_score: unknown option %s\n", a);
            return 1;
        }
    }
    if (o->batch < 1 || o->batch > MAX_BATCH) die("--batch must be 1..16");
    return 0;
}

static ds4_engine *open_engine(const char *path, const options *o, int sessions) {
    ds4_engine_options opt = {
        .model_path = path,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 0,
        .context_size = o->ctx,
        .placement_ctx_hint = o->ctx,
        .placement_session_count_hint = sessions,
        .ssd_streaming = o->ssd,
        .ssd_streaming_cache_experts = o->cache_experts,
        .ssd_streaming_cache_bytes = o->cache_bytes,
    };
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) die("failed to open model");
    return e;
}

/* ------------------------------------------------------------------------------------------------ rows */

static bool *done_rows(const char *out_path, int n_rows) {
    bool *done = calloc((size_t)n_rows, 1);
    FILE *f = fopen(out_path, "rb");
    if (!f) return done;
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        int r;
        if (sscanf(line, "{\"row\": %d", &r) == 1 && r >= 0 && r < n_rows) done[r] = true;
    }
    fclose(f);
    return done;
}

typedef struct {
    ds4_session *s;
    int row;
    int pos;        /* input positions consumed (logits currently predict token pos) */
    double nll;
    long hits;
    double t0;
} row_state;

static int cmd_rows(int argc, char **argv) {
    if (argc < 7) die("usage: rows MODEL TOKENS N_ROWS ROW_LEN OUT [options]");
    options o = {.ctx = 0, .batch = 1};
    const char *model = argv[2], *tok_path = argv[3], *out_path = argv[6];
    const int n_rows = atoi(argv[4]), row_len = atoi(argv[5]);
    if (parse_opts(argc, argv, 7, &o)) return 2;
    if (o.ctx <= 0) o.ctx = row_len + 64;

    FILE *tf = fopen(tok_path, "rb");
    if (!tf) die("cannot open tokens");
    int32_t *toks = malloc((size_t)n_rows * row_len * sizeof(int32_t));
    if (fread(toks, sizeof(int32_t), (size_t)n_rows * row_len, tf) != (size_t)n_rows * row_len)
        die("tokens file shorter than N_ROWS x ROW_LEN");
    fclose(tf);

    bool *done = done_rows(out_path, n_rows);
    int n_done = 0;
    for (int r = 0; r < n_rows; r++) n_done += done[r];
    int limit = o.max_rows > 0 ? o.max_rows : n_rows;
    int todo[n_rows > 0 ? n_rows : 1], n_todo = 0;
    for (int r = 0; r < n_rows && n_done + n_todo < limit; r++)
        if (!done[r]) todo[n_todo++] = r;
    fprintf(stderr, "ds4_score: %d rows done, %d to do (batch %d, ctx %d)\n", n_done, n_todo, o.batch, o.ctx);
    if (n_todo == 0) return 0;

    ds4_engine *e = open_engine(model, &o, o.batch);
    const int vocab = ds4_engine_vocab_size(e);
    int lfd = -1;
    if (o.logits_out) {
        lfd = open(o.logits_out, O_RDWR | O_CREAT, 0644);
        if (lfd < 0) die("cannot open --logits-out");
        const off_t need = (off_t)n_rows * (row_len - 1) * vocab * 2;
        struct stat st;
        if (fstat(lfd, &st) == 0 && st.st_size < need && ftruncate(lfd, need) != 0) die("ftruncate failed");
    }
    FILE *out = fopen(out_path, "ab");
    if (!out) die("cannot open output");
    float *logits = malloc((size_t)vocab * sizeof(float));
    uint16_t *bf = o.logits_out ? malloc((size_t)vocab * 2) : NULL;
    char err[256];

    row_state st[MAX_BATCH];
    int B = o.batch;
    for (int k = 0; k < B; k++) {
        memset(&st[k], 0, sizeof(st[k]));
        if (ds4_session_create(&st[k].s, e, o.ctx) != 0) die("session create failed");
        st[k].row = -1;
    }
    int next = 0, active = 0;
    double t_start = now_s();
    long scored = 0;
    for (;;) {
        /* (re)fill idle slots with new rows: sync the first token */
        for (int k = 0; k < B; k++) {
            if (st[k].row >= 0 || next >= n_todo) continue;
            st[k].row = todo[next++];
            st[k].pos = 1;
            st[k].nll = 0.0;
            st[k].hits = 0;
            st[k].t0 = now_s();
            ds4_tokens p = {0};
            ds4_tokens_push(&p, toks[(size_t)st[k].row * row_len]);
            ds4_session_invalidate(st[k].s);
            if (ds4_session_sync(st[k].s, &p, err, sizeof(err)) != 0) {
                fprintf(stderr, "sync failed: %s\n", err);
                return 1;
            }
            ds4_tokens_free(&p);
            active++;
        }
        if (active == 0) break;
        /* consume the logits that predict token `pos` of every active row */
        for (int k = 0; k < B; k++) {
            if (st[k].row < 0) continue;
            const int32_t *t = toks + (size_t)st[k].row * row_len;
            if (ds4_session_copy_logits(st[k].s, logits, vocab) != vocab) die("copy logits failed");
            float mx = -INFINITY;
            int am = 0;
            for (int i = 0; i < vocab; i++)
                if (logits[i] > mx) { mx = logits[i]; am = i; }
            double sum = 0.0;
            for (int i = 0; i < vocab; i++) sum += exp((double)logits[i] - (double)mx);
            const int target = t[st[k].pos];
            if (target < 0 || target >= vocab || !isfinite(logits[target])) die("target outside the vocabulary");
            st[k].nll += (double)mx + log(sum) - (double)logits[target];
            st[k].hits += am == target;
            if (lfd >= 0) {
                for (int i = 0; i < vocab; i++) bf[i] = f32_to_bf16(logits[i]);
                const off_t off = (((off_t)st[k].row * (row_len - 1)) + (st[k].pos - 1)) * vocab * 2;
                if (pwrite(lfd, bf, (size_t)vocab * 2, off) != (ssize_t)vocab * 2) die("pwrite failed");
            }
            scored++;
            if (st[k].pos == row_len - 1) {          /* row finished */
                const double sec = now_s() - st[k].t0;
                fprintf(out, "{\"row\": %d, \"nll_sum\": %.10f, \"tokens\": %d, \"top1_hits\": %ld, \"seconds\": %.3f, "
                             "\"batch\": %d, \"vocab\": %d}\n",
                        st[k].row, st[k].nll, row_len - 1, st[k].hits, sec, B, vocab);
                fflush(out);
                fprintf(stderr, "ds4_score: row %d nll %.4f (%.1f s, %.1f tok/s overall)\n", st[k].row,
                        st[k].nll / (row_len - 1), sec, scored / (now_s() - t_start));
                st[k].row = -1;
                active--;
            }
        }
        /* advance every still-active row by its next input token, in one batch */
        ds4_decode_item items[MAX_BATCH];
        int n = 0, idx[MAX_BATCH];
        for (int k = 0; k < B; k++) {
            if (st[k].row < 0) continue;
            items[n].session = st[k].s;
            items[n].token = toks[(size_t)st[k].row * row_len + st[k].pos];
            idx[n++] = k;
        }
        if (n > 0) {
            if (ds4_sessions_eval_batch(items, n, err, sizeof(err)) != 0) {
                fprintf(stderr, "eval failed: %s\n", err);
                return 1;
            }
            for (int j = 0; j < n; j++) st[idx[j]].pos++;
        }
    }
    fclose(out);
    if (lfd >= 0) close(lfd);
    for (int k = 0; k < B; k++) ds4_session_free(st[k].s);
    ds4_engine_close(e);
    return 0;
}

/* ------------------------------------------------------------------------------------------------ serve */

static int cmd_serve(int argc, char **argv) {
    if (argc < 3) die("usage: serve MODEL [options]");
    options o = {.ctx = 16384, .batch = 1};
    if (parse_opts(argc, argv, 3, &o)) return 2;
    ds4_engine *e = open_engine(argv[2], &o, 1);
    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, o.ctx) != 0) die("session create failed");
    printf("READY %d\n", ds4_engine_vocab_size(e));
    fflush(stdout);
    char err[256];
    size_t cap = 1 << 20;
    char *line = malloc(cap);
    ssize_t len;
    while ((len = getline(&line, &cap, stdin)) > 0) {
        char *p = line;
        if (strncmp(p, "GEN ", 4) != 0) {
            printf("ERR bad request\n");
            fflush(stdout);
            continue;
        }
        p += 4;
        long max_new = strtol(p, &p, 10), n_stop = strtol(p, &p, 10);
        int stop[64];
        for (long i = 0; i < n_stop; i++) {
            long v = strtol(p, &p, 10);
            if (i < 64) stop[i] = (int)v;
        }
        if (n_stop > 64) n_stop = 64;
        long n_prompt = strtol(p, &p, 10);
        ds4_tokens prompt = {0};
        for (long i = 0; i < n_prompt; i++) ds4_tokens_push(&prompt, (int)strtol(p, &p, 10));
        if (prompt.len + max_new + 1 >= o.ctx) {
            printf("ERR context %d too small for %d + %ld\n", o.ctx, prompt.len, max_new);
            fflush(stdout);
            ds4_tokens_free(&prompt);
            continue;
        }
        double t0 = now_s();
        ds4_session_invalidate(s);
        if (ds4_session_sync(s, &prompt, err, sizeof(err)) != 0) {
            printf("ERR sync %s\n", err);
            fflush(stdout);
            ds4_tokens_free(&prompt);
            continue;
        }
        const double prefill = now_s() - t0;
        int *gen = malloc((size_t)(max_new + 1) * sizeof(int));
        double *ms = malloc((size_t)(max_new + 1) * sizeof(double));
        int n = 0, nms = 0;
        bool stopped = false, failed = false;
        while (n < max_new) {
            const int tok = ds4_session_argmax(s);
            bool is_stop = false;
            for (long i = 0; i < n_stop; i++) is_stop |= tok == stop[i];
            if (is_stop) { stopped = true; break; }
            gen[n++] = tok;
            if (n == max_new) break;
            double t1 = now_s();
            if (ds4_session_eval(s, tok, err, sizeof(err)) != 0) { failed = true; break; }
            ms[nms++] = (now_s() - t1) * 1e3;
        }
        if (failed) {
            printf("ERR eval %s\n", err);
        } else {
            printf("OK %d %.6f %d", stopped ? 1 : 0, prefill, n);
            for (int i = 0; i < n; i++) printf(" %d", gen[i]);
            for (int i = 0; i < nms; i++) printf(" %.4f", ms[i]);
            printf("\n");
        }
        fflush(stdout);
        free(gen);
        free(ms);
        ds4_tokens_free(&prompt);
    }
    ds4_session_free(s);
    ds4_engine_close(e);
    return 0;
}

/* ------------------------------------------------------------------------------------------------ tokenize */

static int cmd_tokenize(int argc, char **argv) {
    if (argc < 4) die("usage: tokenize MODEL TEXT_FILE");
    FILE *f = fopen(argv[3], "rb");
    if (!f) die("cannot open text");
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = malloc((size_t)n + 1);
    if (fread(text, 1, (size_t)n, f) != (size_t)n) die("read failed");
    text[n] = 0;
    fclose(f);
    options o = {.ctx = 4096, .batch = 1};
    if (parse_opts(argc, argv, 4, &o)) return 2;      /* --ssd-streaming: never map a too-large file resident */
    ds4_engine *e = open_engine(argv[2], &o, 1);
    ds4_tokens tv = {0};
    ds4_tokenize_text(e, text, &tv);
    for (int i = 0; i < tv.len; i++) printf("%d%c", tv.v[i], i + 1 == tv.len ? '\n' : ' ');
    ds4_tokens_free(&tv);
    ds4_engine_close(e);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) die("usage: ds4_score rows|serve|tokenize ...");
    if (!strcmp(argv[1], "rows")) return cmd_rows(argc, argv);
    if (!strcmp(argv[1], "serve")) return cmd_serve(argc, argv);
    if (!strcmp(argv[1], "tokenize")) return cmd_tokenize(argc, argv);
    die("unknown command");
    return 2;
}
