// suc1-generate: the glm5_next (GLM-5.3-Flash) branch, included inside main() of suc1_generate.cpp (uses its locals:
// the parsed options, the opened file f, the Context ctx, the MappedFile mf, the optional ExpertStream xs and the
// stream_report lambda). Prompt prefill: the batched (not row-exact) layer-major chunks by default (--chunk rows,
// 4096 when not given (1024 beyond 8,192-token prompts), at most the prompt; Glm5Options::layer_prefill),
// --prefill exact the previous paths (decode steps, --chunk of them per command buffer, 16 by default; streamed
// experts: row-exact layer-major chunks of 64),
// --prefill token decode steps one by one.
{
  // 8,192 tokens, M1 Ultra: 99.8 / 99.7 / 105.9 tok/s at chunks 1024 / 2048 / 4096 (prof_glm_new_8192*,
  // docs/prefill_vs_ds4.md); prompts beyond kLargeChunkMaxPrompt tokens in chunks of 1024 (GPU memory, as deepseek_v4)

  if (xs && mixed_rows == ExpertStack::Rows::All)
    throw std::invalid_argument("--expert-budget with --mixed-experts all: streamed stacks run routed rows only");
  Glm5Options gopt;
  gopt.expert_stream = xs.get();
  const Glm5Config gc = Glm5Config::from_architecture(f.directory()["architecture"]);
  if (!dtype.empty()) gopt.act = parse_act(dtype);
  else gopt.act = gc.torch_dtype == "float16" ? Act::F16 : gc.torch_dtype == "float32" ? Act::F32 : Act::BF16;
  gopt.max_context = max_ctx ? max_ctx : prompt.size() + steps + 1;
  gopt.mixed_rows = mixed_rows;
  const bool want_logits = !logits_out.empty(), want_hidden = !hidden_out.empty(), want_nll = !nll_out.empty();
  // --logits-out alone with an explicit --prefill batch|exact: every prompt row's logits from the batched chunks
  // (forward_rows; scripts/prefill_batch_vs_token.py); without --prefill, --logits-out runs decode steps
  const bool batch_rows = want_logits && !want_hidden && !want_nll && batch_prefill && rf.prefill_given;
  const bool rows_needed = (want_logits || want_hidden || want_nll) && !batch_rows;
  const bool exact = rf.exact_prefill || prefill_exact_env();
  gopt.layer_prefill = batch_prefill && !rows_needed;
  gopt.exact_prefill = rf.exact_prefill;
  gopt.chunk = rows_needed ? 1 : suc::prefill_chunk("glm5_next", prompt.size(), rf.chunk_given ? chunk : 0,
      exact, batch_prefill, bool(xs));
  Glm5Model model(ctx, mf, gopt);
  const bool fast_prefill = model.batched_prefill();
  const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::vector<float> all_logits, all_hidden, nll;
  auto record = [&](std::size_t p) {
    if (want_logits || want_nll) {
      const auto l = model.logits();
      if (want_logits) all_logits.insert(all_logits.end(), l.begin(), l.end());
      if (want_nll && p + 1 < prompt.size()) {
        const double mx = *std::max_element(l.begin(), l.end());
        double se = 0.0;
        for (float v : l) se += std::exp(double(v) - mx);
        nll.push_back(float(mx + std::log(se) - double(l[prompt[p + 1]])));
      }
    }
    if (want_hidden) {
      const auto h = model.hidden_states();
      all_hidden.insert(all_hidden.end(), h.begin(), h.end());
    }
  };
  model.step(prompt[0], true);   // warm-up: compile the pipelines, then reset
  if (fast_prefill && prompt.size() > 1) {   // and the batched-prefill pipelines (one chunk), then one decode step
    model.reset();                            // past it (the decode attention of > 1 keys, else compiled at decode)
    model.forward(std::vector<uint32_t>(prompt.begin(), prompt.begin() + std::min(prompt.size(), gopt.chunk)), true);
    model.step(prompt[0], true);
  }
  model.reset();
  model.reset_profile();
  if (xs) xs->reset_stats();
  const double compile_s = ctx.compile_seconds();
  uint32_t next = 0;
  const auto p0 = std::chrono::steady_clock::now();
  if (rows_needed) {
    for (std::size_t i = 0; i < prompt.size(); ++i) {
      next = model.step(prompt[i], true);
      record(i);
    }
  } else if (batch_rows) {
    all_logits = model.forward_rows(prompt);
    const auto l = model.logits();
    next = uint32_t(std::max_element(l.begin(), l.end()) - l.begin());
  } else {
    next = model.forward(prompt, true);
  }
  const double prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count();
  const double prefill_gpu_s = model.last_gpu_seconds();
  const uint64_t prefill_chunk_fetches = model.chunk_fetches();
  for (const auto& [stage, g] : model.profile())   // $SUC_GLM_PROFILE=1
    std::fprintf(stderr, "prefill profile %-22s GPU %9.3f ms\n", stage.c_str(), 1e3 * g);
  std::string prefill_logits_sha, prefill_hidden_sha;   // cross-checks of the prefill modes (same bits expected)
  {
    auto hex = [](const std::vector<float>& v) {
      const auto d = sha256(v.data(), v.size() * sizeof(float));
      static const char* digits = "0123456789abcdef";
      std::string r;
      for (uint8_t b : d) r += digits[b >> 4], r += digits[b & 15];
      return r;
    };
    const auto l = model.logits();
    const auto h = model.hidden_states();
    prefill_logits_sha = hex(l);
    prefill_hidden_sha = hex(h);
    if (const char* e = std::getenv("SUC_PREFILL_DUMP")) {   // the last prompt token's streams and logits
      write_npy_f32(std::string(e) + ".streams.npy", {gc.layers + 1, h.size() / (gc.layers + 1)}, h.data());
      write_npy_f32(std::string(e) + ".logits.npy", {l.size()}, l.data());
    }
  }
  Sampler sampler(sc);
  auto draw = [&]() {
    const auto l = model.logits();
    return sampler.sample(l.data(), l.size());
  };
  if (!sc.greedy()) next = draw();
  std::vector<uint32_t> generated;
  std::vector<double> decode_ms, decode_gpu_ms, decode_enc_ms;
  std::vector<uint32_t> stop_ids = gc.eos;
  if (tok) {
    for (uint32_t t : tok->eos_ids()) stop_ids.push_back(t);
    for (uint32_t t : tok->turn_end_ids()) stop_ids.push_back(t);
  }
  for (std::size_t s = 0; s < steps; ++s) {
    generated.push_back(next);
    if (stop_eos && std::find(stop_ids.begin(), stop_ids.end(), next) != stop_ids.end()) break;
    if (s + 1 == steps) break;
    const uint32_t feed = forced.empty() ? next : forced[s];
    const auto s0 = std::chrono::steady_clock::now();
    next = model.step(feed, true);
    if (!sc.greedy()) next = draw();
    decode_ms.push_back(1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count());
    decode_gpu_ms.push_back(1e3 * model.last_gpu_seconds());
    decode_enc_ms.push_back(1e3 * model.last_encode_seconds());
    if (rows_needed || batch_rows) record(model.position() - 1);
  }
  const std::size_t positions = model.position();
  const std::size_t L1 = gc.layers + 1, Wd = gc.stream();
  if (want_logits) write_npy_f32(logits_out, {all_logits.size() / gc.vocab, gc.vocab}, all_logits.data());
  if (want_nll) write_npy_f32(nll_out, {nll.size()}, nll.data());
  if (want_hidden) write_npy_f32(hidden_out, {all_hidden.size() / (L1 * Wd), L1, Wd}, all_hidden.data());
  const Glm5Memory mr = model.memory();
  const Glm5StackStats st = model.expert_stacks();
  const ProcessMemory pm = process_memory();
  const uint64_t file_resident = resident_bytes(f.bytes(0, f.file_size()).data(), f.file_size());
  const double dmed = median(decode_ms), dgpu = median(decode_gpu_ms), denc = median(decode_enc_ms);
  std::size_t n_dsa = 0, n_moe = 0;
  for (std::size_t i = 0; i < gc.layers; ++i) n_dsa += gc.dsa[i], n_moe += gc.sparse[i];
  if (prompt_text) std::printf("prompt: %s\n", ids_str(prompt).c_str());
  std::printf("generated: %s\n", ids_str(generated).c_str());
  if (tok) std::printf("text: %s\n", tok->decode(generated).c_str());
  if (!quiet)
    std::fprintf(stderr,
                 "model glm5_next: %zu layers (%zu KDA, %zu MLA + DSA; %zu MoE), hidden %zu x %zu streams, %zu experts "
                 "top-%zu, %zu SUC projections, %zu expert stacks (%zu mixed, %zu sub-stacks), activations %s, device %s\n"
                 "load %.3f s (pipelines compiled %zu in %.3f s)\n"
                 "prefill %zu tokens: %.2f ms/token; decode %zu tokens: median %.2f ms/token (GPU %.2f ms, encode %.2f ms), "
                 "%.1f tokens/s\n"
                 "memory: GPU-mapped file %.1f MB (file %.1f MB resident in the page cache), copied execution arrays "
                 "%.2f MB, KV cache %.2f MB (%zu positions), KDA states %.2f MB, arena %.2f MB, workspace %.2f MB; "
                 "process phys_footprint %.1f MB, resident %.1f MB (peak %.1f MB)\n",
                 gc.layers, gc.layers - n_dsa, n_dsa, n_moe, gc.hidden, gc.hc, gc.experts, gc.top_k,
                 model.suc_projections(), st.stacks, st.mixed, st.sub_stacks, std::string(act_name(gopt.act)).c_str(),
                 ctx.device_name().c_str(), load_s, ctx.pipelines_compiled(), compile_s, prompt.size(),
                 1e3 * prefill_s / double(prompt.size()), decode_ms.size(), dmed, dgpu, denc,
                 dmed > 0 ? 1e3 / dmed : 0.0, mf.mapped_bytes() / 1e6, file_resident / 1e6, mr.copied_weights / 1e6,
                 mr.kv_cache / 1e6, gopt.max_context, mr.states / 1e6, mr.arena / 1e6, mr.workspace / 1e6,
                 pm.phys_footprint / 1e6, pm.resident / 1e6, pm.resident_peak / 1e6);
  if (!json_out.empty()) {
    std::ofstream o(json_out);
    o << "{\"file\": \"" << path << "\", \"device\": \"" << ctx.device_name() << "\", \"architecture\": \""
      << ctx.architecture() << "\", \"model_type\": \"glm5_next\", \"activation_dtype\": \"" << act_name(gopt.act)
      << "\", \"prompt_tokens\": " << prompt.size() << ", \"generated\": [" << ids_str(generated)
      << "], \"teacher_forced\": " << (forced.empty() ? "false" : "true") << ", \"positions\": " << positions
      << ", \"load_s\": " << load_s << ", \"compile_s\": " << compile_s << ", \"chunk\": " << gopt.chunk
      << ", \"prefill\": \"" << (!gopt.layer_prefill ? "token" : fast_prefill ? "batch" : "exact")
      << "\", \"prefill_gpu_s\": " << prefill_gpu_s
      << ", \"prefill_mode\": \"" << (fast_prefill && gopt.chunk > 1 ? "batched" : xs && gopt.chunk > 1 ? "layer_major" : "decode_steps")
      << "\", \"prefill_chunk_fetches\": " << prefill_chunk_fetches << ", \"prefill_logits_sha256\": \""
      << prefill_logits_sha << "\", \"prefill_hidden_sha256\": \"" << prefill_hidden_sha << "\""
      << ", \"prefill_s\": " << prefill_s << ", \"decode_ms_per_token_median\": " << dmed
      << ", \"decode_gpu_ms_per_token_median\": " << dgpu << ", \"decode_encode_ms_per_token_median\": " << denc
      << ", \"mixed_experts\": \"" << (mixed_rows == ExpertStack::Rows::All ? "all" : "routed")
      << "\", \"expert_stacks\": {\"stacks\": " << st.stacks << ", \"mixed\": " << st.mixed << ", \"sub_stacks\": "
      << st.sub_stacks << "}, \"decode_ms_per_token\": [";
    for (std::size_t i = 0; i < decode_ms.size(); ++i) o << (i ? ", " : "") << decode_ms[i];
    o << "], \"memory\": {\"mapped_file\": " << mr.mapped_file << ", \"file_resident\": " << file_resident
      << ", \"file_size\": " << f.file_size() << ", \"copied_execution_arrays\": " << mr.copied_weights
      << ", \"kv_cache\": " << mr.kv_cache << ", \"states\": " << mr.states << ", \"arena\": " << mr.arena
      << ", \"workspace\": " << mr.workspace << ", \"max_context\": " << gopt.max_context
      << ", \"phys_footprint\": " << pm.phys_footprint << ", \"resident\": " << pm.resident
      << ", \"resident_peak\": " << pm.resident_peak << "}, \"suc_projections\": " << model.suc_projections();
    stream_report(&o);
    o << "}\n";
  } else {
    stream_report(nullptr);
  }
  return 0;
}
