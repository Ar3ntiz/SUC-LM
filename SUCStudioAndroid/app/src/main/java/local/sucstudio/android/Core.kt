package local.sucstudio.android

import org.json.JSONArray
import org.json.JSONObject
import org.json.JSONTokener
import java.io.BufferedReader
import java.net.URI
import java.util.UUID

fun JSONArray.objects(): List<JSONObject> = (0 until length()).mapNotNull { optJSONObject(it) }
fun JSONObject.copyJson() = JSONObject(toString())
fun id() = UUID.randomUUID().toString()

object Protocol {
    const val MAX_RESPONSE = 4 * 1024 * 1024
    fun endpoint(base: String, path: String, allowHttp: Boolean = false): String {
        val uri = URI(base.trim())
        require(uri.scheme == "https" || (allowHttp && uri.scheme == "http")) { "Usa HTTPS, oppure abilita HTTP per la rete fidata." }
        require(!uri.host.isNullOrBlank() && uri.rawUserInfo == null && uri.query == null && uri.fragment == null) { "URL non valido: niente credenziali, query o frammenti." }
        val prefix = uri.path.orEmpty().trimEnd('/').ifEmpty { "/v1" }
        return URI(uri.scheme, null, uri.host, uri.port, "$prefix/$path", null, null).toASCIIString()
    }
    fun resource(url: String, allowHttp: Boolean = false): String {
        val uri = URI(url.trim())
        require(uri.scheme == "https" || (allowHttp && uri.scheme == "http")) { "URL HTTPS richiesto." }
        require(!uri.host.isNullOrBlank() && uri.rawUserInfo == null && uri.fragment == null) { "URL non valido." }
        return uri.toASCIIString()
    }
    fun sse(reader: BufferedReader, receive: (JSONObject) -> Unit) {
        val event = StringBuilder()
        fun dispatch(): Boolean {
            val data = event.toString().trim(); event.setLength(0)
            if (data == "[DONE]") return false
            if (data.isNotEmpty()) receive(JSONObject(data))
            return true
        }
        while (true) {
            val line = StringBuilder()
            while (true) {
                val c = reader.read()
                if (c < 0) { if (line.isNotEmpty()) break; dispatch(); return }
                if (c == 10) break
                if (c != 13) line.append(c.toChar())
                require(line.length <= 512 * 1024) { "Evento server troppo grande." }
            }
            if (line.isEmpty()) { if (!dispatch()) return }
            else if (line.startsWith("data:")) {
                if (event.isNotEmpty()) event.append('\n')
                event.append(line.substring(5).removePrefix(" "))
                require(event.length <= 1024 * 1024) { "Evento server troppo grande." }
            }
        }
    }
    data class Action(val name: String, val arguments: JSONObject)
    private val envelopeStart = Regex("(?m)^\\s*(?:```(?:json)?\\s*\\n)?\\s*\\{\\s*\"suc_action\"\\s*:")
    fun action(text: String): Action? {
        val clean = text.replace(Regex("(?s)<think>.*?</think>"), "").trim()
        val matches = envelopeStart.findAll(clean).toList(); if (matches.size != 1) return null
        val match = matches.single()
        val candidate = clean.substring(match.range.first).trim().removePrefix("```json").removePrefix("```").trim().removeSuffix("```").trim()
        return try {
            val parser = JSONTokener(candidate); val json = parser.nextValue() as? JSONObject ?: return null
            require(parser.nextClean() == 0.toChar())
            require(json.length() == 1)
            val a = json.getJSONObject("suc_action")
            val name = a.getString("name"); require(name.length in 1..256)
            Action(name, a.optJSONObject("arguments") ?: JSONObject())
        } catch (_: Exception) { null }
    }
    fun hasActionEnvelope(text: String) = envelopeStart.containsMatchIn(text)
    fun visible(text: String, final: Boolean): String {
        val found = envelopeStart.find(text)
        if (found != null) return text.substring(0, found.range.first).trimEnd()
        if (!final) {
            val pending = Regex("(?m)^\\s*\\{").findAll(text).lastOrNull()
            if (pending != null) {
                val suffix = text.substring(pending.range.first)
                if (!suffix.contains('}')) return text.substring(0, pending.range.first).trimEnd()
            }
            val lines = text.split('\n'); val last = lines.last().trimStart()
            // Hold an unfinished JSON envelope until it can be classified; never leak its prefixes.
            if ((last.startsWith("{") && !last.contains("}")) || last.startsWith("```json") || last == "```") return lines.dropLast(1).joinToString("\n").trimEnd()
            val fence = text.lastIndexOf("\n```json\n")
            if (fence >= 0 && !text.substring(fence + 9).contains("```")) return text.substring(0, fence).trimEnd()
        }
        return text
    }
    fun skillDocument(text: String, fallback: String): JSONObject {
        require(text.toByteArray().size <= 64 * 1024) { "Skill oltre 64 KiB." }
        var title = fallback; var description = ""; var body = text
        if (text.startsWith("---\n") || text.startsWith("---\r\n")) {
            val normalized = text.replace("\r\n", "\n"); val end = normalized.indexOf("\n---", 4)
            require(end > 0) { "Frontmatter non chiuso." }
            normalized.substring(4, end).lines().forEach { line ->
                val parts = line.split(':', limit = 2)
                if (parts.size == 2) {
                    val value = parts[1].trim().trim('"', '\'')
                    when (parts[0].trim()) { "name" -> title = value; "description" -> description = value }
                }
            }
            body = normalized.substring(end + 4).trim()
        }
        require(title.length <= 128 && description.toByteArray().size <= 4096 && description !in listOf("|", ">")) { "Titolo entro 128 caratteri e descrizione semplice entro 4 KiB." }
        require(title.isNotBlank() && body.isNotBlank()) { "Titolo e istruzioni obbligatori." }
        return JSONObject().put("id", id()).put("name", title).put("description", description).put("instructions", body).put("enabled", false)
    }
    fun request(settings: JSONObject, session: JSONObject, catalog: String): JSONObject {
        val server = settings.optJSONObject("server") ?: JSONObject()
        require(server.optString("model").isNotBlank()) { "Seleziona un modello nella sezione Server." }
        val messages = JSONArray()
        val prompt = listOf(settings.optString("systemPrompt"), session.optString("systemPrompt"), catalog).filter { it.isNotBlank() }.joinToString("\n\n")
        if (prompt.isNotBlank()) messages.put(JSONObject().put("role", "system").put("content", prompt))
        session.optJSONArray("messages")?.objects()?.forEach { m ->
            if (m.optBoolean("error") || m.optBoolean("interrupted")) return@forEach
            val role = m.optString("role")
            if (role !in listOf("user", "assistant", "tool")) return@forEach
            if (role == "assistant" && m.optString("text").isBlank() && !m.has("tool_calls")) return@forEach
            val out = JSONObject().put("role", role).put("content", m.optString("text"))
            if (m.has("tool_calls")) out.put("tool_calls", m.getJSONArray("tool_calls"))
            if (m.has("tool_call_id")) out.put("tool_call_id", m.getString("tool_call_id"))
            if (role == "user") {
                val parts = JSONArray(); val text = StringBuilder(m.optString("text"))
                m.optJSONArray("attachments")?.objects()?.forEach { a ->
                    if (a.has("dataUrl")) parts.put(JSONObject().put("type", "image_url").put("image_url", JSONObject().put("url", a.getString("dataUrl"))))
                    else if (a.optString("text").isNotEmpty()) text.append("\n\nAllegato: ${a.optString("name")}\n${a.optString("text")}")
                }
                if (parts.length() > 0) { val all = JSONArray().put(JSONObject().put("type", "text").put("text", text.toString())); parts.objects().forEach { all.put(it) }; out.put("content", all) }
                else out.put("content", text.toString())
            }
            messages.put(out)
        }
        return JSONObject().put("model", server.getString("model")).put("messages", messages).put("stream", true)
            .put("max_tokens", settings.optInt("maxTokens", 4000).coerceIn(1, 131072))
            .put("temperature", settings.optDouble("temperature", 0.7).coerceIn(0.0, 2.0))
            .put("stream_options", JSONObject().put("include_usage", true))
            .apply { for (key in listOf("top_p", "top_k", "min_p", "seed")) if (settings.has(key) && (key == "top_p" || (key == "seed" && settings.optInt(key, -1) >= 0) || (key != "seed" && settings.optDouble(key) > 0))) put(key, settings.get(key)) }
    }
}

class StreamAccumulator {
    var content = ""; var reasoning = ""; var promptTokens = 0; var completionTokens = 0
    var prefillRate: Double? = null; var decodeRate: Double? = null
    private var reportedCachedTokens = 0
    private var timedPromptTokens: Int? = null
    val cachedTokens: Int get() = reportedCachedTokens.coerceIn(0, promptTokens.coerceAtLeast(0))
    private var prefillSeconds: Double? = null; private var decodeSeconds: Double? = null
    var finishReason = ""; val calls = mutableMapOf<Int, JSONObject>()
    private val start = System.nanoTime(); var firstText: Long? = null
    fun accept(event: JSONObject) {
        val usage = event.optJSONObject("usage")
        if (usage != null) {
            promptTokens = usage.optInt("prompt_tokens", promptTokens).coerceAtLeast(0)
            completionTokens = usage.optInt("completion_tokens", completionTokens).coerceAtLeast(0)
            // Usage can arrive in a final SSE frame, separately from timings.
            usage.optJSONObject("prompt_tokens_details")?.let { details ->
                if (details.has("cached_tokens") && !details.isNull("cached_tokens"))
                    reportedCachedTokens = details.optInt("cached_tokens", 0).coerceAtLeast(0)
            }
        }
        val suc = event.optJSONObject("suc_metrics")
        suc?.optDouble("prefill_s", 0.0)?.takeIf { it.isFinite() && it > 0 }?.let { prefillSeconds = it; timedPromptTokens = null }
        suc?.optDouble("decode_s", 0.0)?.takeIf { it.isFinite() && it > 0 }?.let { decodeSeconds = it }
        val timings = event.optJSONObject("timings")
        if (timings != null) {
            if (promptTokens == 0) promptTokens = timings.optInt("prompt_n", 0).coerceAtLeast(0)
            if (completionTokens == 0) completionTokens = timings.optInt("predicted_n", 0).coerceAtLeast(0)
            val promptMs = timings.optDouble("prompt_ms", 0.0)
            if (promptMs.isFinite() && promptMs > 0) {
                prefillSeconds = promptMs / 1000
                // llama.cpp prompt_n already counts evaluated (uncached) tokens.
                // Keep it independent of OpenAI usage.prompt_tokens, the full input.
                timedPromptTokens = timings.optInt("prompt_n", -1).takeIf { it >= 0 }
            }
            val decodeMs = timings.optDouble("predicted_ms", 0.0)
            val predicted = timings.optDouble("predicted_n", 0.0)
            if (decodeMs.isFinite() && decodeMs > 0 && predicted.isFinite() && predicted > 0 && decodeRate == null) decodeRate = predicted / (decodeMs / 1000)
        }
        event.optJSONObject("prompt_progress")?.let { progress ->
            val processed = progress.optInt("processed") - progress.optInt("cache")
            val time = progress.optDouble("time_ms", 0.0)
            if (processed > 0 && time.isFinite() && time > 0) prefillRate = processed / (time / 1000)
        }
        val candidates = listOfNotNull(event.optJSONObject("stats"), timings, event.optJSONObject("metrics"), event.optJSONObject("telemetry"), suc, usage, event)
        for (o in candidates) {
            for (key in listOf("prompt_per_second", "prefill_tokens_per_second", "prefill_tps", "prefill_tok_s")) if (o.optDouble(key, 0.0).isFinite() && o.optDouble(key, 0.0) > 0) prefillRate = o.getDouble(key)
            for (key in listOf("predicted_per_second", "decode_tokens_per_second", "decode_tps", "tokens_per_second", "decode_tok_s")) if (o.optDouble(key, 0.0).isFinite() && o.optDouble(key, 0.0) > 0) decodeRate = o.getDouble(key)
        }
        val choice = event.optJSONArray("choices")?.optJSONObject(0) ?: return
        finishReason = choice.optString("finish_reason").takeUnless { it == "null" || it.isBlank() } ?: finishReason
        val delta = choice.optJSONObject("delta") ?: choice.optJSONObject("message") ?: return
        content += delta.optString("content").takeUnless { it == "null" }.orEmpty()
        reasoning += delta.optString("reasoning_content", delta.optString("reasoning")).takeUnless { it == "null" }.orEmpty()
        if (firstText == null && (content.isNotEmpty() || reasoning.isNotEmpty())) firstText = System.nanoTime()
        delta.optJSONArray("tool_calls")?.objects()?.forEach { item ->
            val index = item.optInt("index"); val call = calls.getOrPut(index) { JSONObject().put("id", "call_${id()}").put("type", "function").put("function", JSONObject().put("name", "").put("arguments", "")) }
            if (item.has("id")) call.put("id", item.getString("id"))
            val function = item.optJSONObject("function") ?: return@forEach
            val target = call.getJSONObject("function")
            for (key in listOf("name", "arguments")) if (function.has(key)) target.put(key, target.optString(key) + function.optString(key))
        }
        require(content.length + reasoning.length <= Protocol.MAX_RESPONSE) { "Risposta oltre il limite di sicurezza (4 MiB)." }
    }
    fun metrics(): JSONObject {
        val elapsed = (System.nanoTime() - start) / 1e9
        val first = firstText?.let { (it - start) / 1e9 }
        val uncachedTokens = (promptTokens - cachedTokens).coerceAtLeast(0)
        val evaluatedTokens = timedPromptTokens ?: uncachedTokens
        val serverPrefill = prefillRate ?: prefillSeconds?.takeIf { evaluatedTokens > 0 }?.let { evaluatedTokens / it }
        val estimatedPrefill = if (first != null && first > 0 && evaluatedTokens > 0) evaluatedTokens / first else null
        val serverDecode = decodeRate ?: decodeSeconds?.takeIf { completionTokens > 1 }?.let { (completionTokens - 1) / it }
        val estimatedDecode = if (first != null && elapsed > first && completionTokens > 1) (completionTokens - 1) / (elapsed - first) else null
        return JSONObject().put("promptTokens", promptTokens).put("completionTokens", completionTokens)
            .put("cachedTokens", cachedTokens).put("uncachedPromptTokens", uncachedTokens)
            .put("prefill", serverPrefill ?: estimatedPrefill ?: JSONObject.NULL)
            .put("decode", serverDecode ?: estimatedDecode ?: JSONObject.NULL)
            .put("prefillEstimated", serverPrefill == null && estimatedPrefill != null).put("decodeEstimated", serverDecode == null && estimatedDecode != null)
            .put("phase", if (firstText == null) "prefill" else "decode")
    }
}
