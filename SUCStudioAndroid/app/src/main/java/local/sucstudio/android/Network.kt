package local.sucstudio.android

import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean

class Cancellation {
    val stopped = AtomicBoolean(false)
    private val connections = ConcurrentHashMap.newKeySet<HttpURLConnection>()
    fun check() { if (stopped.get() || Thread.currentThread().isInterrupted) throw InterruptedException("Richiesta interrotta") }
    fun track(connection: HttpURLConnection) { check(); connections.add(connection); if (stopped.get()) connection.disconnect() }
    fun release(connection: HttpURLConnection) { connections.remove(connection); connection.disconnect() }
    fun cancel() { stopped.set(true); connections.toList().forEach { it.disconnect() } }
}

class Http(private val cancellation: Cancellation = Cancellation()) {
    private class SseResult(val value: JSONObject): RuntimeException()
    data class Reply(val body: String, val headers: Map<String, List<String>>)
    fun exchange(url: String, body: JSONObject? = null, headers: Map<String, String> = emptyMap(), stream: ((JSONObject) -> Unit)? = null, longRequest: Boolean = false): Reply {
        cancellation.check()
        val connection = URL(url).openConnection() as HttpURLConnection
        connection.instanceFollowRedirects = false
        connection.connectTimeout = 15000; connection.readTimeout = if (longRequest) 0 else 30000
        connection.requestMethod = if (body == null) "GET" else "POST"
        connection.setRequestProperty("Accept", "application/json, text/event-stream")
        headers.forEach { (key, value) -> require(!key.contains('\n') && !key.contains('\r') && !value.contains('\n') && !value.contains('\r')) { "Header non valido." }; connection.setRequestProperty(key, value) }
        cancellation.track(connection)
        try {
            if (body != null) {
                connection.doOutput = true; connection.setRequestProperty("Content-Type", "application/json")
                val bytes = body.toString().toByteArray(Charsets.UTF_8)
                require(bytes.size <= 24 * 1024 * 1024) { "Richiesta troppo grande (24 MiB)." }
                connection.setFixedLengthStreamingMode(bytes.size); connection.outputStream.use { it.write(bytes) }
            }
            val status = connection.responseCode
            if (status !in 200..299) {
                val error = connection.errorStream?.use { String(it.readBytesLimited(8192), Charsets.UTF_8) }.orEmpty()
                throw IllegalStateException(if (status in 300..399) "Redirect rifiutato: controlla l’URL finale del server." else "HTTP $status: ${error.take(1000)}")
            }
            val responseHeaders = connection.headerFields.filterKeys { it != null }
            if (status == 204 || status == 202) return Reply("", responseHeaders)
            val sse = connection.contentType.orEmpty().contains("text/event-stream")
            if (stream == null && sse) {
                try {
                    connection.inputStream.bufferedReader(Charsets.UTF_8).use { reader -> Protocol.sse(reader) { if (it.has("id") && (it.has("result") || it.has("error"))) throw SseResult(it) } }
                } catch (reply: SseResult) { return Reply(reply.value.toString(), responseHeaders) }
                error("Risposta SSE senza risultato JSON-RPC.")
            }
            if (stream != null && sse) {
                connection.inputStream.bufferedReader(Charsets.UTF_8).use { reader -> Protocol.sse(reader) { cancellation.check(); stream(it) } }
                return Reply("", responseHeaders)
            }
            val text = connection.inputStream.use { String(it.readBytesLimited(Protocol.MAX_RESPONSE), Charsets.UTF_8) }
            cancellation.check()
            if (stream != null && text.isNotBlank()) {
                if (sse) Protocol.sse(text.reader().buffered(), stream) else stream(JSONObject(text))
            }
            return Reply(text, responseHeaders)
        } finally { cancellation.release(connection) }
    }
    fun models(server: JSONObject, key: String): List<String> {
        val reply = exchange(Protocol.endpoint(server.optString("url"), "models", server.optBoolean("allowHttp")), headers = auth(key))
        return JSONObject(reply.body).optJSONArray("data")?.objects()?.mapNotNull { it.optString("id").takeIf(String::isNotBlank) }?.distinct()?.sorted().orEmpty()
    }
    companion object { fun auth(key: String): Map<String, String> = if (key.isBlank()) emptyMap() else mapOf("Authorization" to "Bearer $key") }
}
fun java.io.InputStream.readBytesLimited(limit: Int): ByteArray {
    val out = java.io.ByteArrayOutputStream(); val buffer = ByteArray(8192)
    while (true) { val count = read(buffer); if (count < 0) break; require(out.size() + count <= limit) { "File o risposta oltre il limite di ${limit / 1024} KiB." }; out.write(buffer, 0, count) }
    return out.toByteArray()
}

class McpClient(val config: JSONObject, private val key: String, private val cancellation: Cancellation) {
    private var session = ""; private var sequence = 0; private var protocol = "2025-03-26"
    private val http = Http(cancellation)
    @Synchronized fun rpc(method: String, params: JSONObject? = null, notification: Boolean = false): JSONObject {
        val request = JSONObject().put("jsonrpc", "2.0").put("method", method)
        if (!notification) request.put("id", ++sequence)
        if (params != null) request.put("params", params)
        val headers = Http.auth(key).toMutableMap()
        if (method != "initialize") headers["MCP-Protocol-Version"] = protocol
        if (session.isNotBlank()) headers["Mcp-Session-Id"] = session
        val reply = http.exchange(Protocol.resource(config.getString("url"), config.optBoolean("allowHttp")), request, headers)
        reply.headers.entries.firstOrNull { it.key.equals("Mcp-Session-Id", true) }?.value?.firstOrNull()?.let { session = it }
        if (notification && reply.body.isBlank()) return JSONObject()
        val response = if (reply.body.trimStart().startsWith("data:") || reply.body.trimStart().startsWith("event:")) {
            var found: JSONObject? = null
            Protocol.sse(reply.body.reader().buffered()) { if (it.optInt("id", -1) == sequence) found = it }
            found ?: error("Risposta MCP SSE senza ID corrispondente.")
        } else JSONObject(reply.body)
        if (response.has("error")) error("MCP: ${response.getJSONObject("error").optString("message")}")
        if (!notification) require(response.optInt("id", -1) == sequence) { "ID MCP non corrispondente." }
        return response.optJSONObject("result") ?: JSONObject()
    }
    fun tools(): List<JSONObject> {
        val result = rpc("initialize", JSONObject().put("protocolVersion", protocol).put("capabilities", JSONObject()).put("clientInfo", JSONObject().put("name", "SUC Studio Android").put("version", "0.1.0")))
        protocol = result.optString("protocolVersion", protocol)
        require(protocol in listOf("2024-11-05", "2025-03-26", "2025-06-18")) { "Versione MCP non supportata: $protocol" }
        rpc("notifications/initialized", notification = true)
        val tools = mutableListOf<JSONObject>(); var cursor: String? = null; var pages = 0
        do {
            require(++pages <= 32) { "Troppe pagine MCP." }
            val list = rpc("tools/list", cursor?.let { JSONObject().put("cursor", it) })
            tools.addAll(list.optJSONArray("tools")?.objects().orEmpty())
            require(tools.size <= 128) { "Troppi strumenti MCP." }
            val next = list.optString("nextCursor").takeIf { it.isNotBlank() }
            require(next == null || next != cursor) { "Paginazione MCP non valida." }; cursor = next
        } while (cursor != null)
        return tools
    }
    fun call(name: String, arguments: JSONObject): JSONObject = rpc("tools/call", JSONObject().put("name", name).put("arguments", arguments))
}
