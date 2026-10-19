package local.sucstudio.android

import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayInputStream
import java.util.zip.ZipInputStream

object PluginPackage {
    fun parse(name: String, bytes: ByteArray): JSONObject {
        require(bytes.size <= 8 * 1024 * 1024) { "Plugin oltre 8 MiB." }
        val entries = linkedMapOf<String, ByteArray>()
        if (bytes.take(2) == listOf(0x50.toByte(), 0x4b.toByte())) {
            var total = 0
            ZipInputStream(ByteArrayInputStream(bytes)).use { zip ->
                while (true) {
                    val entry = zip.nextEntry ?: break
                    require(entries.size < 256 && !entry.name.startsWith('/') && entry.name.split('/').none { it == ".." } && !entry.name.contains('\\')) { "Percorso ZIP non valido o troppi file." }
                    if (!entry.isDirectory) {
                        val data = zip.readBytesLimited(1024 * 1024); total += data.size
                        require(total <= 8 * 1024 * 1024) { "Plugin decompresso oltre 8 MiB." }
                        require(!entries.containsKey(entry.name)) { "File duplicato nel plugin." }; entries[entry.name] = data
                    }
                }
            }
        } else entries["plugin.json"] = bytes
        val manifestPath = entries.keys.firstOrNull { it == "plugin.json" } ?: entries.keys.firstOrNull { it.endsWith("/plugin.json") && !it.contains(".codex-plugin") && !it.contains(".claude-plugin") } ?: error("Manca plugin.json alla radice del pacchetto.")
        val prefix = manifestPath.removeSuffix("plugin.json")
        val manifest = JSONObject(String(entries.getValue(manifestPath), Charsets.UTF_8))
        val title = manifest.optString("name", name); require(title.isNotBlank()) { "Nome plugin obbligatorio." }
        require((!manifest.has("hooks") || manifest.optJSONArray("hooks")?.length() == 0) && !manifest.has("apps") && entries.keys.none { it.endsWith("hooks.json") }) { "Questo plugin contiene hook/app non supportati su Android." }
        val skills = JSONArray(); val mcp = JSONArray()
        if (manifest.has("formatVersion")) {
            require(manifest.getInt("formatVersion") == 1) { "Formato SUC non supportato." }
            require(manifest.optJSONArray("hooks")?.length() ?: 0 == 0) { "Hook non supportati." }
            manifest.optJSONArray("skills")?.objects()?.forEach { source ->
                val skill = Protocol.skillDocument(source.getString("instructions"), source.getString("name"))
                skill.put("description", source.optString("description")); skills.put(skill)
            }
            manifest.optJSONArray("mcpServers")?.objects()?.forEach { source ->
                require(source.optString("transport") == "http") { "MCP stdio non supportato." }
                mcp.put(JSONObject().put("id", id()).put("name", source.getString("name")).put("url", source.getString("endpoint")).put("enabled", false))
            }
        } else {
            require(!manifest.has("extensions")) { "Overlay specifici di altri runtime non supportati: usa il formato portabile." }
            entries.filterKeys { it.startsWith(prefix + "skills/") && it.endsWith("/SKILL.md") }.forEach { (path, data) ->
                val folder = path.removeSuffix("SKILL.md")
                val skill = Protocol.skillDocument(String(data, Charsets.UTF_8), folder.trimEnd('/').substringAfterLast('/'))
                val resources = JSONObject()
                entries.filterKeys { it.startsWith(folder) && it != path }.forEach { (resource, value) ->
                    val relative = resource.removePrefix(folder)
                    // Resource code is never executed. Text references may be read through load_skill_resource.
                    if (value.size <= 64 * 1024 && relative.startsWith("references/")) resources.put(relative, String(value, Charsets.UTF_8))
                }
                skill.put("resources", resources); skills.put(skill)
            }
            entries[prefix + "mcp.json"]?.let { data ->
                val servers = JSONObject(String(data, Charsets.UTF_8)).getJSONObject("mcpServers")
                servers.keys().forEach { key ->
                    val server = servers.getJSONObject(key)
                    require(!server.has("command") && server.optString("type", "http") in listOf("http", "streamable-http")) { "Il plugin contiene MCP stdio non supportati." }
                    require(!server.has("headers") && !server.has("env")) { "Configura le credenziali MCP nell’app, non nel plugin." }
                    mcp.put(JSONObject().put("id", id()).put("name", key).put("url", Protocol.resource(server.getString("url"), true)).put("enabled", false).put("allowHttp", server.getString("url").startsWith("http:")))
                }
            }
        }
        require(skills.length() <= 64 && mcp.length() <= 16) { "Troppi componenti nel plugin." }
        require(skills.length() + mcp.length() > 0) { "Il plugin non contiene skill o MCP HTTP." }
        return JSONObject().put("id", id()).put("name", title).put("description", manifest.optString("description")).put("version", manifest.optString("version", "1.0.0")).put("skills", skills).put("mcp", mcp).put("enabled", false)
    }
}
