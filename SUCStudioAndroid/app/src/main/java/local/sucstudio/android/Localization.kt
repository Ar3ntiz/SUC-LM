package local.sucstudio.android

import android.content.Context
import org.json.JSONObject

/** Display-only translation. The archive owns the language preference; no duplicate preferences. */
object AndroidL10n {
    @Volatile var language = "en"
        private set
    private var catalog = JSONObject()
    private var templates = listOf<Pair<String, Regex>>()
    fun configure(context: Context, saved: String?) {
        if (catalog.length() == 0) {
            catalog = JSONObject(context.assets.open("localization.json").bufferedReader().use { it.readText() })
            templates = catalog.keys().asSequence().filter { it.contains("{0}") }.sortedByDescending { it.length }.map { key ->
                key to Regex(key.split(Regex("\\{\\d+\\}")).joinToString("([\\s\\S]*?)") { Regex.escape(it) })
            }.toList()
        }
        language = if (saved == "it") "it" else "en"
    }
    fun text(source: String): String {
        if (language == "it") return source
        if (catalog.has(source)) return catalog.getString(source)
        for ((key, pattern) in templates) {
            val match = pattern.matchEntire(source) ?: continue
            return Regex("\\{(\\d+)\\}").replace(catalog.getString(key)) { placeholder ->
                match.groupValues.getOrNull(placeholder.groupValues[1].toInt() + 1) ?: placeholder.value
            }
        }
        return source
    }
}
fun tr(source: String) = AndroidL10n.text(source)
