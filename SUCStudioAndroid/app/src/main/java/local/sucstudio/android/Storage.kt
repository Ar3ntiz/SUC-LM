package local.sucstudio.android

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.AtomicFile
import android.util.Base64
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

class Vault(context: Context) {
    private val prefs = context.getSharedPreferences("secrets", Context.MODE_PRIVATE)
    private fun key(): SecretKey {
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (store.getKey("suc.api.keys", null) as? SecretKey)?.let { return it }
        return KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore").apply {
            init(KeyGenParameterSpec.Builder("suc.api.keys", KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT).setBlockModes(KeyProperties.BLOCK_MODE_GCM).setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE).build())
        }.generateKey()
    }
    fun write(name: String, value: String) {
        if (value.isBlank()) { prefs.edit().remove(name).apply(); return }
        val cipher = Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.ENCRYPT_MODE, key()) }
        val encrypted = cipher.doFinal(value.toByteArray(Charsets.UTF_8))
        prefs.edit().putString(name, Base64.encodeToString(cipher.iv + encrypted, Base64.NO_WRAP)).apply()
    }
    fun read(name: String): String {
        val text = prefs.getString(name, null) ?: return ""
        val bytes = Base64.decode(text, Base64.NO_WRAP); require(bytes.size > 28) { "Credenziale non valida: reinserisci l’API key." }
        return String(Cipher.getInstance("AES/GCM/NoPadding").run { init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(128, bytes.copyOfRange(0, 12))); doFinal(bytes.copyOfRange(12, bytes.size)) }, Charsets.UTF_8)
    }
}
class ArchiveStore(context: Context) {
    private val file = AtomicFile(File(context.filesDir, "archive.json"))
    fun load(): JSONObject = if (file.baseFile.exists()) JSONObject(file.openRead().use { String(it.readBytesLimited(64 * 1024 * 1024), Charsets.UTF_8) }) else JSONObject()
    @Synchronized fun save(text: String) {
        require(text.toByteArray().size <= 64 * 1024 * 1024) { "Archivio oltre 64 MiB: esporta ed elimina alcune conversazioni." }
        val output = file.startWrite()
        try { output.write(text.toByteArray(Charsets.UTF_8)); file.finishWrite(output) } catch (failure: Exception) { file.failWrite(output); throw failure }
    }
    companion object {
        fun normalize(root: JSONObject): JSONObject {
            root.put("formatVersion", 1)
            if (!root.has("settings")) root.put("settings", JSONObject())
            val settings = root.getJSONObject("settings")
            settings.put("language", if (settings.optString("language") == "it") "it" else "en")
            if (!settings.has("server")) settings.put("server", JSONObject().put("url", "").put("model", "").put("allowHttp", false))
            if (!settings.has("maxTokens")) settings.put("maxTokens", 4000)
            for (name in listOf("skills", "mcp", "plugins")) if (!settings.has(name)) settings.put(name, JSONArray())
            if (!root.has("sessions")) root.put("sessions", JSONArray())
            root.getJSONArray("sessions").objects().forEach { s -> s.optJSONArray("messages")?.objects()?.forEach { m -> if (m.optBoolean("generating")) { m.put("generating", false); m.put("interrupted", true) } } }
            return root
        }
    }
}
