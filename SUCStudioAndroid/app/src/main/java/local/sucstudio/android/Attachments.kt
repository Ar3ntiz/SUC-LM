package local.sucstudio.android

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Base64
import android.util.Xml
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper
import org.apache.poi.hwpf.HWPFDocument
import org.apache.poi.hwpf.extractor.WordExtractor
import org.json.JSONObject
import org.xmlpull.v1.XmlPullParser
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.util.zip.ZipInputStream

class Attachments(private val context: Context) {
    fun read(uri: Uri): JSONObject {
        val name = context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { if (it.moveToFirst()) it.getString(0) else null } ?: uri.lastPathSegment ?: "allegato"
        val bytes = context.contentResolver.openInputStream(uri)?.use { it.readBytesLimited(10 * 1024 * 1024) } ?: error("File non leggibile.")
        val mime = context.contentResolver.getType(uri).orEmpty()
        val result = JSONObject().put("id", id()).put("name", name).put("bytes", bytes.size)
        if (mime.startsWith("image/") || BitmapFactory.Options().also { BitmapFactory.decodeByteArray(bytes, 0, bytes.size, it.apply { inJustDecodeBounds = true }) }.outWidth > 0) {
            val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }; BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options)
            require(options.outWidth > 0 && options.outHeight > 0) { "Immagine non supportata." }
            val longest = maxOf(options.outWidth, options.outHeight)
            options.inSampleSize = maxOf(1, longest / 1600); options.inJustDecodeBounds = false
            val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options) ?: error("Immagine non decodificabile.")
            val out = ByteArrayOutputStream(); bitmap.compress(Bitmap.CompressFormat.JPEG, 85, out)
            val ratio = minOf(1.0, 240.0 / maxOf(bitmap.width, bitmap.height)); val thumb = Bitmap.createScaledBitmap(bitmap, maxOf(1, (bitmap.width * ratio).toInt()), maxOf(1, (bitmap.height * ratio).toInt()), true)
            val preview = ByteArrayOutputStream(); thumb.compress(Bitmap.CompressFormat.JPEG, 65, preview); if (thumb !== bitmap) thumb.recycle(); bitmap.recycle()
            result.put("preview", "data:image/jpeg;base64," + Base64.encodeToString(preview.toByteArray(), Base64.NO_WRAP))
            val file = File(context.filesDir, "attachments/${result.getString("id")}.jpg").apply { parentFile?.mkdirs(); writeBytes(out.toByteArray()) }
            result.put("image", file.name).put("mime", "image/jpeg").put("notice", "Immagine inviata al server: richiede un modello vision.")
        } else {
            val extension = name.substringAfterLast('.', "").lowercase()
            val text = when {
                extension == "pdf" || bytes.take(4) == "%PDF".toByteArray().toList() -> {
                    PDFBoxResourceLoader.init(context)
                    PDDocument.load(bytes).use { document ->
                        require(!document.isEncrypted) { "PDF protetto: rimuovi la protezione prima di allegarlo." }
                        require(document.numberOfPages <= 300) { "PDF oltre 300 pagine." }
                        PDFTextStripper().getText(document)
                    }
                }
                extension == "doc" -> HWPFDocument(ByteArrayInputStream(bytes)).use { doc -> WordExtractor(doc).use { it.text } }
                extension in listOf("docx", "odt", "pptx", "xlsx", "odp", "ods") || bytes.take(2) == listOf(0x50.toByte(), 0x4b.toByte()) -> office(bytes)
                else -> decode(bytes).let { if (extension == "rtf") rtf(it) else it }
            }
            require(text.isNotBlank()) { "Nessun testo leggibile. Per scansioni usa un’immagine e un modello vision." }
            require(text.toByteArray().size <= 256 * 1024) { "Testo oltre 256 KiB: dividi il documento." }
            result.put("text", text)
        }
        return result
    }
    fun hydrate(attachment: JSONObject): JSONObject = attachment.copyJson().apply {
        if (has("image")) {
            val name = getString("image"); require(Regex("[a-f0-9-]+\\.jpg").matches(name)) { "Riferimento immagine non valido." }
            val file = File(context.filesDir, "attachments/$name")
            put("dataUrl", "data:image/jpeg;base64," + Base64.encodeToString(file.readBytes(), Base64.NO_WRAP))
        }
    }
    private fun decode(bytes: ByteArray): String {
        val (offset, charset) = when {
            bytes.size >= 2 && bytes[0] == 0xff.toByte() && bytes[1] == 0xfe.toByte() -> 2 to Charsets.UTF_16LE
            bytes.size >= 2 && bytes[0] == 0xfe.toByte() && bytes[1] == 0xff.toByte() -> 2 to Charsets.UTF_16BE
            else -> 0 to Charsets.UTF_8
        }
        val text = charset.newDecoder().onMalformedInput(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(bytes, offset, bytes.size - offset)).toString().removePrefix("\uFEFF")
        require(text.count { it == '\u0000' || (it.code < 32 && it !in "\n\r\t") } < maxOf(1, text.length / 100)) { "Il file sembra binario, non testo leggibile." }
        return text
    }
    private fun office(bytes: ByteArray): String {
        val parts = linkedMapOf<String, String>(); var total = 0
        ZipInputStream(ByteArrayInputStream(bytes)).use { zip ->
            var count = 0
            while (true) {
                val entry = zip.nextEntry ?: break
                require(++count <= 2048) { "Troppi elementi nel documento." }
                val selected = entry.name == "word/document.xml" || entry.name == "content.xml" || entry.name.startsWith("ppt/slides/slide") || entry.name == "xl/sharedStrings.xml"
                if (selected && entry.name.endsWith(".xml")) {
                    val data = zip.readBytesLimited(2 * 1024 * 1024); total += data.size; require(total <= 8 * 1024 * 1024) { "Documento decompresso troppo grande." }
                    parts[entry.name] = xmlText(String(data, Charsets.UTF_8))
                }
            }
        }
        require(parts.isNotEmpty()) { "Archivio non supportato: seleziona un documento Office o un file testuale." }
        return parts.toSortedMap().values.joinToString("\n\n")
    }
    private fun xmlText(xml: String): String {
        require(!xml.contains("<!DOCTYPE", true) && !xml.contains("<!ENTITY", true)) { "Entità XML esterne non ammesse." }
        val parser = Xml.newPullParser(); parser.setInput(xml.reader()); val text = StringBuilder()
        while (parser.eventType != XmlPullParser.END_DOCUMENT) {
            if (parser.eventType == XmlPullParser.TEXT) text.append(parser.text)
            if (parser.eventType == XmlPullParser.END_TAG && parser.name in listOf("p", "h", "table-row", "tr")) text.append('\n')
            if (parser.eventType == XmlPullParser.START_TAG && parser.name in listOf("tab", "br", "line-break")) text.append(if (parser.name == "tab") '\t' else '\n')
            parser.next()
        }
        return text.toString()
    }
    private fun rtf(text: String): String = text.replace(Regex("\\\\'([0-9a-fA-F]{2})")) { it.groupValues[1].toInt(16).toChar().toString() }.replace(Regex("\\\\(?:par|line)\\b ?"), "\n").replace(Regex("\\\\[a-zA-Z]+-?\\d* ?"), "").replace("{", "").replace("}", "")
}
