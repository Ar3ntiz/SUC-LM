package local.sucstudio.android

import android.app.Activity
import android.app.Instrumentation
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.view.ViewGroup
import android.webkit.WebView
import org.json.JSONObject
import org.json.JSONTokener
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class SmokeInstrumentation: Instrumentation() {
    private lateinit var web:WebView
    override fun onCreate(arguments:Bundle?) { super.onCreate(arguments);start() }
    override fun onStart() {
        val output=StringBuilder()
        try {
            val activity=startActivitySync(Intent(targetContext,MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            runOnMainSync { fun find(view:View):WebView? { if(view is WebView)return view;if(view is ViewGroup)for(i in 0 until view.childCount)find(view.getChildAt(i))?.let{return it};return null };web=find(activity.window.decorView)!! }
            waitFor("typeof window.nativeEvent==='function' && document.getElementById('chatTitle').textContent!=='SUC Studio'")
            command("server",JSONObject().put("value",JSONObject().put("url","http://127.0.0.1:9876/v1").put("model","fixture-chat").put("allowHttp",true)).put("apiKey","fixture-key"))
            command("settings",JSONObject().put("value",JSONObject().put("webEnabled",true).put("manualApproval",true)))
            eval("document.getElementById('composer').value='tool test';submit()")
            waitFor("!!document.querySelector('.approval')")
            check(!eval("document.getElementById('transcript').innerText.includes('suc_action')").toBoolean())
            eval("document.querySelector('.approval button').click()")
            waitFor("document.getElementById('transcript').innerText.includes('Web verificato')",90)
            waitFor("document.getElementById('stop').classList.contains('hidden')")
            check(eval("!!document.querySelector('.reasoning')").toBoolean());check(eval("!!document.querySelector('[data-save]')").toBoolean())
            check(eval("document.getElementById('stats').innerText.includes('35.0')").toBoolean())
            uiAutomation.takeScreenshot().let { image -> File(targetContext.filesDir,"qa-phone.png").outputStream().use { image.compress(android.graphics.Bitmap.CompressFormat.PNG,100,it) };image.recycle() }
            eval("document.getElementById('composer').value='captcha test';submit()")
            waitFor("!!document.querySelector('.approval')");eval("document.querySelector('.approval button').click()")
            waitFor("!!state.webStatus && state.webStatus.verification===true")
            val browserField=MainActivity::class.java.getDeclaredField("webBrowser").apply{isAccessible=true}
            val panel=browserField.get(activity) as WebToolBrowser
            val pageField=WebToolBrowser::class.java.getDeclaredField("browser").apply{isAccessible=true}
            val challengeBrowser=pageField.get(panel) as WebView
            runOnMainSync { check(challengeBrowser.width>200 && challengeBrowser.height>200) }
            uiAutomation.takeScreenshot().let { image -> File(targetContext.filesDir,"qa-captcha.png").outputStream().use { image.compress(android.graphics.Bitmap.CompressFormat.PNG,100,it) };image.recycle() }
            uiAutomation.executeShellCommand("input keyevent 4").close()
            Thread.sleep(500)
            check(eval("!document.getElementById('webBanner').classList.contains('hidden')").toBoolean())
            eval("document.querySelector('#webBanner button').click()")
            Thread.sleep(300);runOnMainSync{check(challengeBrowser.isShown)}
            evalIn(challengeBrowser,"document.getElementById('solve').click()")
            waitFor("!state.webStatus && document.getElementById('stop').classList.contains('hidden')",60)
            check(eval("document.getElementById('transcript').innerText.includes('Web verificato')").toBoolean())
            check(eval("state.messages.filter(m=>m.role==='assistant').slice(-1)[0].text.includes('Web verificato')").toBoolean())
            output.append("PASS: full-height interactive CAPTCHA fixture, back to chat, reopen page and dynamic verification without navigation; tool resumes automatically.\n")
            eval("document.getElementById('composer').value='captcha test';submit()")
            waitFor("!!document.querySelector('.approval')");eval("document.querySelector('.approval button').click()")
            waitFor("!!state.webStatus && state.webStatus.verification===true")
            val pastePanel=browserField.get(activity) as WebToolBrowser
            runOnMainSync {
                WebToolBrowser::class.java.getDeclaredMethod("pasteText").apply{isAccessible=true}.invoke(pastePanel)
            }
            Thread.sleep(300) // Let Dialog.onShow install the same listener a human would click.
            runOnMainSync {
                val popup=WebToolBrowser::class.java.getDeclaredField("pasteDialog").apply{isAccessible=true}.get(pastePanel) as android.app.AlertDialog
                fun input(view:View):android.widget.EditText? { if(view is android.widget.EditText)return view;if(view is ViewGroup)for(i in 0 until view.childCount)input(view.getChildAt(i))?.let{return it};return null }
                input(popup.window!!.decorView)!!.setText("Fixture page: testo fornito manualmente dopo verifica nel browser.")
                popup.getButton(android.app.AlertDialog.BUTTON_POSITIVE).performClick()
            }
            waitFor("!state.webStatus && document.getElementById('stop').classList.contains('hidden')",60)
            check(eval("state.messages.filter(m=>m.role==='assistant').slice(-1)[0].text.includes('Web verificato')").toBoolean())
            output.append("PASS: manual pasted page text becomes the actual tool result and the model completes its answer.\n")
            val count=eval("state.sessions.length")
            eval("editMessage(state.messages.find(m=>m.role==='user').id);document.getElementById('composer').value='edited test';submit()")
            waitFor("document.getElementById('transcript').innerText.includes('Risposta di prova')")
            check(eval("state.sessions.length")==count)
            waitFor("document.getElementById('stop').classList.contains('hidden')")
            eval("Array.from(document.querySelectorAll('.actions button')).find(b=>b.textContent==='Rigenera').click()");waitFor("!document.getElementById('stop').classList.contains('hidden')");waitFor("document.getElementById('stop').classList.contains('hidden')")
            check(eval("state.sessions.length")==count)
            eval("document.getElementById('composer').value='long test';submit()")
            waitFor("!document.getElementById('stop').classList.contains('hidden')");command("stop");waitFor("document.getElementById('stop').classList.contains('hidden')")
            eval("document.getElementById('composer').value='artifact test';submit()")
            waitFor("!!document.querySelector('.approval')");eval("document.querySelector('.approval button').click()")
            waitFor("document.getElementById('transcript').innerText.includes('Artifact creato')")
            waitFor("document.getElementById('stop').classList.contains('hidden')")
            check(eval("state.messages.some(m=>(m.artifacts||[]).some(a=>a.name==='hello.py'))").toBoolean())
            output.append("PASS: real Android WebView, SSE, inline approval, web extraction, thinking, metrics, artifacts, edit/reload same session and cancellation.\n")
            // Real attachment parsers on Android, including the legacy Word runtime.
            val attach=Attachments(targetContext)
            val text=File(targetContext.filesDir,"note.anything").apply{writeText("Testo leggibile con qualsiasi estensione.")}
            check(attach.read(Uri.fromFile(text)).getString("text").contains("Testo leggibile"))
            val doc=File(targetContext.filesDir,"SampleDoc.doc");context.assets.open("SampleDoc.doc").use{input->doc.outputStream().use{input.copyTo(it)}}
            check(attach.read(Uri.fromFile(doc)).getString("text").length>20)
            val docx=File(targetContext.filesDir,"example.docx");ZipOutputStream(docx.outputStream()).use{z->z.putNextEntry(ZipEntry("word/document.xml"));z.write("<w:document xmlns:w=\"urn:w\"><w:p><w:r><w:t>Documento Word XML</w:t></w:r></w:p></w:document>".toByteArray());z.closeEntry()}
            check(attach.read(Uri.fromFile(docx)).getString("text").contains("Documento Word XML"))
            val odt=File(targetContext.filesDir,"example.odt");ZipOutputStream(odt.outputStream()).use{z->z.putNextEntry(ZipEntry("content.xml"));z.write("<root><p>Documento OpenDocument</p></root>".toByteArray());z.closeEntry()}
            check(attach.read(Uri.fromFile(odt)).getString("text").contains("OpenDocument"))
            val image=File(targetContext.filesDir,"image.png");val bitmap=android.graphics.Bitmap.createBitmap(40,40,android.graphics.Bitmap.Config.ARGB_8888);image.outputStream().use{bitmap.compress(android.graphics.Bitmap.CompressFormat.PNG,100,it)};bitmap.recycle()
            val extracted=attach.read(Uri.fromFile(image));check(attach.hydrate(extracted).getString("dataUrl").startsWith("data:image/jpeg;base64,"))
            val pdf=File(targetContext.filesDir,"sample.pdf");context.assets.open("sample.pdf").use{input->pdf.outputStream().use{input.copyTo(it)}}
            check(attach.read(Uri.fromFile(pdf)).getString("text").contains("PDF Android"))
            val vault=Vault(targetContext);vault.write("test-only","secret-test");check(vault.read("test-only")=="secret-test");vault.write("test-only","");check(vault.read("test-only").isEmpty())
            output.append("PASS: DOC, DOCX, ODT, PDF, arbitrary text extensions, image hydration and Android Keystore round-trip.\n")
            val store=ArchiveStore(targetContext);Thread.sleep(1000);val saved=store.load();check(saved.getJSONArray("sessions").length()>0);check(!saved.toString().contains("fixture-key"));output.append("PASS: persistent sessions and no plaintext API key in archive.\n")
            eval("tab('chat')")
            val initial=eval("state.sessions.length").toInt();repeat(35){command("new")}
            waitFor("state.sessions.length>=${initial+35}")
            eval("document.getElementById('sidebar').classList.remove('closed')")
            check(eval("document.getElementById('sessions').scrollHeight>document.getElementById('sessions').clientHeight").toBoolean())
            output.append("PASS: session list remains scrollable after 35 new chats.\n")
            finish(Activity.RESULT_OK,Bundle().apply{putString("stream",output.toString())})
        } catch(e:Throwable){finish(Activity.RESULT_CANCELED,Bundle().apply{putString("stream",output.toString()+"FAIL: "+e.stackTraceToString())})}
    }
    private fun command(type:String,data:JSONObject=JSONObject()){data.put("type",type);eval("Studio.command(${JSONObject.quote(data.toString())})")}
    private fun eval(script:String):String = evalIn(web,script)
    private fun evalIn(target:WebView,script:String):String {val latch=CountDownLatch(1);var result="";runOnMainSync{target.evaluateJavascript(script){result=it;latch.countDown()}};check(latch.await(10,TimeUnit.SECONDS));return try{JSONTokener(result).nextValue().toString()}catch(_:Exception){result}}
    private fun waitFor(script:String,seconds:Int=30){repeat(seconds*5){if(eval(script)=="true")return;Thread.sleep(200)};error("Timeout UI: $script")}
}
