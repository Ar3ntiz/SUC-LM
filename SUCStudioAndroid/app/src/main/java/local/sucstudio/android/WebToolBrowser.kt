package local.sucstudio.android

import android.app.Activity
import android.app.AlertDialog
import android.app.Dialog
import android.content.Intent
import android.graphics.Color
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.view.ViewGroup
import android.view.WindowManager
import android.webkit.CookieManager
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import org.json.JSONObject

/** Foreground-only browser, deliberately without JavascriptInterface or app/API credentials. */
class WebToolBrowser(
    private val activity: Activity,
    val sessionId: String,
    private val initialUrl: String,
    search: Boolean,
    private val status: (JSONObject) -> Unit,
    private val finish: (JSONObject?) -> Unit
) {
    private val handler = Handler(Looper.getMainLooper())
    private var closed = false
    private var extracting = false
    private var verification = false
    private var external = false
    private var navigationVersion = 0
    private var pasteDialog: AlertDialog? = null
    private val root = LinearLayout(activity).apply { orientation = LinearLayout.VERTICAL; setBackgroundColor(Color.rgb(23,24,34)) }
    private val title = TextView(activity).apply { text = if(search) tr("Ricerca web") else tr("Lettura pagina"); textSize=20f; setTextColor(Color.WHITE);setPadding(16,16,16,4) }
    private val address = TextView(activity).apply { text=initialUrl; textSize=12f;maxLines=2;setTextColor(Color.LTGRAY);setPadding(16,0,16,8) }
    private val message = TextView(activity).apply { text=tr("Caricamento della pagina…");textSize=14f;setTextColor(Color.rgb(210,204,255));setPadding(16,8,16,8) }
    private val browser = WebView(activity).apply {
        settings.javaScriptEnabled=true; settings.domStorageEnabled=true; settings.allowFileAccess=false;settings.allowContentAccess=false
        settings.setSupportMultipleWindows(false);settings.builtInZoomControls=true;settings.displayZoomControls=false
        settings.useWideViewPort=true;settings.loadWithOverviewMode=true
        isFocusable=true;isFocusableInTouchMode=true
        CookieManager.getInstance().setAcceptCookie(true)
        // Challenge widgets can use cross-site frames; confined to this browser, never the chat renderer.
        CookieManager.getInstance().setAcceptThirdPartyCookies(this,true)
    }
    private val readButton=Button(activity).apply { text=tr("Usa questa pagina");setOnClickListener{extract(false)} }
    private val dialog = object:Dialog(activity,android.R.style.Theme_DeviceDefault_NoActionBar) {
        @Deprecated("Native back button controls the browser, not the underlying chat")
        override fun onBackPressed() { if(browser.canGoBack())browser.goBack() else hide() }
    }
    private val poll=object:Runnable { override fun run(){if(closed)return;extract(true);handler.postDelayed(this,1000)} }
    init {
        root.addView(title);root.addView(address)
        fun row(vararg buttons:Button){ val row=LinearLayout(activity).apply { orientation=LinearLayout.HORIZONTAL };buttons.forEach{row.addView(it,LinearLayout.LayoutParams(0,ViewGroup.LayoutParams.WRAP_CONTENT,1f))};root.addView(row) }
        row(Button(activity).apply{text=tr("Torna alla chat");setOnClickListener{hide()}},Button(activity).apply{text=tr("Browser esterno");setOnClickListener{openExternal()}})
        root.addView(message)
        root.addView(browser,LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,0,1f))
        row(readButton,Button(activity).apply{text=tr("Ricarica");setOnClickListener{external=false;verification=false;browser.reload()}},Button(activity).apply{text=tr("Incolla testo");setOnClickListener{pasteText()}})
        root.addView(Button(activity).apply{text=tr("Annulla ricerca");setOnClickListener{complete(null)}})
        dialog.setContentView(root,ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,ViewGroup.LayoutParams.MATCH_PARENT))
        dialog.window?.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE)
        browser.webChromeClient=object:WebChromeClient(){override fun onProgressChanged(view:WebView,progress:Int){if(!verification&&!external)message.text=tr("Caricamento pagina $progress%")}}
        browser.webViewClient=object:WebViewClient(){
            override fun shouldOverrideUrlLoading(view:WebView,request:WebResourceRequest):Boolean=request.url.scheme !in listOf("https","http")
            override fun onPageStarted(view:WebView,url:String,favicon:android.graphics.Bitmap?){if(closed)return;navigationVersion++;address.text=url;publish(tr("Caricamento pagina"))}
            override fun onReceivedError(view:WebView,request:WebResourceRequest,error:android.webkit.WebResourceError){if(!closed&&request.isForMainFrame){message.text=tr("Pagina non caricata: usa Ricarica o il browser esterno.");publish(tr("Errore pagina: apri il browser per riprovare"))}}
            override fun onPageFinished(view:WebView,url:String){if(closed)return;address.text=url;CookieManager.getInstance().flush();extract(true)}
        }
        show();browser.loadUrl(initialUrl);handler.postDelayed(poll,1000)
    }
    fun show(){if(closed)return;dialog.show();dialog.window?.setLayout(ViewGroup.LayoutParams.MATCH_PARENT,ViewGroup.LayoutParams.MATCH_PARENT);publish(if(verification)"Verifica CAPTCHA richiesta" else tr("Lettura web in corso"))}
    private fun hide(){if(closed)return;dialog.hide();publish(if(verification)tr("Verifica CAPTCHA richiesta: apri la pagina") else "Ricerca in corso: apri la pagina")}
    fun payload():JSONObject=JSONObject().put("session",sessionId).put("url",browser.url?:initialUrl).put("verification",verification).put("external",external)
    private fun publish(text:String){if(closed)return;status(payload().put("text",text))}
    private fun extract(automatic:Boolean){
        if(closed||extracting||(automatic&&external))return
        extracting=true;val version=navigationVersion
        browser.evaluateJavascript(WebPagePolicy.extraction){raw->
            extracting=false;if(closed||version!=navigationVersion)return@evaluateJavascript
            try{
                val page=JSONObject(raw);val before=verification;verification=WebPagePolicy.challenge(page)
                if(verification){message.text=tr("Completa la verifica qui sotto. La ricerca attende senza scadenza. Se il sito non accetta questo browser, aprilo nel browser esterno e incolla il testo.");readButton.isEnabled=false;if(!before)publish(tr("Verifica CAPTCHA richiesta: apri la pagina"));return@evaluateJavascript}
                readButton.isEnabled=true
                if(WebPagePolicy.canRead(page,automatic))complete(page)
                else if(!automatic)message.text=tr("Pagina non ancora leggibile: completa il caricamento o incolla il testo letto nel browser.")
            }catch(_:Exception){if(!automatic)message.text=tr("Impossibile leggere questa pagina. Puoi usare il browser esterno e incollare il testo.")}
        }
    }
    private fun openExternal(){
        val url=browser.url?:initialUrl
        try{Protocol.resource(url,true);activity.startActivity(Intent(Intent.ACTION_VIEW,Uri.parse(url)));external=true
            message.text=tr("Risolvi la verifica nel browser esterno, poi copia il testo utile. Torna qui e scegli Incolla testo: i cookie del browser esterno non vengono importati automaticamente.")
            publish("Browser esterno aperto: torna e incolla il testo")
        }catch(_:Exception){message.text=tr("Nessun browser esterno disponibile. Usa la pagina integrata.")}
    }
    private fun pasteText(){
        val editor=EditText(activity).apply{hint=tr("Incolla il testo della pagina dopo la verifica");minLines=6;gravity=android.view.Gravity.TOP;setTextColor(Color.BLACK);setBackgroundColor(Color.WHITE)}
        val popup=AlertDialog.Builder(activity).setTitle(tr("Testo letto nel browser")).setMessage(tr("Incolla il contenuto utile della pagina, non il testo del CAPTCHA. Verrà inviato al modello.")).setView(editor).setNegativeButton(tr("Indietro"),null).setPositiveButton(tr("Usa testo"),null).create()
        pasteDialog=popup
        popup.setOnShowListener{editor.requestFocus();popup.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener{
            val text=editor.text.toString().trim()
            if(text.isBlank()||text.length>48000){editor.error="Inserisci da 1 a 48.000 caratteri.";return@setOnClickListener}
            popup.dismiss();complete(JSONObject().put("text",text).put("url",browser.url?:initialUrl).put("title","Testo fornito dall’utente").put("source","manual_paste").put("challenge",false))
        }};popup.show()
    }
    private fun complete(page:JSONObject?){if(closed)return;close();finish(page)}
    fun close(){if(closed)return;closed=true;handler.removeCallbacks(poll);pasteDialog?.dismiss();dialog.dismiss();browser.stopLoading();root.removeView(browser);browser.destroy();CookieManager.getInstance().flush()}
}
