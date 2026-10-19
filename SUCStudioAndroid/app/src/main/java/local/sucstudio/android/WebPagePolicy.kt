package local.sucstudio.android

import org.json.JSONObject

/** A verification page is never returned as a successful tool result. */
object WebPagePolicy {
    fun challenge(page: JSONObject) = page.optBoolean("challenge")
    fun canRead(page: JSONObject, automatic: Boolean): Boolean = !page.optBoolean("wait") && !challenge(page) &&
        page.optString("text").isNotBlank() && (!automatic || (page.optBoolean("ready") && page.optString("text").length > 150))
    // ES5: this script also runs on older system WebViews, independently of the UI transpiler.
    val extraction = """
        (function(){
          var body=document.body, text=(body?body.innerText:'').slice(0,48000);
          var consent=document.querySelector('#L2AGLb');
          if(consent){consent.click();return {wait:true};}
          function visible(node){var r=node.getBoundingClientRect();return r.width>0&&r.height>0&&getComputedStyle(node).visibility!=='hidden'&&getComputedStyle(node).display!=='none';}
          var nodes=document.querySelectorAll('#captcha-form,form[action*="/sorry/"],.g-recaptcha,[data-sitekey],iframe[src*="recaptcha"],iframe[src*="hcaptcha"],#challenge-stage,#challenge-running,#cf-challenge-running,#challenge-form');
          var challenge=/^https?:\/\/([^/]*\.)?google\.[^/]+\/sorry(?:\/|\?)/i.test(location.href);
          for(var i=0;i<nodes.length;i++){if(visible(nodes[i]))challenge=true;}
          var links=[], all=document.querySelectorAll('a[href]');
          for(var n=0;n<all.length&&links.length<40;n++){var a=all[n];if(a.innerText.trim()&&/^https?:/.test(a.href))links.push({title:a.innerText.slice(0,150),url:a.href});}
          return {text:text,url:location.href,title:document.title,challenge:challenge,ready:document.readyState==='complete',links:links};
        })()
    """.trimIndent()
}
