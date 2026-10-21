package local.sucstudio.android

import org.junit.Assert.*
import org.junit.Test
import org.json.JSONObject
import org.json.JSONArray
import java.io.ByteArrayOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference

class ProtocolTest {
    @Test fun languageDefaultsAndArchiveRoundTrip() {
        val fresh=ArchiveStore.normalize(JSONObject())
        assertEquals("en", fresh.getJSONObject("settings").getString("language"))
        fresh.getJSONObject("settings").put("language", "it")
        val restored=ArchiveStore.normalize(JSONObject(fresh.toString()))
        assertEquals("it",restored.getJSONObject("settings").getString("language"))
        restored.getJSONObject("settings").put("language", "fr")
        assertEquals("en",ArchiveStore.normalize(restored).getJSONObject("settings").getString("language"))
    }

    @Test fun challengeNeverBecomesToolResultAndDynamicResolutionResumes() {
        val challenge=JSONObject().put("challenge",true).put("ready",true).put("text","CAPTCHA verification " .repeat(20))
        assertFalse(WebPagePolicy.canRead(challenge,true));assertFalse(WebPagePolicy.canRead(challenge,false))
        challenge.put("challenge",false);assertTrue(WebPagePolicy.canRead(challenge,true))
        assertFalse(WebPagePolicy.canRead(JSONObject().put("wait",true).put("text","pending"),false))
        assertTrue(WebPagePolicy.canRead(JSONObject().put("text","Short useful page"),false))
    }

    @Test fun endpointsRejectSecretsAndRequireExplicitHttp() {
        assertEquals("https://example.org/v1/models", Protocol.endpoint("https://example.org/", "models"))
        assertEquals("http://127.0.0.1:9090/custom/chat/completions", Protocol.endpoint("http://127.0.0.1:9090/custom/", "chat/completions", true))
        for (url in listOf("http://example.org", "https://user:key@example.org", "file:///etc/passwd", "https://example.org?key=secret", "https://example.org/#frag")) assertThrows(IllegalArgumentException::class.java) { Protocol.endpoint(url,"models") }
    }
    @Test fun streamingMultilineDoneAndUsage() {
        val input = ": comment\r\ndata: {\r\ndata: \"choices\":[{\"delta\":{\"content\":\"ciao\",\"reasoning_content\":\"pensiero\"}}]}\r\n\r\ndata: {\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3},\"timings\":{\"prompt_per_second\":100,\"predicted_per_second\":20}}\n\ndata: [DONE]\n\ndata: invalid\n\n"
        val accumulator = StreamAccumulator(); Protocol.sse(input.reader().buffered()) { accumulator.accept(it) }
        assertEquals("ciao",accumulator.content);assertEquals("pensiero",accumulator.reasoning);assertEquals(100.0,accumulator.prefillRate!!,0.001);assertEquals(20.0,accumulator.decodeRate!!,0.001);assertFalse(accumulator.metrics().getBoolean("decodeEstimated"))
    }
    @Test fun telemetrySUCStudioLMStudioAndLlamaTimings() {
        val suc=StreamAccumulator();suc.accept(JSONObject("{\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":21},\"suc_metrics\":{\"prefill_s\":0.5,\"decode_s\":1}}"))
        assertEquals(200.0,suc.metrics().getDouble("prefill"),0.001);assertEquals(20.0,suc.metrics().getDouble("decode"),0.001);assertFalse(suc.metrics().getBoolean("prefillEstimated"))
        val lm=StreamAccumulator();lm.accept(JSONObject("{\"stats\":{\"tokens_per_second\":40,\"prefill_tokens_per_second\":140}}"));assertEquals(140.0,lm.metrics().getDouble("prefill"),0.001);assertEquals(40.0,lm.metrics().getDouble("decode"),0.001)
        val llama=StreamAccumulator();llama.accept(JSONObject("{\"timings\":{\"prompt_n\":50,\"prompt_ms\":100,\"predicted_n\":10,\"predicted_ms\":1000}}"));assertEquals(500.0,llama.metrics().getDouble("prefill"),0.001);assertEquals(10.0,llama.metrics().getDouble("decode"),0.001)
    }
    @Test fun sucCachedUsageCountsOnlyEvaluatedTokensAcrossSseFrames() {
        val accumulator = StreamAccumulator()
        val events = listOf(
            JSONObject().put("choices", JSONArray().put(JSONObject().put("delta", JSONObject().put("content", "Ciao")))),
            JSONObject().put("suc_metrics", JSONObject().put("prefill_s", 0.5).put("decode_s", 1)),
            JSONObject().put("usage", JSONObject().put("prompt_tokens", 1000).put("completion_tokens", 21)
                .put("prompt_tokens_details", JSONObject().put("cached_tokens", 800)))
        )
        val stream = events.joinToString("") { "data: $it\n\n" } + "data: [DONE]\n\n"
        Protocol.sse(stream.reader().buffered()) { accumulator.accept(it) }
        val result = accumulator.metrics()
        assertEquals(1000, result.getInt("promptTokens"))
        assertEquals(800, result.getInt("cachedTokens"))
        assertEquals(200, result.getInt("uncachedPromptTokens"))
        assertEquals(400.0, result.getDouble("prefill"), 0.001)
        assertEquals(20.0, result.getDouble("decode"), 0.001)
        assertFalse(result.getBoolean("prefillEstimated"))
    }
    @Test fun cachedUsageSurvivesPartialFramesAndLatePromptCount() {
        val accumulator = StreamAccumulator()
        accumulator.accept(JSONObject("""{"usage":{"prompt_tokens_details":{"cached_tokens":800}}}"""))
        accumulator.accept(JSONObject("""{"usage":{"prompt_tokens":1000},"suc_metrics":{"prefill_s":0.5}}"""))
        accumulator.accept(JSONObject("""{"usage":{"completion_tokens":21}}"""))
        assertEquals(800, accumulator.metrics().getInt("cachedTokens"))
        assertEquals(400.0, accumulator.metrics().getDouble("prefill"), 0.001)
    }
    @Test fun llamaPromptNIsAlreadyUncachedAndExplicitRateWins() {
        val accumulator = StreamAccumulator()
        accumulator.accept(JSONObject("""{"usage":{"prompt_tokens":1000,"prompt_tokens_details":{"cached_tokens":800}},"timings":{"prompt_n":200,"prompt_ms":500}}"""))
        assertEquals(400.0, accumulator.metrics().getDouble("prefill"), 0.001)
        accumulator.accept(JSONObject("""{"stats":{"prefill_tokens_per_second":375}}"""))
        assertEquals(375.0, accumulator.metrics().getDouble("prefill"), 0.001)
        assertFalse(accumulator.metrics().getBoolean("prefillEstimated"))
    }
    @Test fun fullCacheAndInvalidCountersNeverInventPrefillSpeed() {
        for (cached in listOf(1000, 2000)) {
            val accumulator = StreamAccumulator()
            accumulator.accept(JSONObject().put("usage", JSONObject().put("prompt_tokens", 1000)
                .put("prompt_tokens_details", JSONObject().put("cached_tokens", cached)))
                .put("suc_metrics", JSONObject().put("prefill_s", 0.5)))
            accumulator.accept(JSONObject("""{"choices":[{"delta":{"content":"Ciao"}}]}"""))
            assertEquals(1000, accumulator.metrics().getInt("cachedTokens"))
            assertTrue(accumulator.metrics().isNull("prefill"))
            assertFalse(accumulator.metrics().getBoolean("prefillEstimated"))
        }
        val negative = StreamAccumulator()
        negative.accept(JSONObject("""{"usage":{"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":-20}},"suc_metrics":{"prefill_s":0.5}}"""))
        assertEquals(0, negative.metrics().getInt("cachedTokens"))
        assertEquals(200.0, negative.metrics().getDouble("prefill"), 0.001)
    }
    @Test fun missingCacheFieldsAndProgressKeepServerCompatibility() {
        val empty = StreamAccumulator().metrics()
        assertTrue(empty.isNull("prefill")); assertFalse(empty.getBoolean("prefillEstimated"))
        val progress = StreamAccumulator()
        progress.accept(JSONObject("""{"prompt_progress":{"processed":1000,"cache":800,"time_ms":500}}"""))
        assertEquals(400.0, progress.metrics().getDouble("prefill"), 0.001)
        val estimated = StreamAccumulator()
        estimated.accept(JSONObject("""{"usage":{"prompt_tokens":1000,"prompt_tokens_details":{"cached_tokens":800}},"choices":[{"delta":{"content":"Ciao"}}]}"""))
        assertTrue(estimated.metrics().getDouble("prefill").isFinite())
        assertTrue(estimated.metrics().getBoolean("prefillEstimated"))
        assertEquals(200, estimated.metrics().getInt("uncachedPromptTokens"))
    }
    @Test fun actionsStayHiddenForEveryStreamingPrefix() {
        for (text in listOf("Procedo.\n{\"suc_action\":{\"name\":\"web_search\",\"arguments\":{\"query\":\"test\"}}}", "Procedo.\n```json\n{\"suc_action\":{\"name\":\"web_search\",\"arguments\":{\"query\":\"test\"}}}\n```", "Procedo.\n{\n  \"suc_action\":{\"name\":\"web_search\",\"arguments\":{}}}")) {
            for (end in 1..text.length) { val visible=Protocol.visible(text.take(end),false);assertFalse("leak at $end: $visible",visible.contains("suc_action"));assertFalse(visible.contains("web_search")) }
            assertEquals("web_search",Protocol.action(text)!!.name);assertEquals("Procedo.",Protocol.visible(text,true))
        }
        assertNull(Protocol.action("Un esempio inline: {\"suc_action\":{\"name\":\"web_search\"}}"))
        assertNull(Protocol.action("{\"suc_action\":{\"name\":\"web_search\"}} trailing"))
        assertEquals("{\"answer\":42}",Protocol.visible("{\"answer\":42}",true))
    }
    @Test fun requestKeepsImagesAndPromptLayersWithoutInactiveSkillBodies() {
        val settings=JSONObject("{\"server\":{\"model\":\"vision\"},\"systemPrompt\":\"GLOBAL\",\"top_k\":0,\"seed\":-1}")
        val session=JSONObject("{\"systemPrompt\":\"SESSION\",\"messages\":[{\"role\":\"user\",\"text\":\"hello\",\"attachments\":[{\"dataUrl\":\"data:image/jpeg;base64,AAAA\"},{\"dataUrl\":\"data:image/jpeg;base64,BBBB\"},{\"name\":\"doc\",\"text\":\"document\"}]}]}")
        val request=Protocol.request(settings,session,"CATALOG")
        val messages=request.getJSONArray("messages");assertEquals("GLOBAL\n\nSESSION\n\nCATALOG",messages.getJSONObject(0).getString("content"));val parts=messages.getJSONObject(1).getJSONArray("content");assertEquals(3,parts.length());assertEquals("text",parts.getJSONObject(0).getString("type"));assertTrue(parts.getJSONObject(0).getString("text").contains("document"));assertEquals("image_url",parts.getJSONObject(1).getString("type"));assertEquals(4000,request.getInt("max_tokens"));assertFalse(request.has("top_k"));assertFalse(request.has("seed"))
    }
    private fun zip(entries:Map<String,String>):ByteArray {val bytes=ByteArrayOutputStream();ZipOutputStream(bytes).use { z -> entries.forEach { (path,text)->z.putNextEntry(ZipEntry(path));z.write(text.toByteArray());z.closeEntry() } };return bytes.toByteArray()}
    @Test fun combinedPluginsStayDisabledAndRejectExecutableComponents() {
        val entries=mapOf("plugin.json" to "{\"name\":\"demo\",\"version\":\"1\"}","skills/review/SKILL.md" to "---\nname: review\ndescription: Review prose\n---\nReview carefully.","skills/review/references/style.md" to "Style guide", "mcp.json" to "{\"mcpServers\":{\"remote\":{\"type\":\"http\",\"url\":\"https://example.org/mcp\"}}}")
        val plugin=PluginPackage.parse("demo",zip(entries));assertFalse(plugin.getBoolean("enabled"));assertEquals(1,plugin.getJSONArray("skills").length());assertEquals(1,plugin.getJSONArray("mcp").length());assertEquals("Style guide",plugin.getJSONArray("skills").getJSONObject(0).getJSONObject("resources").getString("references/style.md"))
        assertThrows(IllegalArgumentException::class.java){PluginPackage.parse("bad",zip(entries+mapOf("../escape" to "bad")))}
        assertThrows(IllegalArgumentException::class.java){PluginPackage.parse("bad",zip(entries+mapOf("hooks/hooks.json" to "{}")))}
        assertThrows(IllegalArgumentException::class.java){PluginPackage.parse("bad",zip(entries+mapOf("mcp.json" to "{\"mcpServers\":{\"shell\":{\"command\":\"sh\"}}}")))}
    }
    private class Fixture : AutoCloseable {
        val socket=ServerSocket(0,8,java.net.InetAddress.getByName("127.0.0.1"));val port=socket.localPort
        val failure=AtomicReference<Throwable?>();val pool=Executors.newCachedThreadPool()
        var handler: (String,Map<String,String>,String)->Triple<Int,Map<String,String>,String> = { _,_,_->Triple(404,emptyMap(),"") }
        init { pool.execute { while(!socket.isClosed) try { val client=socket.accept();pool.execute { serve(client) } } catch (_:Exception){} } }
        private fun serve(client:Socket) { client.use { try {
            val input=it.getInputStream().bufferedReader();val line=input.readLine();val path=line.split(' ')[1];val headers=mutableMapOf<String,String>()
            while(true){val h=input.readLine()?:break;if(h.isEmpty())break;val pair=h.split(':',limit=2);headers[pair[0].lowercase()]=pair.getOrElse(1){""}.trim()}
            val length=headers["content-length"]?.toInt()?:0;val chars=CharArray(length);var count=0;while(count<length){val n=input.read(chars,count,length-count);if(n<0)break;count+=n}
            val (status,responseHeaders,body)=handler(path,headers,String(chars));val bytes=body.toByteArray();val header="HTTP/1.1 $status Test\r\nContent-Length: ${bytes.size}\r\nConnection: close\r\n"+responseHeaders.entries.joinToString(""){h->"${h.key}: ${h.value}\r\n"}+"\r\n"
            it.getOutputStream().write(header.toByteArray());it.getOutputStream().write(bytes);it.getOutputStream().flush()
        }catch(e:Throwable){failure.set(e)} } }
        override fun close(){socket.close();pool.shutdownNow();failure.get()?.let{throw AssertionError(it)}}
    }
    @Test fun actualHttpMcpSessionAndRedirectGuards() {
        Fixture().use { server ->
            var forbidden=0
            server.handler={path,headers,text->when(path){
                "/v1/models"->{assertEquals("Bearer test",headers["authorization"]);Triple(200,emptyMap(),"{\"data\":[{\"id\":\"model-a\"}]}")}
                "/redirect"->Triple(302,mapOf("Location" to "http://127.0.0.1:${server.port}/forbidden"),"")
                "/forbidden"->{forbidden++;Triple(200,emptyMap(),"{}")}
                "/mcp"->{val request=JSONObject(text);val method=request.getString("method");if(method=="notifications/initialized")Triple(202,emptyMap(),"")else{
                    val result=when(method){"initialize"->JSONObject("{\"protocolVersion\":\"2025-03-26\",\"capabilities\":{}}")
                        "tools/list"->{assertEquals("session-1",headers["mcp-session-id"]);JSONObject("{\"tools\":[{\"name\":\"echo\",\"inputSchema\":{\"type\":\"object\"}}]}")}
                        else->JSONObject("{\"content\":[{\"type\":\"text\",\"text\":\"ok\"}]}")}
                    val response=JSONObject().put("jsonrpc","2.0").put("id",request.getInt("id")).put("result",result)
                    if(method=="tools/call")Triple(200,mapOf("Content-Type" to "text/event-stream"),"event: message\ndata: $response\n\n")else Triple(200,mapOf("Mcp-Session-Id" to "session-1"),response.toString())
                }}
                else->Triple(404,emptyMap(),"")
            }}
            val base="http://127.0.0.1:${server.port}"
            assertEquals(listOf("model-a"),Http().models(JSONObject().put("url",base).put("allowHttp",true),"test"))
            assertThrows(IllegalStateException::class.java){Http().exchange("$base/redirect")};assertEquals(0,forbidden)
            val client=McpClient(JSONObject().put("url","$base/mcp").put("allowHttp",true),"",Cancellation());assertEquals("echo",client.tools().single().getString("name"));assertEquals("ok",client.call("echo",JSONObject()).getJSONArray("content").getJSONObject(0).getString("text"))
        }
    }
}
