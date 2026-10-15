import Foundation
@main struct PresentationTests {
 static func main() throws {
    let parts=ThinkingParts.split(" <think>Ragiono</think>Risposta")
    assert(parts.reasoning=="Ragiono" && parts.answer=="Risposta" && !parts.active)
    let partial=ThinkingParts.split("<think>Ragiono</thi");assert(partial.reasoning=="Ragiono" && partial.active)
    assert(ThinkingParts.split("testo</think>finale",implicit:true).answer=="finale")
    assert(ThinkingParts.split("risposta",separate:"reasoning").reasoning=="reasoning")
    assert(ThinkingParts.split("```html\n<think>example</think>\n```").reasoning.isEmpty)
    let fence=MessagePresentation.codeFence("```inside```",language:"ht\nml")
    assert(fence.hasPrefix("````html\n") && ChatMarkdown.parse(fence).first?.text=="```inside```")
    let html="<!doctype html><html>"+String(repeating:"<div><a href=\"https://example.com/long-url?q=1&amp;b=2\">test</a></div>\n",count:5000)+"</html>"
    assert(MessagePresentation.isLong(html));let start=Date();let document=PreparedMarkdown.make(html)
    assert(document.blocks.count==1 && document.blocks[0].text.count<=8000 && document.formatted.isEmpty)
    print("Long HTML bounded preparation:",html.utf8.count,"bytes in",Date().timeIntervalSince(start),"seconds")
    var llamaConfig=RemoteConfiguration();llamaConfig.telemetryProfile="llama"
    var llama=RemoteTelemetry(configuration:llamaConfig);llama.ingest(["prompt_progress":["total":300,"cache":100,"processed":200,"time_ms":500]])
    assert(llama.payload()["prefill_tps"] as? Double==200)
    var lm=RemoteTelemetry();lm.ingest(["stats":["tokens_per_second":42.5,"time_to_first_token":10]])
    assert(lm.payload()["decode_tps"] as? Double==42.5 && lm.payload()["prefill_tps"]==nil)
    var config=RemoteConfiguration();config.estimateLiveDecode=true;config.telemetryProfile="ds4"
    var ds4=RemoteTelemetry(configuration:config);ds4.observeText("some reasoning",at:Date(timeIntervalSince1970:1));ds4.observeText(" then an answer",at:Date(timeIntervalSince1970:2))
    assert(ds4.payload()["decode_source"] as? String=="client_estimate_live" && ds4.payload()["prefill_tps"]==nil)
    ds4.ingest(["usage":["prompt_tokens":100,"completion_tokens":11]]);assert(ds4.payload(final:true)["decode_tps"] as? Double==10)
    config.telemetryProfile="custom";config.customPrefillRatePath="metrics.p";config.customDecodeRatePath="metrics.d"
    var custom=RemoteTelemetry(configuration:config);custom.ingest(["metrics":["p":123,"d":45]])
    assert(custom.payload()["prefill_tps"] as? Double==123 && custom.payload()["decode_tps"] as? Double==45)
    let message=ChatMessage(reasoning:"pensiero",implicitThinking:false,role:"assistant",text:"risposta")
    let restored=try JSONDecoder().decode(ChatMessage.self,from:JSONEncoder().encode(message));assert(restored.reasoning=="pensiero" && restored.modelContent=="risposta")
    print("Presentation tests passed: thinking/partial tags, code fences, bounded HTML, live profiles, optional estimates, persistence")
 }
}
