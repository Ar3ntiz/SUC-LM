import Foundation
@main struct RemoteTests {
 @MainActor static func main() async throws {
  precondition(RemoteInference.generationInactivityLimit>600 && RemoteInference.generationResourceLimit>600)
  var cachedTelemetry=RemoteTelemetry()
  cachedTelemetry.ingest(["suc_metrics":["prefill_s":0.5]])
  cachedTelemetry.ingest(["usage":["prompt_tokens":1000,"prompt_tokens_details":["cached_tokens":800]]])
  precondition(cachedTelemetry.payload()["prefill_tps"] as? Double==400)
  var cachedMetrics=Metrics();cachedMetrics.promptTokens=1000;cachedMetrics.cachedPromptTokens=800;cachedMetrics.prefillSeconds=0.5
  precondition(cachedMetrics.prefillRate==400)
  cachedTelemetry.ingest(["usage":["completion_tokens":21]])
  precondition(cachedTelemetry.payload()["prefill_tps"] as? Double==400)
  cachedTelemetry.ingest(["usage":["prompt_tokens_details":["cached_tokens":1000]]])
  precondition(cachedTelemetry.payload()["prefill_tps"]==nil)
  cachedMetrics.cachedPromptTokens=1000;precondition(cachedMetrics.prefillRate==nil)
  var llamaTelemetry=RemoteTelemetry()
  llamaTelemetry.ingest(["usage":["prompt_tokens":1000,"prompt_tokens_details":["cached_tokens":800]],"timings":["prompt_n":200,"prompt_ms":500]])
  precondition(llamaTelemetry.payload()["prefill_tps"] as? Double==400)
  llamaTelemetry.ingest(["timings":["prompt_per_second":375]])
  precondition(llamaTelemetry.payload()["prefill_tps"] as? Double==375)
  let p=Process(), pipe=Pipe();p.executableURL=URL(fileURLWithPath:"/usr/bin/python3");p.arguments=[CommandLine.arguments[1]];p.standardOutput=pipe
  try p.run();defer{p.terminate()}
  guard let port=Int(String(decoding:pipe.fileHandleForReading.availableData,as:UTF8.self).trimmingCharacters(in:.whitespacesAndNewlines)) else {throw MCPConnection.failure("Fixture HTTP non disponibile")}
  var config=RemoteConfiguration(endpoint:"http://127.0.0.1:\(port)/v1",model:"fixture")
  let client=RemoteInference(config,key:"test");defer{client.stop()}
  let models=try await client.models();precondition(models==["fixture","slow"])
  let health=await client.healthStatus();precondition(health["status"] as? String=="ok")
  let fallbackClient=RemoteInference(RemoteConfiguration(endpoint:"http://127.0.0.1:\(port)/fallback/v1",model:"fixture"),key:"test")
  let fallbackHealth=await fallbackClient.healthStatus();precondition(fallbackHealth["status"] as? String=="ok");fallbackClient.stop()
  for mode in ["router","effort","template","think","server","disabled"] {
   var controls=config;controls.model="control-"+mode;controls.sendExtensions=mode != "disabled"
   controls.telemetryProfile="router";controls.thinkingControl=mode=="router" ? "auto":mode
   let controlClient=RemoteInference(controls,key:"test");defer{controlClient.stop()}
   try await controlClient.generate(["messages":[["role":"user","content":"OK"]],"chat_template_kwargs":["enable_thinking":false]]){_ in}
   let restored=try JSONDecoder().decode(RemoteConfiguration.self,from:JSONEncoder().encode(controls));precondition(restored==controls)
  }
  var text="",usage=0
  let body:[String:Any]=["messages":[["role":"user","content":"Test"]],"top_k":1,"min_p":0.1,"seed":-1]
  try await client.generate(body){event in
   let payload=event["payload"] as? [String:Any] ?? [:]
   if event["event"] as? String=="delta"{text+=payload["text"] as? String ?? ""}
   if event["event"] as? String=="metrics"{usage=payload["completion_tokens"] as? Int ?? 0;precondition(payload["decode_s"]==nil)}
  }
  precondition(text=="Hello world" && usage==2)
  var progressConfig=config;progressConfig.model="progress-live";progressConfig.estimateLiveDecode=true
  let progressClient=RemoteInference(progressConfig,key:"test");defer{progressClient.stop()}
  var progressCount=0,liveCount=0,progressText="",lastProgressMetrics:[String:Any]=[:]
  try await progressClient.generate(body){event in
   let payload=event["payload"] as? [String:Any] ?? [:]
   if event["event"] as? String=="prefill_progress"{progressCount+=1;precondition(payload["cache"] as? Int==2048)}
   if event["event"] as? String=="delta"{progressText+=payload["text"] as? String ?? ""}
   if event["event"] as? String=="metrics" {
    lastProgressMetrics=payload
    if payload["completion_tokens"] as? Int==41 {
     liveCount+=1;precondition(payload["decode_tps"] as? Double==20);precondition(payload["decode_source"] as? String=="server")
    }
   }
  }
  precondition(progressCount==2 && liveCount>0 && progressText=="AB")
  precondition(lastProgressMetrics["decode_tps"] as? Double==20 && lastProgressMetrics["prefill_tps"] as? Double==117)
  print("Actual remote SSE progress and server-token live/final rate parity: PASS")
  var visionConfig=config;visionConfig.model="vision"
  let vision=RemoteInference(visionConfig,key:"test");defer{vision.stop()}
  let imageMessage=ChatMessage(role:"user",text:"Describe",attachments:[ChatAttachment(name:"test.jpg",text:"",originalBytes:3,imageData:Data([1,2,3]),imageMIME:"image/jpeg")])
  try await vision.generate(["messages":[["role":"user","content":imageMessage.remoteContent]]]){_ in}
  var timedConfig=config;timedConfig.model="timed"
  let timed=RemoteInference(timedConfig,key:"test");defer{timed.stop()};var timedPayload:[String:Any]=[:]
  try await timed.generate(body) { if $0["event"] as? String=="metrics" {timedPayload=$0["payload"] as? [String:Any] ?? [:]} }
  precondition(timedPayload["prefill_tps"] as? Double==5 && timedPayload["decode_tps"] as? Double==2)
  for profile in ["ds4","llama","lmstudio","custom"] {
   var options=config;options.model=profile;options.telemetryProfile=profile;options.liveTelemetry=true;options.customPrefillRatePath="rates.p";options.customDecodeRatePath="rates.d"
   let profileClient=RemoteInference(options,key:"test");defer{profileClient.stop()}
   var thought="",answer="",payload:[String:Any]=[:],metricEvents=0
   try await profileClient.generate(body){event in
    let value=event["payload"] as? [String:Any] ?? [:]
    switch event["event"] as? String {case "reasoning":thought+=value["text"] as? String ?? "";case "delta":answer+=value["text"] as? String ?? "";case "metrics":payload=value;metricEvents+=1;default:break}
   }
   precondition(thought=="Analizzo." && answer=="Risposta" && metricEvents>1)
   if profile=="llama"{precondition(payload["prefill_tps"] as? Double==200)}
   if profile=="ds4"{precondition(payload["prefill_tps"]==nil && payload["decode_source"] as? String=="client_estimate")}
   if profile=="lmstudio"{precondition(payload["decode_tps"] as? Double==42.5)}
   if profile=="custom"{precondition(payload["decode_tps"] as? Double==45)}
  }
  var burstConfig=config;burstConfig.model="burst"
  let burst=RemoteInference(burstConfig,key:"test");defer{burst.stop()}
  var burstText="",burstMetrics=0,burstDeltas=0,burstUsage=0,finishReason=""
  let burstStart=Date()
  try await burst.generate(body){event in
   let payload=event["payload"] as? [String:Any] ?? [:]
   switch event["event"] as? String {
   case "delta":burstText+=payload["text"] as? String ?? "";burstDeltas+=1
   case "metrics":burstMetrics+=1;burstUsage=payload["completion_tokens"] as? Int ?? 0
   case "finish_reason":finishReason=payload["reason"] as? String ?? ""
   default:break
   }
  }
  precondition(burstText==String(repeating:"x",count:1500) && burstUsage==1500 && finishReason=="stop")
  precondition(burstMetrics<100 && burstDeltas<200)
  print("Burst SSE: 1500 chunks/usage preserved; \(burstMetrics) metric events, \(burstDeltas) UI deltas, \(Date().timeIntervalSince(burstStart)) seconds")
  var quietOptions=config;quietOptions.liveTelemetry=false
  let quiet=RemoteInference(quietOptions,key:"test");defer{quiet.stop()};var quietMetrics=0
  try await quiet.generate(body){if $0["event"] as? String=="metrics"{quietMetrics+=1}}
  precondition(quietMetrics==1)
  for (model,expected) in [("eof","EOF"),("unicode","Città 🌍"),("json","JSON response")] {
   var options=config;options.model=model
   let reader=RemoteInference(options,key:"test");defer{reader.stop()};var received="",metrics:[String:Any]=[:]
   try await reader.generate(body){event in
    let payload=event["payload"] as? [String:Any] ?? [:]
    if event["event"] as? String=="delta"{received+=payload["text"] as? String ?? ""}
    if event["event"] as? String=="metrics"{metrics=payload}
   }
   precondition(received==expected)
   if model=="json"{precondition(metrics["prefill_tps"] as? Double==400)}
  }
  for model in ["oversized-line","oversized-total","oversized-json"] {
   var options=config;options.model=model
   let oversized=RemoteInference(options,key:"test");defer{oversized.stop()}
   do{try await oversized.generate(body){_ in};preconditionFailure("Response limit expected")}
   catch {precondition(error.localizedDescription.contains(model=="oversized-total" ? "16 MiB":"troppo grande"))}
  }
  let bad=RemoteInference(config,key:"");defer{bad.stop()}
  do{_ = try await bad.models();preconditionFailure("401 expected")}catch{precondition(error.localizedDescription.contains("401"))}
  config.model="slow";let slow=RemoteInference(config,key:"test")
  let work=Task{try await slow.generate(body){_ in}}
  try await Task.sleep(nanoseconds:100_000_000);work.cancel();slow.stop()
  do{try await work.value;preconditionFailure("Cancellation expected")}catch{}
  let url = try RemoteInference.endpoint("http://localhost:8080",path:"models"); precondition(url.path=="/v1/models")
  do{_ = try RemoteInference.endpoint("http://user:pass@localhost",path:"models");preconditionFailure("Credentials in URL") }catch{}
  print("RemoteTests: models/auth, SSE CRLF, deltas/usage, parameter compatibility, cancellation and endpoint validation passed")
 }
}
