import Foundation
@main struct RemoteTests {
 @MainActor static func main() async throws {
  precondition(RemoteInference.generationInactivityLimit>600 && RemoteInference.generationResourceLimit>600)
  let p=Process(), pipe=Pipe();p.executableURL=URL(fileURLWithPath:"/usr/bin/python3");p.arguments=[CommandLine.arguments[1]];p.standardOutput=pipe
  try p.run();defer{p.terminate()}
  guard let port=Int(String(decoding:pipe.fileHandleForReading.availableData,as:UTF8.self).trimmingCharacters(in:.whitespacesAndNewlines)) else {throw MCPConnection.failure("Fixture HTTP non disponibile")}
  var config=RemoteConfiguration(endpoint:"http://127.0.0.1:\(port)/v1",model:"fixture")
  let client=RemoteInference(config,key:"test");defer{client.stop()}
  let models=try await client.models();precondition(models==["fixture","slow"])
  let health=await client.healthStatus();precondition(health["status"] as? String=="ok")
  var text="",usage=0
  let body:[String:Any]=["messages":[["role":"user","content":"Test"]],"top_k":1,"min_p":0.1,"seed":-1]
  try await client.generate(body){event in
   let payload=event["payload"] as? [String:Any] ?? [:]
   if event["event"] as? String=="delta"{text+=payload["text"] as? String ?? ""}
   if event["event"] as? String=="metrics"{usage=payload["completion_tokens"] as? Int ?? 0;precondition(payload["decode_s"]==nil)}
  }
  precondition(text=="Hello world" && usage==2)
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
  var quietOptions=config;quietOptions.liveTelemetry=false
  let quiet=RemoteInference(quietOptions,key:"test");defer{quiet.stop()};var quietMetrics=0
  try await quiet.generate(body){if $0["event"] as? String=="metrics"{quietMetrics+=1}}
  precondition(quietMetrics==1)
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
