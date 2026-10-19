import Foundation
import CryptoKit

@main struct StreamLatencyProbe {
    @MainActor static func main() async throws {
        let fixture=Process(),pipe=Pipe()
        fixture.executableURL=URL(fileURLWithPath:"/usr/bin/python3")
        fixture.arguments=[CommandLine.arguments[1]];fixture.standardOutput=pipe
        try fixture.run();defer{fixture.terminate()}
        let port=Int(String(decoding:pipe.fileHandleForReading.availableData,as:UTF8.self).trimmingCharacters(in:.whitespacesAndNewlines))!
        let base="http://127.0.0.1:\(port)/v1"
        var results:[[String:Any]]=[]
        for count in [4000,16000,64000] {
            for mode in ["raw-http","remote-client","remote-client-io-log"] {
                let run=UUID().uuidString
                let body:[String:Any]=["messages":[["role":"user","content":"Synthetic stream latency probe"]],"diagnostic_chunks":count,"diagnostic_run":run]
                var text="",callbacks=0,log=ModelIOLog(),maximumHeartbeat:Double=0
                let start=Date()
                let heartbeat=Task { @MainActor in
                    var last=Date()
                    while !Task.isCancelled {
                        do {try await Task.sleep(nanoseconds:10_000_000)}catch{return}
                        let now=Date();maximumHeartbeat=max(maximumHeartbeat,now.timeIntervalSince(last));last=now
                    }
                }
                var bytes=0
                if mode=="raw-http" {
                    var request=URLRequest(url:URL(string:base+"/chat/completions")!);request.httpMethod="POST"
                    request.httpBody=try JSONSerialization.data(withJSONObject:body)
                    let(data,_)=try await URLSession.shared.data(for:request);bytes=data.count
                } else {
                    let client=RemoteInference(RemoteConfiguration(endpoint:base,model:"synthetic"),key:"")
                    defer{client.stop()}
                    try await client.generate(body){ event in
                        guard event["event"] as? String=="delta",let payload=event["payload"] as? [String:Any],let delta=payload["text"] as? String else{return}
                        text+=delta;callbacks+=1
                        if mode=="remote-client-io-log" {log.append(requestID:run,direction:"OUTPUT",text:delta,coalesce:true)}
                    }
                    precondition(text==String(repeating:"riga **test**\n",count:count))
                }
                let end=Date();heartbeat.cancel()
                let(data,_)=try await URLSession.shared.data(from:URL(string:"http://127.0.0.1:\(port)/stats?run=\(run)")!)
                let server=try JSONSerialization.jsonObject(with:data) as! [String:Any]
                let serverEnd=server["send_end_unix"] as! Double
                var result:[String:Any]=["chunks":count,"mode":mode,"wall_s":end.timeIntervalSince(start),"server":server,"client_after_server_send_s":end.timeIntervalSince1970-serverEnd,"maximum_main_actor_heartbeat_s":maximumHeartbeat,"callbacks":callbacks]
                if mode=="raw-http" {result["received_sse_bytes"]=bytes}
                else {result["text_bytes"]=text.utf8.count;result["text_sha256"]=SHA256.hash(data:Data(text.utf8)).map{String(format:"%02x",$0)}.joined()}
                results.append(result)
                let line=try JSONSerialization.data(withJSONObject:result,options:[.sortedKeys]);print(String(decoding:line,as:UTF8.self));fflush(stdout)
            }
        }
        let output=try JSONSerialization.data(withJSONObject:["kind":"synthetic_actual_remote_client_latency","version":StudioPlatform.version,"scope":"Actual RemoteInference and optional ModelIOLog; no SwiftUI window, no GPU/model, no network beyond loopback","results":results],options:[.prettyPrinted,.sortedKeys])
        try output.write(to:URL(fileURLWithPath:CommandLine.arguments[2]))
    }
}
