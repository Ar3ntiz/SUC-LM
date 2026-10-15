import Foundation
@main struct TelemetryTests {
    static func main() {
        var t=RemoteTelemetry();t.ingest(["usage":["prompt_tokens":100,"completion_tokens":11],"suc_metrics":["prefill_s":2.0,"decode_s":5.0]])
        assert(t.payload()["prefill_tps"] as? Double==50 && t.payload()["decode_tps"] as? Double==2)
        assert(t.payload()["decode_source"] as? String=="server")
        var estimate=RemoteTelemetry();estimate.ingest(["usage":["prompt_tokens":100,"completion_tokens":11]])
        estimate.observeText("a",at:Date(timeIntervalSince1970:1));estimate.observeText("b",at:Date(timeIntervalSince1970:3))
        assert(estimate.payload()["decode_tps"]==nil);assert(estimate.payload(final:true)["decode_tps"] as? Double==5)
        assert(estimate.payload(final:true)["prefill_tps"]==nil && estimate.payload(final:true)["decode_source"] as? String=="client_estimate")
        var llama=RemoteTelemetry();llama.ingest(["timings":["prompt_n":20,"predicted_n":10,"prompt_ms":1000.0,"predicted_ms":500.0]])
        assert(llama.payload()["prefill_tps"] as? Double==20 && llama.payload()["decode_tps"] as? Double==20)
        assert(RemoteTelemetry.positive(Double.infinity)==nil && RemoteTelemetry.positive(-2)==nil && RemoteTelemetry.positive(true)==nil)
        print("Telemetry tests passed: server phases, llama timings, explicit estimate and unavailable prefill")
    }
}
