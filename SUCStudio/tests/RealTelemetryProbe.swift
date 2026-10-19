import Foundation
// Minimal configuration contract for exercising the production telemetry parser without the UI.
struct RemoteConfiguration {
    var telemetryProfile:String?
    var estimateLiveDecode:Bool?
    var customPrefillRatePath:String?
    var customDecodeRatePath:String?
}
@main struct RealTelemetryProbe {
    static func main() throws {
        let data=try Data(contentsOf:URL(fileURLWithPath:CommandLine.arguments[1]))
        let source=try JSONSerialization.jsonObject(with:data) as! [String:Any]
        let usage=source["server_usage"] as! [String:Any]
        let metrics=source["server_metrics"] as! [String:Any]
        var telemetry=RemoteTelemetry()
        telemetry.ingest(["usage":usage,"suc_metrics":metrics])
        let result=telemetry.payload(final:true)
        let expected=Double((usage["prompt_tokens"] as! Int)-((usage["prompt_tokens_details"] as! [String:Any])["cached_tokens"] as! Int))/(metrics["prefill_s"] as! Double)
        precondition(abs((result["prefill_tps"] as! Double)-expected)<1e-9)
        precondition(result["prefill_tokens"] as! Int == 117)
        print(String(data:try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]),encoding:.utf8)!)
    }
}
