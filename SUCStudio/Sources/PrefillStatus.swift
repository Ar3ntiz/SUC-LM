import Foundation

struct PrefillStatus:Equatable {
    let processed:Int
    let total:Int
    let cached:Int?
    init?(_ payload:[String:Any]) {
        guard let total=RemoteTelemetry.count(payload["total"]),total>0,
              let processed=RemoteTelemetry.count(payload["processed"]),processed<=total else{return nil}
        let cached=RemoteTelemetry.count(payload["cache"])
        if let cached,cached>processed{return nil}
        self.total=total;self.processed=processed;self.cached=cached
    }
    var fraction:Double {
        let base=cached ?? 0
        return total>base ? Double(processed-base)/Double(total-base):1
    }
    var detail:String {
        if let cached { return "\(processed-cached) / \(total-cached) token nuovi · \(cached) riusati" }
        return "\(processed) / \(total) token"
    }
}
