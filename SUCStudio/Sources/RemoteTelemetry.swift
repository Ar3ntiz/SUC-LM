import Foundation
import CoreFoundation

/// Server measurements take precedence. A client estimate never claims to measure prefill.
struct RemoteTelemetry {
    var configuration:RemoteConfiguration?
    var characters=0
    var prompt=0, completion=0
    var prefillSeconds:Double?, decodeSeconds:Double?, prefillRate:Double?, decodeRate:Double?
    var firstText:Date?, lastText:Date?
    mutating func observeText(_ text:String,at date:Date=Date()) { guard !text.isEmpty else{return};if firstText==nil{firstText=date};lastText=date;characters+=text.count }
    static func positive(_ value:Any?)->Double? { guard let n=value as? NSNumber,CFGetTypeID(n) != CFBooleanGetTypeID(),n.doubleValue.isFinite,n.doubleValue>0 else{return nil};return n.doubleValue }
    static func count(_ value:Any?)->Int? { guard let n=value as? NSNumber,CFGetTypeID(n) != CFBooleanGetTypeID(),n.doubleValue.isFinite,n.doubleValue>=0,n.doubleValue<=1_000_000_000,n.doubleValue.rounded(.down)==n.doubleValue else{return nil};return n.intValue }
    mutating func ingest(_ value:[String:Any]) {
        if let usage=value["usage"] as? [String:Any] {
            if let n=Self.count(usage["prompt_tokens"]),n>=0{prompt=n}
            if let n=Self.count(usage["completion_tokens"]),n>=0{completion=n}
        }
        if let suc=value["suc_metrics"] as? [String:Any] {
            prefillSeconds=Self.positive(suc["prefill_s"]) ?? prefillSeconds
            decodeSeconds=Self.positive(suc["decode_s"]) ?? decodeSeconds
        }
        if let stats=value["stats"] as? [String:Any] {
            decodeRate=Self.positive(stats["tokens_per_second"]) ?? decodeRate
            prefillRate=Self.positive(stats["prefill_tokens_per_second"]) ?? prefillRate
        }
        if let progress=value["prompt_progress"] as? [String:Any],let processed=Self.count(progress["processed"]),let ms=Self.positive(progress["time_ms"]) {
            let cached=Self.count(progress["cache"]) ?? 0
            if processed>cached {prefillRate=Double(processed-cached)/(ms/1000)}
            if let total=Self.count(progress["total"]){prompt=total}
        }
        if configuration?.telemetryProfile == "custom" {
            prefillRate=Self.positive(Self.field(configuration?.customPrefillRatePath,in:value)) ?? prefillRate
            decodeRate=Self.positive(Self.field(configuration?.customDecodeRatePath,in:value)) ?? decodeRate
        }
        if let timings=value["timings"] as? [String:Any] {
            prefillRate=Self.positive(timings["prompt_per_second"]) ?? prefillRate
            decodeRate=Self.positive(timings["predicted_per_second"]) ?? decodeRate
            if let ms=Self.positive(timings["prompt_ms"]){prefillSeconds=ms/1000}
            // llama-style servers count all predicted tokens in their reported decode timing.
            if decodeRate==nil,let ms=Self.positive(timings["predicted_ms"]),let n=Self.positive(timings["predicted_n"]){decodeRate=n/(ms/1000)}
            if prompt==0,let n=Self.count(timings["prompt_n"]),n>0{prompt=n}
            if completion==0,let n=Self.count(timings["predicted_n"]),n>0{completion=n}
        }
    }
    static func field(_ path:String?,in value:[String:Any])->Any? {
        guard let path,!path.isEmpty,path.count<200 else{return nil}
        var current:Any=value
        for key in path.split(separator:".").prefix(12){guard let next=(current as? [String:Any])?[String(key)] else{return nil};current=next}
        return current
    }
    func payload(final:Bool=false)->[String:Any] {
        var result:[String:Any]=["prompt_tokens":prompt,"completion_tokens":completion,"remote":true]
        if let seconds=prefillSeconds{result["prefill_s"]=seconds}
        if let seconds=decodeSeconds{result["decode_s"]=seconds}
        if let rate=prefillRate ?? prefillSeconds.flatMap({prompt>0 ? Double(prompt)/$0:nil}) {result["prefill_tps"]=rate;result["prefill_source"]="server"}
        if let rate=decodeRate ?? decodeSeconds.flatMap({completion>1 ? Double(completion-1)/$0:nil}) {result["decode_tps"]=rate;result["decode_source"]="server"}
        else if final,completion>1,let first=firstText,let last=lastText,last.timeIntervalSince(first)>0.001 {
            result["decode_tps"]=Double(completion-1)/last.timeIntervalSince(first);result["decode_source"]="client_estimate"
        }
        if result["decode_tps"]==nil,configuration?.estimateLiveDecode==true,let first=firstText,let last=lastText,last.timeIntervalSince(first)>0.2,characters>4 {
            result["decode_tps"]=Double(characters/4)/last.timeIntervalSince(first);result["decode_source"]="client_estimate_live"
        }
        return result
    }
}
