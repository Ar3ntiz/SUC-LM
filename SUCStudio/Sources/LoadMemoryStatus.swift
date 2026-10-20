import Foundation

struct LoadMemoryStatus {
    let resident:Double
    let initial:Double
    let total:Double
    let footprint:Double?
    let metal:Double?
    let initialMetal:Double?
    init?(_ payload:[String:Any]) {
        guard let resident=payload["resident_bytes"] as? Double,let total=payload["target_bytes"] as? Double,
              resident.isFinite,total.isFinite,total>0,resident>=0,resident<=total else{return nil}
        self.resident=resident;self.total=total
        let initial=payload["initial_resident_bytes"] as? Double ?? 0
        self.initial=initial.isFinite ? min(total,max(0,initial)):0
        let footprint=payload["footprint_bytes"] as? Double
        self.footprint=footprint.flatMap{$0.isFinite && $0>=0 ? $0:nil}
        metal=(payload["metal_allocated_bytes"] as? Double).flatMap{$0.isFinite && $0>=0 ? $0:nil}
        initialMetal=(payload["initial_metal_bytes"] as? Double).flatMap{$0.isFinite && $0>=0 ? $0:nil}
    }
    var fraction:Double{resident/total}
    var label:String{String(format:"Pagine del modello in memoria: %.2f / %.2f GiB",resident/1_073_741_824,total/1_073_741_824)}
    var metalLabel:String?{metal.map{String(format:"Metal allocata: %.2f GiB · iniziale %.2f GiB",$0/1_073_741_824,(initialMetal ?? 0)/1_073_741_824)}}
    var detail:String{String(format:"Già residenti all’avvio: %.2f GiB",initial/1_073_741_824)}
}
