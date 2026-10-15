import Foundation

struct ModelIOLogEntry:Identifiable,Equatable {
    let id=UUID()
    var date=Date()
    var requestID:String
    var direction:String
    var text:String
}

struct ModelIOLog {
    static let maximumEntries=160
    static let maximumEntryCharacters=64*1024
    static let maximumTotalCharacters=256*1024
    private(set) var entries:[ModelIOLogEntry]=[]
    mutating func append(requestID:String,direction:String,text:String,coalesce:Bool=false) {
        guard !text.isEmpty else{return}
        if coalesce,let last=entries.indices.last,entries[last].requestID==requestID,entries[last].direction==direction {
            entries[last].text=String((entries[last].text+text).suffix(Self.maximumEntryCharacters));entries[last].date=Date()
        } else {
            entries.append(ModelIOLogEntry(requestID:requestID,direction:direction,text:String(text.prefix(Self.maximumEntryCharacters))))
        }
        while entries.count>Self.maximumEntries || entries.reduce(0,{$0+$1.text.count})>Self.maximumTotalCharacters {entries.removeFirst()}
    }
    mutating func clear(){entries.removeAll(keepingCapacity:true)}
    static func request(_ body:[String:Any])->String {
        func safe(_ value:Any)->Any {
            if let string=value as? String {
                if string.hasPrefix("data:image"),let comma=string.firstIndex(of:",") {return "[immagine data URL · \(string.distance(from:comma,to:string.endIndex)) caratteri]"}
                return string
            }
            if let array=value as? [Any] {return array.map(safe)}
            if let object=value as? [String:Any] {return object.mapValues(safe)}
            return value
        }
        guard let data=try? JSONSerialization.data(withJSONObject:safe(body),options:[.prettyPrinted,.sortedKeys]) else{return "[input non serializzabile]"}
        let text=String(decoding:data,as:UTF8.self)
        return text.count>Self.maximumEntryCharacters ? String(text.prefix(Self.maximumEntryCharacters))+"\n[… input completo omesso dal registro; richiesta inviata senza tagli …]":text
    }
}
