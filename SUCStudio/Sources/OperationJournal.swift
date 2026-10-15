import Foundation

struct OperationEntry: Identifiable, Equatable {
    var id:String
    var date=Date()
    var title:String
    var state:String
    var sessionID:UUID?
}
struct OperationJournal {
    static let capacity=200
    private(set) var entries:[OperationEntry]=[]
    mutating func update(id:String,title:String?=nil,state:String,sessionID:UUID?=nil) {
        if let i=entries.firstIndex(where:{$0.id==id}) {
            if ["Completata","Errore","Interrotta","Rifiutata"].contains(entries[i].state){return}
            if let title{entries[i].title=String(title.prefix(240))}
            entries[i].state=state
        } else {
            entries.append(OperationEntry(id:id,title:String((title ?? "Richiesta API").prefix(240)),state:state,sessionID:sessionID))
            if entries.count>Self.capacity{entries.removeFirst(entries.count-Self.capacity)}
        }
    }
}
struct ChatScrollPolicy {
    private(set) var following=true
    mutating func userScrolled(){following=false}
    mutating func resume(){following=true}
}
struct ChatScrollExtent:Equatable {
    var contentHeight:Double
    var viewportHeight:Double
    var viewportWidth:Double
}
