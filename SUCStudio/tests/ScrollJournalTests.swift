import Foundation
@main struct ScrollJournalTests {
 @MainActor static func main() async throws {
    var policy=ChatScrollPolicy();assert(policy.following);policy.userScrolled();assert(!policy.following);policy.resume();assert(policy.following)
    var io=ModelIOLog();io.append(requestID:"r",direction:"INPUT",text:"prompt");io.append(requestID:"r",direction:"OUTPUT",text:"a",coalesce:true);io.append(requestID:"r",direction:"OUTPUT",text:"b",coalesce:true)
    assert(io.entries.count==2 && io.entries.last?.text=="ab")
    let encoded=ModelIOLog.request(["messages":[["content":"data:image/png;base64,SECRET"]]])
    assert(encoded.contains("immagine data URL") && !encoded.contains("SECRET"))
    var journal=OperationJournal();journal.update(id:"a",title:"read_file",state:"In coda");journal.update(id:"a",state:"In corso");assert(journal.entries.count==1)
    journal.update(id:"a",state:"Errore");journal.update(id:"a",state:"Completata");assert(journal.entries[0].state=="Errore")
    for i in 0..<250 { journal.update(id:String(i),title:"Op",state:"Completata") };assert(journal.entries.count==200 && journal.entries.first?.id=="50")
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:root)}
    let store=StudioStore(storageURL:root.appendingPathComponent("sessions.json"));assert(!store.archive.settings.ioLogEnabled && store.modelIOLog.entries.isEmpty)
    store.setModelIOLogEnabled(true);assert(store.archive.settings.ioLogEnabled)
    store.modelState="ready";store.loadedModel="fixture";store.mutateSelected{$0.draft="hello"};store.sendMessage()
    let id=store.requests.keys.first!;assert(store.operationJournal.entries.last?.state=="In coda" && store.modelIOLog.entries.last?.direction=="INPUT")
    store.receive(["event":"started","id":id,"payload":[:]]);assert(store.operationJournal.entries.last?.state=="In corso")
    store.receive(["event":"delta","id":id,"payload":["text":"risposta"]]);assert(store.modelIOLog.entries.last?.text=="risposta")
    store.cancelSelected();store.receive(["event":"finished","id":id,"payload":[:]]);store.receive(["event":"idle","id":id,"payload":[:]])
    assert(store.operationJournal.entries.last?.state=="Interrotta")
    store.recordAction(store.selected!.id,ActionRecord(name:"read_file",details:"not logged",state:"completed",result:"not logged"))
    assert(store.operationJournal.entries.last?.title=="read_file" && store.operationJournal.entries.last?.sessionID==store.selected!.id)
    store.setModelIOLogEnabled(false);assert(store.modelIOLog.entries.isEmpty)
    store.receive(["event":"model_output","id":"api-disabled","payload":["text":"must not be retained"]]);assert(store.modelIOLog.entries.isEmpty)
    print("Scroll policy / operation journal tests passed: pause/resume, bound, states, cancellation, request and tool routing")
 }
}
