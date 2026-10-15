import Foundation
@main struct AgentReportTests {
 @MainActor static func main() async throws {
    var settings=AppSettings();assert(settings.actionLimit==12);settings.maxConsecutiveActions=0;assert(settings.actionLimit==1);settings.maxConsecutiveActions=500;assert(settings.actionLimit==100);settings.maxConsecutiveActions=3
    let restored=try JSONDecoder().decode(AppSettings.self,from:JSONEncoder().encode(settings));assert(restored.actionLimit==3)
    var session=ChatSession();session.agentEnabled=true;session.workspacePath="/tmp";assert(PromptComposer.compose(settings:settings,session:session).contains("Massimo 3 azioni"))
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
    let store=StudioStore(storageURL:root.appendingPathComponent("archive.json"));store.modelState="ready";store.loadedModel="fixture";store.archive.settings.maxConsecutiveActions=1
    store.mutateSelected{$0.agentEnabled=true;$0.workspacePath=root.path;$0.approvalMode="automatic";$0.draft="Crea un file"};store.sendMessage()
    func answer(_ text:String) {let id=store.requests.keys.first!;store.receive(["event":"delta","id":id,"payload":["text":text]]);store.receive(["event":"finished","id":id,"payload":[:]])}
    answer("{\"suc_action\":{\"name\":\"write_file\",\"arguments\":{\"path\":\"a.txt\",\"content\":\"hello\"}}}")
    for _ in 0..<500 { if !store.requests.isEmpty {break};try await Task.sleep(nanoseconds:10_000_000) }
    let written=try String(contentsOf:root.appendingPathComponent("a.txt"),encoding:.utf8);assert(written=="hello")
    assert(store.selected!.messages.contains{$0.kind=="tool_result"})
    answer("{\"suc_action\":{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.txt\"}}}")
    assert(store.requests.isEmpty && store.selected!.actions?.count==1)
    let visible=store.selected!.messages.filter{$0.visibleInChat(agent:true)}
    assert(!visible.contains{$0.text.contains("suc_action") || $0.isOperational});assert(visible.contains{$0.kind=="action_notice"});assert(visible.last?.kind=="action_report")
    store.mutateSelected{$0.draft="Leggi un file inesistente"};store.sendMessage()
    answer("{\"suc_action\":{\"name\":\"read_file\",\"arguments\":{\"path\":\"missing.txt\"}}}")
    for _ in 0..<500 {if !store.requests.isEmpty{break};try await Task.sleep(nanoseconds:10_000_000)}
    assert(store.selected!.messages.contains{$0.kind=="action_notice" && $0.text.contains("Errore")})
    answer("Il file non esiste.")
    assert(store.selected!.messages.last?.text.contains("1 non completate")==true)
    let partial=ChatMessage(role:"assistant",text:"{\"suc_",state:"generating");assert(!partial.visibleInChat(agent:true));assert(partial.visibleInChat(agent:false))
    store.mutateSelected{$0.draft="Richiesta da annullare"};store.sendMessage();let cancelled=store.requests.keys.first!
    store.cancelSelected();store.receive(["event":"delta","id":cancelled,"payload":["text":"{\"suc_action\":{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.txt\"}}}"]]);store.receive(["event":"finished","id":cancelled,"payload":[:]])
    assert(!store.agentRuns.contains(store.selected!.id) && store.requests.isEmpty)
    print("Agent report tests passed: settings/clamping, dynamic prompt/limit, hidden operational messages, immediate error and final summary")
 }
}
