import AppKit
import Foundation
import WebKit

@main struct Release146Tests {
    @MainActor static func wait(_ label:String,_ condition:() -> Bool) async throws {
        for _ in 0..<900 {if condition(){return};try await Task.sleep(nanoseconds:50_000_000)}
        preconditionFailure("Timed out: "+label)
    }
    @MainActor static func main() async throws {
        _ = NSApplication.shared; NSApp.setActivationPolicy(.accessory)
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer{try? FileManager.default.removeItem(at:root)}
        let wire="{\"suc_action\":{\"name\":\"web_search\",\"arguments\":{\"query\":\"prezzo pellet\"}}}"
        for value in [wire,"```json\n"+wire+"\n```","<think>ragiono</think>\n"+wire,"<analysis>x</analysis>\n```json\n"+wire+"\n```"] {
            precondition(WorkspaceAction.parse(value)?.name=="web_search")
            precondition(!ChatMessage(role:"assistant",text:value).visibleInChat(agent:false,web:false))
        }
        for value in [wire,"```json\n"+wire+"\n```","```JSON\n"+wire+"\n```"] {
            for count in 1...value.count {
                precondition(!ChatMessage(role:"assistant",text:String(value.prefix(count)),state:"generating").visibleInChat(agent:false,web:true))
            }
        }
        precondition(!ChatMessage(implicitThinking:true,role:"assistant",text:"Ragiono</think>"+wire).visibleInChat(agent:true))
        precondition(ChatMessage(role:"assistant",text:"{\"answer\":42}").visibleInChat(agent:true))
        precondition(ChatMessage(role:"assistant",text:"```python\nprint(1)\n```",state:"generating").visibleInChat(agent:true))
        precondition(WorkspaceAction.parse("Esempio: "+wire)==nil)
        let intro="Ho capito, vuoi che io verifichi ora i dati più recenti su questi portali per darti un quadro aggiornato e preciso. Procedo con le ricerche mirate."
        for value in [intro+"\n\n"+wire,intro+"\n\n```json\n"+wire+"\n```","<think>pianifico</think>\n"+intro+"\n"+wire] {
            precondition(WorkspaceAction.parse(value)?.name=="web_search")
            precondition(!ChatMessage(role:"assistant",text:value).visibleInChat(agent:false,web:true))
        }
        for suffix in [wire,"```json\n"+wire+"\n```"] {
            for count in 1...suffix.count {
                let partial=intro+"\n\n"+String(suffix.prefix(count))
                precondition(!ChatMessage(role:"assistant",text:partial,state:"generating").visibleInChat(agent:false,web:true))
            }
        }
        precondition(WorkspaceAction.parse("```python\n"+wire+"\n```")==nil)
        precondition(WorkspaceAction.parse("````json\n"+wire+"\n````")==nil)
        precondition(WorkspaceAction.parse(intro+"\n"+wire+"\n"+wire)==nil)
        precondition(WorkspaceAction.isActionEnvelope(intro+"\n"+wire+"\n"+wire))
        precondition(ChatMessage(role:"assistant",text:intro+"\n{\"answer\":42}").visibleInChat(agent:true))
        precondition(ChatSession().maxTokens==4000)
        var saved=ChatSession();saved.maxTokens=777
        let decodedSaved=try JSONDecoder().decode(ChatSession.self,from:JSONEncoder().encode(saved));precondition(decodedSaved.maxTokens==777)
        let legacyReport=ChatMessage(role:"assistant",text:"**Riepilogo attività**\n\n1 azioni completate.\n\n1 ricerca web\n\nDettagli nel registro azioni della sessione.",kind:"action_report")
        precondition(!legacyReport.compactActivityText.contains("\n") && !legacyReport.compactActivityText.contains("Dettagli nel"))

        let iso=ISO8601DateFormatter();var calendar=Calendar(identifier:.gregorian);calendar.timeZone=TimeZone(secondsFromGMT:0)!
        var repeatRoutine=StudioRoutine();repeatRoutine.repeatEnabled=true;repeatRoutine.repeatCount=1;repeatRoutine.repeatUnit="months"
        let january=iso.date(from:"2027-01-31T12:00:00Z")!
        precondition(repeatRoutine.nextOccurrence(after:january,calendar:calendar)==iso.date(from:"2027-02-28T12:00:00Z"))
        for (unit,seconds) in [("minutes",120.0),("hours",7200.0),("days",172800.0)] {
            repeatRoutine.repeatUnit=unit;repeatRoutine.repeatCount=2
            precondition(repeatRoutine.nextOccurrence(after:january,calendar:calendar)!.timeIntervalSince(january)==seconds)
        }
        calendar.timeZone=TimeZone(identifier:"Europe/Rome")!;repeatRoutine.repeatCount=1;repeatRoutine.repeatUnit="days"
        let beforeDST=iso.date(from:"2026-03-28T11:00:00Z")!
        precondition(repeatRoutine.nextOccurrence(after:beforeDST,calendar:calendar)!.timeIntervalSince(beforeDST)==23*3600)
        var legacy=try JSONSerialization.jsonObject(with:JSONEncoder().encode(StudioRoutine())) as! [String:Any]
        for key in ["repeatEnabled","repeatCount","repeatUnit"] {legacy.removeValue(forKey:key)}
        legacy["intervalMinutes"]=90
        let restored=try JSONDecoder().decode(StudioRoutine.self,from:JSONSerialization.data(withJSONObject:legacy))
        precondition(restored.isRepeating && restored.recurrenceCount==90 && restored.recurrenceUnit=="minutes")
        var oneShot=StudioRoutine();oneShot.enabled=true;oneShot.advance(after:Date());precondition(!oneShot.enabled)

        let skill=SkillDocument.parse("---\nname: Verifiche\ndescription: >\n  Descrizione per verificare documenti\n  in modo accurato.\n---\nCORPO_SKILL_RISERVATO",fallbackName:"fallback")
        precondition(skill.name=="Verifiche" && skill.description=="Descrizione per verificare documenti in modo accurato." && skill.instructions=="CORPO_SKILL_RISERVATO")
        var active=skill;active.enabled=true
        let inactive=StudioSkill(name:"Spenta",description:"NON_ELENCARE",instructions:"NON_CARICARE",enabled:false)
        var settings=AppSettings();settings.skills=[active,inactive]
        let catalog=PromptComposer.compose(settings:settings,session:ChatSession())
        precondition(catalog.contains("Descrizione per verificare documenti") && catalog.contains("load_skill") && !catalog.contains("CORPO_SKILL_RISERVATO") && !catalog.contains("NON_ELENCARE"))
        let loaded=try SkillDocument.load(WorkspaceAction(name:"load_skill",arguments:["id":active.id.uuidString]),skills:[active])
        precondition(loaded.contains("CORPO_SKILL_RISERVATO"))
        do{_ = try SkillDocument.load(WorkspaceAction(name:"load_skill",arguments:["id":inactive.id.uuidString]),skills:[inactive]);preconditionFailure("Disabled skill loaded")}catch{}
        var oldSkill=try JSONSerialization.jsonObject(with:JSONEncoder().encode(active)) as! [String:Any];oldSkill.removeValue(forKey:"description")
        let decodedSkill=try JSONDecoder().decode(StudioSkill.self,from:JSONSerialization.data(withJSONObject:oldSkill));precondition(!decodedSkill.effectiveDescription.isEmpty)
        print("Release146Tests: tool JSON streaming/fences/thinking, normal JSON/code, summaries, defaults, recurrence/DST/months/legacy and skill catalog/import passed")

        let fixture=Process(),pipe=Pipe();fixture.executableURL=URL(fileURLWithPath:"/usr/bin/python3");fixture.arguments=[CommandLine.arguments[1]];fixture.standardOutput=pipe
        try fixture.run();defer{fixture.terminate()}
        let port=Int(String(decoding:pipe.fileHandleForReading.availableData,as:UTF8.self).trimmingCharacters(in:.whitespacesAndNewlines))!
        let store=StudioStore(storageURL:root.appendingPathComponent("sessions.json"));store.schedulingEnabled=false
        store.archive.settings.chatBackend="remote";store.archive.settings.remote=RemoteConfiguration(endpoint:"http://127.0.0.1:\(port)/v1",model:"routine-web")
        store.webBrowser=WebBrowser{WKWebsiteDataStore.nonPersistent()};store.webBrowser?.searchEndpoint="http://127.0.0.1:\(port)/results?q="
        let target=store.selected!.id
        store.mutateSelected{$0.approvalMode="automatic";$0.draft="Bozza da preservare"}
        store.newSession();let foreground=store.selected!.id
        var routine=StudioRoutine();routine.name="Ricerca programmata";routine.prompt="Cerca sul web";routine.sessionID=target;routine.backend="remote";routine.enabled=true;routine.nextRun=Date(timeIntervalSince1970:1)
        try store.saveRoutine(routine)
        store.schedulingEnabled=true;store.tickRoutines();store.schedulingEnabled=false
        precondition(store.archive.settings.routines![0].enabled==false)
        try await wait("scheduled web tool result") {store.archive.sessions.first{$0.id==target}!.messages.contains{$0.text=="Risultato web verificato dalla routine." && $0.state=="complete"}}
        store.tickRoutines()
        let result=store.archive.sessions.first{$0.id==target}!
        precondition(store.archive.selected==foreground && result.draft=="Bozza da preservare")
        precondition(result.hasUnreadMessages==true && store.webBrowserAllowed && store.approvals.isEmpty)
        precondition(result.actions?.last?.state=="completed" && result.messages.last?.text=="1 ricerca web")
        precondition(!result.messages.filter{$0.visibleInChat(agent:false,web:true)}.contains{$0.text.contains("suc_action") || $0.isOperational})
        precondition(store.archive.settings.routines![0].history.last?.state=="completed")
        let json=try JSONEncoder().encode(store.archive)
        let decodedArchive=try JSONDecoder().decode(Archive.self,from:json);precondition(decodedArchive.sessions.first{$0.id==target}!.hasUnreadMessages==true)
        let previousUpdate=result.updated;store.selectSession(target)
        precondition(store.selected!.hasUnreadMessages==false && store.selected!.updated==previousUpdate)
        store.schedulingEnabled=true;store.tickRoutines(now:Date().addingTimeInterval(3600));store.schedulingEnabled=false
        precondition(store.requests.isEmpty && store.activeRoutines.isEmpty)
        print("Release146Tests: actual scheduled SSE → hidden web action → WebKit localhost search → real tool result → final answer; draft/selection preserved, one-shot not replayed and unread persisted/cleared")

        store.selectSession(foreground)
        for mode in ["routine-commentary","routine-commentary-fenced"] {
            store.archive.settings.remote?.model=mode
            var next=StudioRoutine();next.name=mode;next.prompt="Cerca prezzi GPL";next.sessionID=target;next.backend="remote";next.enabled=true;next.nextRun=Date(timeIntervalSince1970:1)
            try store.saveRoutine(next)
            let before=store.archive.sessions.first{$0.id==target}!.messages.count
            store.schedulingEnabled=true;store.tickRoutines();store.schedulingEnabled=false
            try await wait("commentary scheduled tool round trip") {store.archive.sessions.first{$0.id==target}!.messages.dropFirst(before).contains{$0.text=="Risultato web verificato dalla routine." && $0.state=="complete"}}
            store.tickRoutines()
            let updated=store.archive.sessions.first{$0.id==target}!
            precondition(store.archive.settings.routines?.first{$0.id==next.id}?.history.last?.state=="completed")
            precondition(updated.actions?.last?.state=="completed" && updated.hasUnreadMessages==true)
            precondition(!updated.messages.dropFirst(before).filter{$0.visibleInChat(agent:false,web:true)}.contains{$0.text.contains("suc_action")})
            precondition(store.archive.selected==foreground && updated.draft=="Bozza da preservare")
        }
        print("Scheduled web: commentary + raw/fenced JSON stays hidden and executes through WebKit to a real result and final answer; no premature completion.")
        store.selectSession(foreground);store.archive.settings.skills=[active];store.archive.settings.remote?.model="skill"
        store.mutateSelected{$0.allowWeb=false;$0.approvalMode="manual";$0.draft="Applica la skill"};store.sendMessage()
        try await wait("lazy skill load") {store.selected!.messages.contains{$0.text=="Skill applicata correttamente." && $0.state=="complete"}}
        precondition(store.approvals.isEmpty && store.selected!.actions?.last?.name=="load_skill")
        precondition(store.selected!.messages.filter{$0.kind=="tool_result"}.contains{$0.text.contains("CORPO_SKILL_RISERVATO")})

        store.archive.settings.remote?.model="routine-web";store.newSession();let manual=store.selected!.id
        store.mutateSelected{$0.approvalMode="manual"};store.selectSession(foreground)
        var manualRoutine=StudioRoutine();manualRoutine.name="Manuale";manualRoutine.prompt="Cerca";manualRoutine.sessionID=manual;manualRoutine.backend="remote";manualRoutine.enabled=true;manualRoutine.nextRun=Date(timeIntervalSince1970:1)
        try store.saveRoutine(manualRoutine);store.schedulingEnabled=true;store.tickRoutines();store.schedulingEnabled=false
        try await wait("inline manual approval") {!store.approvals.isEmpty}
        precondition(store.approvals[0].sessionID==manual && store.archive.sessions.first{$0.id==manual}!.hasUnreadMessages==true)
        precondition(store.archive.settings.routines?.last?.status=="Attende approvazione nella chat")
        store.resolveApproval(store.approvals[0].id,allowed:false)
        try await wait("denied routine cleanup") {!store.agentRuns.contains(manual)}
        store.tickRoutines();precondition(store.archive.settings.routines?.last?.history.last?.state=="error")
        store.webBrowser?.shutdown()
        print("Release146Tests: skill loads on demand with web/agent off; manual scheduled approvals remain inline, notify the target chat and cancel cleanly")
    }
}
