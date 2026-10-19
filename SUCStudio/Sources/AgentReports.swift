import Foundation

extension ChatMessage {
    var protocolText: String {ThinkingParts.split(text,separate:reasoning,implicit:implicitThinking==true).answer}
    var isOperational:Bool { kind=="tool_call" || kind=="tool_result" }
    var isUIReport:Bool { kind=="action_report" || kind=="action_notice" }
    func visibleInChat(agent:Bool,web:Bool=false)->Bool {
        if isOperational {return false}
        guard role=="assistant",!isUIReport else{return true}
        if WorkspaceAction.parse(protocolText) != nil || WorkspaceAction.isActionEnvelope(protocolText) {return false}
        let streaming=["queued","prefill","remote_wait","generating","cancelling"].contains(state)
        return !(streaming && WorkspaceAction.pendingEnvelope(protocolText))
    }
    var compactActivityText: String {
        text.replacingOccurrences(of:"**Riepilogo attività**",with:"")
            .replacingOccurrences(of:"Dettagli nel registro azioni della sessione.",with:"")
            .replacingOccurrences(of:"**",with:"")
            .components(separatedBy:.newlines).map{$0.trimmingCharacters(in:.whitespacesAndNewlines)}
            .filter{!$0.isEmpty}.joined(separator:" · ")
    }
}
extension StudioStore {
    func actionNotice(_ sessionID:UUID,_ text:String) {
        mutateSession(sessionID){$0.messages.append(ChatMessage(role:"assistant",text:text,kind:"action_notice"))}
        markSessionUnreadIfNeeded(sessionID)
    }
    func recordAction(_ sessionID:UUID,_ record:ActionRecord) {
        operationJournal.update(id:record.id.uuidString,title:WebTools.names.contains(record.name) ? nil:record.name,state:["completed":"Completata","error":"Errore","denied":"Rifiutata","cancelled":"Interrotta"][record.state] ?? record.state,sessionID:sessionID)
        mutateSession(sessionID){$0.actions=($0.actions ?? [])+[record]}
        if record.state != "completed" { actionNotice(sessionID,"**\(record.state == "denied" ? "Azione rifiutata" : record.state == "cancelled" ? "Azione interrotta" : "Errore nell’azione") · \(record.name)**\n\n\(String(record.result.prefix(1200)))") }
    }
    func finishAgentReport(_ sessionID:UUID,reason:String?=nil) {
        guard let start=agentTurnStarts.removeValue(forKey:sessionID),let session=archive.sessions.first(where:{$0.id==sessionID}) else{return}
        let records=Array((session.actions ?? []).dropFirst(start))
        guard !records.isEmpty || reason != nil else{return}
        let done=records.filter{$0.state=="completed"},failed=records.count-done.count
        let names=["list_files":"elenchi directory","read_file":"letture file","write_file":"scritture file","make_directory":"directory create","run_command":"comandi","delegate_task":"subtask","web_search":"ricerche web","visit_page":"pagine web","load_skill":"skill caricate"]
        let singles=["list_files":"elenco directory","read_file":"lettura file","write_file":"scrittura file","make_directory":"directory creata","run_command":"comando","delegate_task":"subtask","web_search":"ricerca web","visit_page":"pagina web","load_skill":"skill caricata"]
        let groups=Dictionary(grouping:done,by:{$0.name=="google_search" ? "web_search":$0.name}).sorted{$0.key<$1.key}.map{"\($0.value.count) \(($0.value.count==1 ? singles[$0.key]:names[$0.key]) ?? $0.key)"}.joined(separator:" · ")
        var parts:[String]=[]
        if !groups.isEmpty {parts.append(groups)}
        else if !done.isEmpty {parts.append("\(done.count) azioni completate")}
        if failed>0 {parts.append("\(failed) \(failed == 1 ? "non completata":"non completate")")}
        if let reason {parts.append(reason)}
        let text=parts.joined(separator:" · ")
        mutateSession(sessionID){$0.messages.append(ChatMessage(role:"assistant",text:text,kind:"action_report"))}
    }
}
