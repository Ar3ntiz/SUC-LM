import Foundation

extension ChatMessage {
    var protocolText: String {ThinkingParts.split(text,separate:reasoning,implicit:implicitThinking==true).answer}
    var isOperational:Bool { kind=="tool_call" || kind=="tool_result" }
    var isUIReport:Bool { kind=="action_report" || kind=="action_notice" || kind=="command_help" }
    var commandAction:WorkspaceAction? {
        guard role=="assistant",let action=executedAction ?? WorkspaceAction.parse(protocolText),
              ["run_command","run_interactive"].contains(action.name) else{return nil}
        return action
    }
    var chatParts:ThinkingParts {MessagePresentationCache.parts(self)}
    var uncachedChatParts:ThinkingParts {
        var parts=ThinkingParts.split(text,separate:reasoning,implicit:implicitThinking==true)
        guard role=="assistant",!isUIReport else{return parts}
        let streaming=["queued","prefill","remote_wait","generating","cancelling"].contains(state)
        if kind=="tool_call" || WorkspaceAction.isActionEnvelope(parts.answer) || (streaming && WorkspaceAction.pendingEnvelope(parts.answer)) {
            parts.answer=WorkspaceAction.commentaryBeforeEnvelope(parts.answer,partial:streaming)
        }
        return parts
    }
    func visibleInChat(agent:Bool,web:Bool=false)->Bool {
        if kind=="tool_result" {return false}
        guard role=="assistant",!isUIReport else{return true}
        let parts=chatParts
        if !parts.answer.isEmpty || !parts.reasoning.isEmpty || commandAction != nil {return true}
        if isOperational || WorkspaceAction.isActionEnvelope(protocolText) {return false}
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
        operationJournal.update(id:record.id.uuidString,title:WebTools.names.contains(record.name) ? nil:record.name,state:["completed":"Completata","error":"Errore","denied":"Rifiutata","cancelled":"Interrotta","running":"In corso"][record.state] ?? record.state,sessionID:sessionID)
        mutateSession(sessionID){$0.actions=($0.actions ?? [])+[record]}
        var notice=String(record.result.prefix(1200))
        if CommandJobTools.names.contains(record.name),let payload=try? JSONSerialization.jsonObject(with:Data(record.result.utf8)) as? [String:Any],payload["job_id"] != nil {
            notice=L("Comando terminato con errore. Apri il registro azioni della sessione per dettagli e output.")
        }
        if record.name=="mcp_call" || MCPJobTools.management.contains(record.name),
           let payload=try? JSONSerialization.jsonObject(with:Data(record.result.utf8)) as? [String:Any],payload["job_id"] != nil {
            notice=L("Chiamata MCP terminata con errore. Apri il registro azioni per i dettagli.")
        }
        if record.state != "completed" && record.state != "running" { actionNotice(sessionID,"**\(record.state == "denied" ? "Azione rifiutata" : record.state == "cancelled" ? "Azione interrotta" : "Errore nell’azione") · \(record.name)**\n\n\(notice)") }
    }
    func finishAgentReport(_ sessionID:UUID,reason:String?=nil) {
        if reason != nil { pauseChatQueue(sessionID) }
        defer { if reason == nil { scheduleContextContinuation(sessionID) } }
        activeSkillInstructions.removeValue(forKey:sessionID)
        guard let start=agentTurnStarts.removeValue(forKey:sessionID),let session=archive.sessions.first(where:{$0.id==sessionID}) else{return}
        let records=Array((session.actions ?? []).dropFirst(start)).filter{!CommandJobTools.management.contains($0.name) && !MCPJobTools.management.contains($0.name)}
        guard !records.isEmpty || reason != nil else{return}
        let done=records.filter{$0.state=="completed"},running=records.filter{$0.state=="running"}.count
        let failed=records.count-done.count-running
        let names=["list_files":"elenchi directory","read_file":"letture file","write_file":"scritture file","make_directory":"directory create","run_command":"comandi","delegate_task":"subtask","web_search":"ricerche web","visit_page":"pagine web","load_skill":"skill caricate"]
        let singles=["list_files":"elenco directory","read_file":"lettura file","write_file":"scrittura file","make_directory":"directory creata","run_command":"comando","delegate_task":"subtask","web_search":"ricerca web","visit_page":"pagina web","load_skill":"skill caricata"]
        let groups=Dictionary(grouping:done,by:{$0.name=="google_search" ? "web_search":$0.name}).sorted{$0.key<$1.key}.map{"\($0.value.count) \(($0.value.count==1 ? singles[$0.key]:names[$0.key]) ?? $0.key)"}.joined(separator:" · ")
        var parts:[String]=[]
        if !groups.isEmpty {parts.append(groups)}
        else if !done.isEmpty {parts.append("\(done.count) azioni completate")}
        if running>0 {parts.append(L("\(running) operazioni avviate"))}
        if failed>0 {parts.append("\(failed) \(failed == 1 ? "non completata":"non completate")")}
        if let reason {parts.append(reason)}
        let text=parts.joined(separator:" · ")
        mutateSession(sessionID){$0.messages.append(ChatMessage(role:"assistant",text:text,kind:"action_report"))}
    }
}

/// Stable historical messages are parsed once instead of on every incoming token.
private enum MessagePresentationCache {
    final class Entry:NSObject {
        let message:ChatMessage
        let parts:ThinkingParts
        init(_ message:ChatMessage){self.message=message;self.parts=message.uncachedChatParts}
    }
    static let cache:NSCache<NSUUID,Entry> = {let c=NSCache<NSUUID,Entry>();c.countLimit=256;c.totalCostLimit=16*1024*1024;return c}()
    static func parts(_ message:ChatMessage)->ThinkingParts {
        let key=message.id as NSUUID
        if let entry=cache.object(forKey:key),entry.message==message {return entry.parts}
        let entry=Entry(message);cache.setObject(entry,forKey:key,cost:message.text.utf8.count+(message.reasoning?.utf8.count ?? 0));return entry.parts
    }
}
