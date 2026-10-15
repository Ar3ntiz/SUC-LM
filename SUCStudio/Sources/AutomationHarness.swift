import Foundation

extension StudioStore {
    func runHooks(_ event: String, session: ChatSession, action: WorkspaceAction? = nil, helperURL: URL? = nil) async throws {
        let hooks=(archive.settings.hooks ?? []).filter{$0.enabled && $0.event==event}
        guard !hooks.isEmpty, !(session.workspacePath ?? "").isEmpty else{return}
        let root=try workspaceURL(session)
        for hook in hooks {
            try Task.checkCancellation()
            let request=WorkspaceAction(name:"hook · \(hook.name)",arguments:["event":event,"command":hook.command])
            guard await approve(request,session:session,root:root) else{throw MCPConnection.failure("Hook rifiutato: \(hook.name)")}
            let arguments=String(decoding:try JSONEncoder().encode(action?.arguments ?? [:]),as:UTF8.self)
            let environment=["SUC_HOOK_EVENT":event,"SUC_SESSION_ID":session.id.uuidString,"SUC_WORKSPACE":root.path,"SUC_TOOL_NAME":action?.name ?? "","SUC_TOOL_ARGUMENTS":arguments]
            let logID=UUID().uuidString;operationJournal.update(id:logID,title:"Hook · "+hook.name,state:"In corso",sessionID:session.id)
            do {
                let output=try await WorkspaceCommand(helperURL:helperURL).run(hook.command,root:root,environment:environment)
                let success=output.hasPrefix("Exit status: 0\n")
                mutateSession(session.id){$0.actions=($0.actions ?? [])+[ActionRecord(name:"Hook · \(hook.name)",details:request.summary,state:success ? "completed":"error",result:output)]}
                operationJournal.update(id:logID,state:success ? "Completata":"Errore")
                guard success else{throw MCPConnection.failure("Hook \(hook.name) terminato con errore; vedi registro azioni.")}
            }catch{operationJournal.update(id:logID,state:Task.isCancelled ? "Interrotta":"Errore");throw MCPConnection.failure("Hook \(hook.name): \(error.localizedDescription)")}
        }
    }
    func finishGeneration(sessionID: UUID, messageID: UUID) {
        guard let session=archive.sessions.first(where:{$0.id==sessionID}) else{return}
        guard (archive.settings.hooks ?? []).contains(where:{$0.enabled && $0.event=="request_end"}),!(session.workspacePath ?? "").isEmpty else{handleAgentResponse(sessionID:sessionID,messageID:messageID);return}
        agentRuns.insert(sessionID)
        agentTasks[sessionID]=Task {
            do {try await runHooks("request_end",session:session);try Task.checkCancellation();agentRuns.remove(sessionID);agentTasks.removeValue(forKey:sessionID);handleAgentResponse(sessionID:sessionID,messageID:messageID)}
            catch {agentRuns.remove(sessionID);agentTasks.removeValue(forKey:sessionID);completeSubagent(sessionID,result:.failure(error));actionNotice(sessionID,"Hook finale interrotto: "+error.localizedDescription);finishAgentReport(sessionID,reason:"Hook finale non completato.")}
        }
    }
    func delegateTask(_ action: WorkspaceAction, session: ChatSession) async throws -> String {
        guard session.allowSubagents==true,session.parentSessionID==nil else{throw MCPConnection.failure("Delega non abilitata o subagent già annidato.")}
        guard (subtaskCounts[session.id] ?? 0)<4, let task=action.arguments["task"],!task.isEmpty,task.utf8.count<=16*1024 else{throw MCPConnection.failure("Massimo 4 subtask per turno; obiettivo obbligatorio fino a 16 KiB.")}
        subtaskCounts[session.id,default:0]+=1
        var child=ChatSession();child.title="Subagent · "+String(task.prefix(50));child.parentSessionID=session.id
        child.workspacePath=session.workspacePath;child.workspaceBookmark=session.workspaceBookmark;child.agentEnabled=true;child.allowSubagents=false
        child.approvalMode=session.approvalMode;child.allowCommands=session.allowCommands;child.allowWeb=session.allowWeb;child.systemPrompt=session.systemPrompt
        child.temperature=session.temperature;child.maxTokens=session.maxTokens;child.thinking=session.thinking;child.sampling=session.sampling;child.draft=task
        archive.sessions.insert(child,at:0)
        let id=child.id;mutateSession(session.id){$0.subtasks=($0.subtasks ?? [])+[SubtaskRecord(title:task,sessionID:id)]}
        return try await withTaskCancellationHandler(operation:{
            try await withCheckedThrowingContinuation { continuation in
                subagentWaiters[id]=continuation
                if Task.isCancelled {completeSubagent(id,result:.failure(CancellationError()))}
                else {sendMessage(sessionID:id)}
            }
        },onCancel:{Task{@MainActor in
            self.stopAgent(id)
            for (requestID,target) in self.requests where target.session==id {self.requestHookTasks[requestID]?.cancel();self.remoteTasks[requestID]?.cancel();self.remoteClients[requestID]?.stop();self.engine.send(["command":"cancel","id":requestID]);self.mutateMessage(target.session,target.message){$0.state="cancelling"}}
            self.completeSubagent(id,result:.failure(CancellationError()))
        }})
    }
    func completeSubagent(_ id: UUID, result: Result<String, Error>) {
        guard let waiter=subagentWaiters.removeValue(forKey:id),let child=archive.sessions.first(where:{$0.id==id}),let parent=child.parentSessionID else{return}
        let text: String, state: String
        switch result {
        case .success(let value):text=value.utf8.count>32*1024 ? String(value.prefix(8000))+"\n[Risultato abbreviato: testo completo nella sessione del subagent.]" : value;state="completed"
        case .failure(let error):text=error.localizedDescription;state="error"
        }
        mutateSession(parent){session in if let i=session.subtasks?.firstIndex(where:{$0.sessionID==id}){session.subtasks?[i].state=state;session.subtasks?[i].result=text}}
        switch result {case .success:waiter.resume(returning:text);case .failure(let error):waiter.resume(throwing:error)}
    }
}
