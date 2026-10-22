import Foundation

extension StudioStore {
    func runHooks(_ event: String, session: ChatSession, action: WorkspaceAction? = nil, helperURL: URL? = nil) async throws {
        let result=await invokeHooks(event,session:session,action:action,helperURL:helperURL)
        hookContexts[session.id,default:[]]+=result.context
        if let deny=result.deny {throw ExtensionFiles.failure(deny)}
    }
    func finishGeneration(sessionID: UUID, messageID: UUID) {
        guard let session=archive.sessions.first(where:{$0.id==sessionID}),let message=session.messages.first(where:{$0.id==messageID}) else{return}
        let contextWarning:String?=message.generation.flatMap { snapshot in
            snapshot.maxTokens<session.maxTokens ? L("Risposta limitata dallo spazio nel contesto: budget effettivo \(snapshot.maxTokens) token. Usa /compact o riduci l’input; aumentare Output max non crea spazio nella finestra."):nil
        }
        if message.protocolText.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty {
            let detail=message.finishReason == "length" ? contextWarning ?? "Il modello ha esaurito il massimo output durante il thinking. Aumenta Output max o disattiva il thinking sul server.":"Il modello ha restituito solo thinking o una risposta vuota, senza risultato finale."
            mutateMessage(sessionID,messageID){$0.state="error"}
            actionNotice(sessionID,detail)
            finishAgentReport(sessionID,reason:"Nessun risultato finale ricevuto.")
            completeSubagent(sessionID,result:.failure(MCPConnection.failure(detail)))
            return
        }
        if message.finishReason == "length" {
            actionNotice(sessionID,contextWarning ?? "Risposta interrotta al limite Output max: il risultato può essere incompleto.")
        }
        if forwardQueuedAtBoundary(sessionID,messageID:messageID) {return}
        let final=WorkspaceAction.parse(message.protocolText)==nil && !WorkspaceAction.isActionEnvelope(message.protocolText)
        guard final,(archive.settings.hooks ?? []).contains(where:{$0.enabled && ["Stop","SubagentStop"].contains(HookDefinition.event($0.event))}) else{handleAgentResponse(sessionID:sessionID,messageID:messageID);return}
        agentRuns.insert(sessionID)
        agentTasks[sessionID]=Task {
            let event=session.parentSessionID==nil ? "Stop":"SubagentStop"
            let result=await invokeHooks(event,session:session,extra:["last_assistant_message":message.text,"stop_hook_active":stopHookActive.contains(sessionID),"agent_id":sessionID.uuidString])
            hookContexts[sessionID,default:[]]+=result.context
            agentRuns.remove(sessionID);agentTasks.removeValue(forKey:sessionID)
            guard !Task.isCancelled else{return}
            if let reason=result.continuation,!stopHookActive.contains(sessionID),(agentIterations[sessionID] ?? 0)<archive.settings.actionLimit {
                stopHookActive.insert(sessionID);agentIterations[sessionID,default:0]+=1
                sendMessage(sessionID:sessionID,toolResult:"Feedback hook: "+reason)
            } else {handleAgentResponse(sessionID:sessionID,messageID:messageID)}
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
        _ = await invokeHooks("SubagentStart",session:session,extra:["agent_id":child.id.uuidString,"agent_type":"general","prompt":task])
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
