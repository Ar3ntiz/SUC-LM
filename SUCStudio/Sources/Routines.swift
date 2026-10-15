import Foundation
extension StudioStore {
    func routineStatus(_ id: UUID, _ value: String) {
        guard let i=archive.settings.routines?.firstIndex(where:{$0.id==id}),archive.settings.routines?[i].status != value else{return}
        archive.settings.routines?[i].status=value;scheduleSave()
    }
    func tickRoutines(now: Date = Date()) {
        for (id,sessionID) in activeRoutines {
            guard !requests.values.contains(where:{$0.session==sessionID}),!agentRuns.contains(sessionID) else{continue}
            let session=archive.sessions.first{$0.id==sessionID}
            let last=session?.messages.last(where:{!$0.isUIReport})
            let success=last?.role=="assistant" && last?.state=="complete" && WorkspaceAction.parse(last?.text ?? "")==nil
            if let i=archive.settings.routines?.firstIndex(where:{$0.id==id}) {
                archive.settings.routines?[i].status=success ? "Completata":"Interrotta o errore"
                archive.settings.routines?[i].history.append(RoutineRun(state:success ? "completed":"error",detail:String((last?.text ?? "Sessione non disponibile").prefix(2000))))
                archive.settings.routines?[i].history=Array((archive.settings.routines?[i].history ?? []).suffix(50));scheduleSave()
            }
            activeRoutines.removeValue(forKey:id)
        }
        guard schedulingEnabled,!hasWork else{return}
        for routine in (archive.settings.routines ?? []).filter({$0.enabled && $0.nextRun<=now}).sorted(by:{$0.nextRun<$1.nextRun}) {
            guard routine.backend==(usesRemote ? "remote":"local") else{routineStatus(routine.id,"Attende backend \(routine.backend)");continue}
            guard chatReady else{routineStatus(routine.id,"Attende modello disponibile");continue}
            guard let sessionID=routine.sessionID,archive.sessions.contains(where:{$0.id==sessionID}) else{routineStatus(routine.id,"Sessione non disponibile");continue}
            guard let i=archive.settings.routines?.firstIndex(where:{$0.id==routine.id}) else{continue}
            // Persist the next date before dispatch. Never replay a backlog after sleep/relaunch.
            archive.settings.routines?[i].advance(after:now);routineStatus(routine.id,"In esecuzione")
            agentIterations[sessionID]=0;subtaskCounts[sessionID]=0
            sendMessage(sessionID:sessionID,toolResult:routine.prompt,isToolResult:false)
            if requests.values.contains(where:{$0.session==sessionID}) {activeRoutines[routine.id]=sessionID}
            else{routineStatus(routine.id,"Invio non riuscito")}
            scheduleSave();break
        }
    }
    func saveRoutine(_ value: StudioRoutine) throws {
        guard !value.name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,!value.prompt.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,value.prompt.utf8.count<=16*1024,(1...525600).contains(value.intervalMinutes),["local","remote"].contains(value.backend),let sessionID=value.sessionID,archive.sessions.contains(where:{$0.id==sessionID && $0.parentSessionID==nil}) else{throw MCPConnection.failure("Routine: nome, prompt, sessione e intervallo valido obbligatori (1–525600 minuti; prompt fino a 16 KiB).")}
        guard activeRoutines[value.id]==nil else{throw MCPConnection.failure("Attendi la fine della routine prima di modificarla.")}
        var items=archive.settings.routines ?? []
        if let index=items.firstIndex(where:{$0.id==value.id}){items[index]=value}else{guard items.count<64 else{throw MCPConnection.failure("Massimo 64 routine.")};items.append(value)}
        archive.settings.routines=items;scheduleSave()
    }
}
