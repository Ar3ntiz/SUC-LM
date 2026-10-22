import Foundation

extension StudioStore {
    func sessionTurnBusy(_ id: UUID) -> Bool {
        mcpJobs.running(id) || commandJobs.running(id) || pendingContextTools[id] != nil || compacting.contains(id) || requests.values.contains{$0.session==id} || agentRuns.contains(id) || mcpRuns.contains(id)
            || agentTurnStarts[id] != nil || editingSessions.contains(id)
    }
    var selectedTurnBusy: Bool { archive.selected.map{sessionTurnBusy($0)} ?? false }
    func enqueueDraft(_ id: UUID) {
        guard chatReady, !attachmentLoads.contains(id), !editingSessions.contains(id),
              let session=archive.sessions.first(where:{$0.id==id}) else{return}
        let text=session.draft.trimmingCharacters(in:.whitespacesAndNewlines),files=session.draftAttachments ?? []
        guard !text.isEmpty || !files.isEmpty else{return}
        guard text.utf8.count<1_000_000 else{error="Messaggio troppo grande.";return}
        do {try AttachmentReader.validate(files)} catch {self.error=error.localizedDescription;return}
        guard usesRemote || !files.contains(where:\.isImage) else{error="Le immagini richiedono Server remoto con un modello vision.";return}
        let item=ChatMessage(role:"user",text:text,attachments:files.isEmpty ? nil:files)
        mutateSession(id){$0.queuedMessages=($0.queuedMessages ?? [])+[item];$0.draft="";$0.draftAttachments=nil}
        scheduleChatQueue(id)
    }
    func removeQueuedMessage(_ id:UUID,sessionID:UUID) {
        mutateSession(sessionID){$0.queuedMessages?.removeAll{$0.id==id}}
    }
    func editQueuedMessage(_ id:UUID,sessionID:UUID) {
        guard !attachmentLoads.contains(sessionID),let session=archive.sessions.first(where:{$0.id==sessionID}),
              let item=session.queuedMessages?.first(where:{$0.id==id}) else{return}
        guard session.draft.isEmpty,(session.draftAttachments ?? []).isEmpty else{error="Salva o invia la bozza corrente prima di modificare un messaggio in coda.";return}
        mutateSession(sessionID){$0.queuedMessages?.removeAll{$0.id==id};$0.draft=item.text;$0.draftAttachments=item.attachments}
    }
    func pauseChatQueue(_ id:UUID) {
        guard archive.sessions.first(where:{$0.id==id})?.queuedMessages?.isEmpty==false else{return}
        mutateSession(id){$0.queuePaused=true}
    }
    func resumeChatQueue(_ id:UUID) {
        guard chatReady else {error="Carica il modello o configura il server remoto prima di riprendere la coda.";return}
        mutateSession(id){$0.queuePaused=false};scheduleChatQueue(id)
    }
    func scheduleChatQueue(_ id:UUID) {
        guard !queueDrainScheduled.contains(id) else{return}
        queueDrainScheduled.insert(id)
        Task { @MainActor [weak self] in
            await Task.yield()
            guard let self else{return}
            self.queueDrainScheduled.remove(id)
            guard !self.quitting, !self.sessionTurnBusy(id), self.chatReady, !self.attachmentLoads.contains(id),
                  let session=self.archive.sessions.first(where:{$0.id==id}), session.queuePaused != true,
                  let item=session.queuedMessages?.first else{return}
            self.sendMessage(sessionID:id,queuedInput:item)
            // Validation failures leave the original queued input intact and pause dispatch.
            if !self.compacting.contains(id),self.pendingContextTools[id]==nil,self.archive.sessions.first(where:{$0.id==id})?.queuedMessages?.first?.id==item.id {self.pauseChatQueue(id)}
        }
    }
}
