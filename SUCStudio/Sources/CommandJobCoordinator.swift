import Foundation

extension StudioStore {
    func commandJobFinished(_ job:CommandJob) {
        objectWillChange.send()
        let value=job.snapshot()
        if let session=archive.sessions.first(where:{$0.id==job.sessionID}),let record=session.actions?.first(where:{$0.id==job.recordID && $0.state=="running"}) {
            var updated=record
            updated.state=value.state=="completed" ? "completed":value.state=="cancelled" ? "cancelled":"error"
            updated.result=value.json
            mutateSession(job.sessionID){s in if let index=s.actions?.firstIndex(where:{$0.id==job.recordID}){s.actions?[index]=updated}}
            operationJournal.update(id:job.recordID.uuidString,state:value.state=="completed" ? "Completata":value.state=="cancelled" ? "Interrotta":"Errore")
            if value.state=="failed" || value.state=="timed_out" {actionNotice(job.sessionID,L("Comando terminato con errore. Apri il pannello comandi per dettagli e output."))}
        }
        markSessionUnreadIfNeeded(job.sessionID)
        scheduleChatQueue(job.sessionID)
    }
}
