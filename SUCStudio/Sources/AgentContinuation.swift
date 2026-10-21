import Foundation

enum AgentContinuation {
    static let limit=2
    static let invalidFeedback="""
Feedback harness · formato azione non valido (nessuno strumento eseguito).
La precedente richiesta suc_action non è un'azione completa utilizzabile. Ricontrolla il JSON: oggetto unico suc_action con name e arguments, virgolette e caratteri di controllo correttamente escaped, tutte le parentesi chiuse. Per write_file, content deve essere una stringa JSON valida anche quando contiene Markdown, blocchi codice o ritorni a capo. Non troncare il documento e non dichiararlo salvato.
Se l'azione è ancora necessaria e autorizzata, invia UNA richiesta completa corretta; attendi il risultato reale. In alternativa spiega concretamente il blocco. Non cambiare permessi, non inventare risultati e non ripetere azioni già concluse.
"""
    static let feedback="""
Feedback harness · promessa senza azione (non è un risultato di strumento).
La risposta precedente annuncia un'operazione ma non richiede alcuno strumento: non c'è lavoro in esecuzione implicito.
Se il compito dell'utente richiede ancora lavoro, richiedi ora UNA vera azione suc_action completa per il prossimo passo necessario e già autorizzato. Non ripetere soltanto «procedo» o «sto cercando». Rispetta directory, strumenti abilitati e approvazioni; non inventare comandi già eseguiti o nuove autorizzazioni.
Se il lavoro è concluso, serve una decisione dell'utente o sei bloccato, fornisci invece una risposta finale concreta che spieghi risultato o impedimento. Non compiere azioni inutili per soddisfare questo controllo.
"""
    /// Conservative IT/EN heuristic, not a task-completion or truth classifier.
    static func isPromise(_ text:String)->Bool {
        let value=WorkspaceAction.answerText(text).trimmingCharacters(in:.whitespacesAndNewlines)
        guard !value.isEmpty,value.count<=1600,!value.contains("?"),!value.contains("```"),
              !value.hasPrefix(">"),!value.hasPrefix("\""),!value.hasPrefix("“"),
              !WorkspaceAction.isActionEnvelope(value) else{return false}
        let lower=value.lowercased()
        let blocked=["non posso","non riesco","ho bisogno","mi serve","attendo","conferma","confermi","permesso","cannot","can't","can’t","waiting for","need your","please provide","permission","approval","would you","shall i"]
        guard !blocked.contains(where:lower.contains) else{return false}
        // Only a short terminal sentence/paragraph promising immediate work qualifies.
        let terminal=lower.components(separatedBy:CharacterSet(charactersIn:"\n.!;")).map{$0.trimmingCharacters(in:.whitespaces)}.last(where:{!$0.isEmpty}) ?? lower
        guard terminal.count<=500 else{return false}
        let prefix="^(?:(?:hai ragione|va bene|ok|okay|certo|bene|sure)[,: ]+)?"
        let italian="(?:procedo(?:\\s|$)|(?:ora|adesso)\\s+(?:cerco|verifico|controllo|eseguo|leggo|analizzo|modifico|avvio|installo)\\b|sto\\s+(?:cercando|verificando|controllando|eseguendo|leggendo|analizzando)\\b)"
        let english="(?:(?:i['’]ll|i will|let me|i['’]m(?: going to)?)\\s+(?:now\\s+)?(?:check|search|look|read|run|execute|inspect|verify|update|install|investigate|proceed|continue)\\b)"
        return terminal.range(of:prefix+"(?:"+italian+"|"+english+")",options:.regularExpression) != nil
    }
}

extension StudioStore {
    /// Called only for completed plain replies after the running-job path.
    func recoverPromisedAction(_ session:ChatSession,message:ChatMessage)->Bool {
        guard message.state=="complete",message.finishReason != "length",chatReady,
              session.fullAgent || session.webToolsEnabled || !mcpTools.isEmpty,
              !approvals.contains(where:{$0.sessionID==session.id}),
              AgentContinuation.isPromise(message.protocolText) else{return false}
        return requestAgentRepair(session,message:message,feedback:AgentContinuation.feedback,
            notice:"Il modello ha annunciato un’operazione senza avviarla. Richiedo il prossimo passo concreto.",
            failure:"Il modello continua ad annunciare operazioni senza richiedere strumenti. Recupero automatico fermato.")
    }
    func recoverInvalidAction(_ session:ChatSession,message:ChatMessage)->Bool {
        guard message.state=="complete",message.finishReason != "length",chatReady,
              message.protocolText.utf8.count<=128*1024 else{return false}
        return requestAgentRepair(session,message:message,feedback:AgentContinuation.invalidFeedback,
            notice:"Richiesta di strumento incompleta: chiedo al modello di correggere il formato. Nessuna azione eseguita.",
            failure:"Il modello non ha corretto la richiesta di strumento. Recupero automatico fermato.")
    }
    private func requestAgentRepair(_ session:ChatSession,message:ChatMessage,feedback:String,notice:String,failure:String)->Bool {
        if agentRepairMessages[session.id,default:[]].contains(message.id) {return true}
        agentRepairMessages[session.id,default:[]].insert(message.id)
        let used=agentRepairs[session.id,default:0]
        guard used<AgentContinuation.limit,(agentIterations[session.id] ?? 0)<archive.settings.actionLimit else {
            mutateMessage(session.id,message.id){$0.state="error"}
            actionNotice(session.id,failure);finishAgentReport(session.id,reason:failure)
            completeSubagent(session.id,result:.failure(MCPConnection.failure(failure)))
            return true
        }
        agentRepairs[session.id]=used+1;agentIterations[session.id,default:0]+=1
        if used==0 {actionNotice(session.id,notice)}
        sendMessage(sessionID:session.id,toolResult:feedback)
        return true
    }
}
