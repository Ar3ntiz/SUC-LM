import Foundation

extension StudioStore {
    func commandNotice(_ id:UUID,_ text:String) {mutateSession(id){$0.messages.append(ChatMessage(role:"assistant",text:text.isEmpty ? "Nessun elemento disponibile.":text,kind:"command_help"))}}
    static let commandHelp="""
    Comandi della chat
    /help · questo elenco
    /compact [priorità] · sintetizza il contesto, conserva lo storico
    /context [token] · utilizzo e limite della sessione; 0 = automatico
    /auto on|off · compattazione automatica al 99%
    /status · modello, contesto e coda
    /new · nuova chat
    /clear · nuovo contesto nella stessa chat, storico conservato
    /stop · interrompi richiesta o compattazione
    /queue · mostra la coda; /pause e /resume la gestiscono
    /skills · skill attive; /skill nome richiesta le richiama
    /plugins · plugin attivi; /plugin identificatore richiesta li richiama
    /model · modello corrente
    /workspace · scegli la directory di lavoro
    /export · esporta la sessione in JSON
    /retry · rigenera l’ultima risposta
    //testo · invia un messaggio che inizia con /
    """
    func handleChatCommand(_ id:UUID)->Bool {
        guard let session=archive.sessions.first(where:{$0.id==id}) else{return false}
        let text=session.draft.trimmingCharacters(in:.whitespacesAndNewlines)
        guard text.hasPrefix("/") else{return false}
        if text.hasPrefix("//") {mutateSession(id){$0.draft=String(text.dropFirst())};return false}
        let pair=text.split(maxSplits:1,whereSeparator:{$0.isWhitespace}),command=String(pair[0]).lowercased(),argument=pair.count>1 ? String(pair[1]):""
        guard (session.draftAttachments ?? []).isEmpty else{error="Invia gli allegati in un messaggio separato dal comando.";return true}
        let known=["/help","/compact","/context","/auto","/status","/new","/clear","/stop","/queue","/pause","/resume","/skills","/skill","/plugins","/plugin","/model","/workspace","/export","/retry"]
        guard known.contains(command) else{commandNotice(id,"Comando sconosciuto: \(command). Usa /help; per inviare testo che inizia con / usa //.");return true}
        if command=="/plugin" || command=="/skill" {
            let parts=argument.split(maxSplits:1,whereSeparator:{$0.isWhitespace}),key=parts.first.map(String.init) ?? "",goal=parts.count>1 ? String(parts[1]):"Applica le istruzioni disponibili."
            if command=="/plugin" {
                guard let plugin=PluginCatalog.find(key,plugins:archive.settings.plugins ?? []) else{error="Plugin non attivo o sconosciuto. Usa /plugins.";return true}
                mutateSession(id){$0.draft="@plugin:"+plugin.manifest.identifier+" "+goal}
            } else {
                guard let skill=(archive.settings.skills ?? []).first(where:{$0.enabled && ($0.name==key || $0.id.uuidString==key)}) else{error="Skill non attiva o sconosciuta. Usa /skills.";return true}
                mutateSession(id){$0.draft="@"+skill.name+" "+goal}
            }
            return false
        }
        mutateSession(id){$0.draft=""}
        switch command {
        case "/help":commandNotice(id,Self.commandHelp)
        case "/compact":requestCompaction(id,instructions:argument)
        case "/context":
            if !argument.isEmpty {
                guard let value=Int(argument),value==0 || (1024...10_000_000).contains(value) else{error="Usa /context 0 oppure un limite tra 1024 e 10000000 token.";return true}
                mutateSession(id){$0.contextWindow=value==0 ? nil:value}
            }
            if let current=archive.sessions.first(where:{$0.id==id}) {commandNotice(id,contextDescription(current))}
        case "/auto":
            guard ["on","off"].contains(argument.lowercased()) else{error="Usa /auto on oppure /auto off.";return true}
            mutateSession(id){$0.autoCompact=argument.lowercased()=="on"};commandNotice(id,"Compattazione automatica "+(argument.lowercased()=="on" ? "attiva al 99%.":"disattivata."))
        case "/status":commandNotice(id,"Modello: \(chatModel)\n"+contextDescription(session)+"\nMessaggi in coda: \(session.queuedMessages?.count ?? 0)")
        case "/model":commandNotice(id,"Modello corrente: "+chatModel+". Configurazione nella sezione Server.")
        case "/new":newSession()
        case "/clear":
            guard !sessionTurnBusy(id) else{error="Interrompi o attendi il turno prima di azzerare il contesto.";return true}
            mutateSession(id){$0.contextSummary=nil;$0.contextThroughID=$0.messages.last?.id};commandNotice(id,"Nuovo contesto. Lo storico è conservato; le richieste successive ripartono dalle istruzioni di sistema.")
        case "/stop":cancelSelected()
        case "/pause":pauseChatQueue(id)
        case "/resume":resumeChatQueue(id)
        case "/queue":commandNotice(id,(session.queuedMessages ?? []).enumerated().map{"\($0.offset+1). \(String($0.element.text.prefix(120)))"}.joined(separator:"\n").isEmpty ? "Nessun messaggio in coda.":(session.queuedMessages ?? []).enumerated().map{"\($0.offset+1). \(String($0.element.text.prefix(120)))"}.joined(separator:"\n"))
        case "/skills":commandNotice(id,(archive.settings.skills ?? []).filter{$0.enabled}.map{"\($0.name): \($0.effectiveDescription)"}.joined(separator:"\n"))
        case "/plugins":commandNotice(id,(archive.settings.plugins ?? []).filter{$0.enabled}.map{"\($0.manifest.identifier) · \($0.manifest.name): \($0.manifest.description ?? "")"}.joined(separator:"\n"))
        case "/workspace":pickWorkspace()
        case "/export":exportSession(json:true)
        case "/retry":if let last=session.messages.last(where:{$0.role=="assistant" && !$0.isUIReport}) {Task{await reviseMessage(last.id,regenerate:true)}}
        default:break
        }
        return true
    }
    func contextDescription(_ session:ChatSession)->String {
        let usage=contextUsage(session,includeDraft:false)
        return "Contesto: \(usage.estimated ? "≈":"")\(usage.used) / \(usage.limit.map(String.init) ?? "limite sconosciuto") token. Auto-compact: \(session.autoCompact != false ? "99%":"off")."+(usage.limit==nil ? " Imposta /context TOKEN oppure verifica il modello remoto.":"")
    }
}
extension PluginCatalog {
    static func explicitInstructions(_ text:String,plugins:[InstalledPlugin],skills:[StudioSkill])->String {
        guard let first=text.split(whereSeparator:{$0.isWhitespace}).first,first.hasPrefix("@plugin:") else{return ""}
        return load(String(first.dropFirst(8)),plugins:plugins,skills:skills,explicit:true) ?? "Plugin richiesto non disponibile: non affermare di averlo eseguito."
    }
}
