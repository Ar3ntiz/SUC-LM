import SwiftUI
import AppKit
import UniformTypeIdentifiers

@MainActor final class StudioStore: ObservableObject {
    @Published var archive: Archive
    @Published var tab = "chat"
    @Published var modelState = "unloaded"
    @Published var loadPhase = "Preparazione del caricamento"
    @Published var loadMemory:LoadMemoryStatus?
    @Published var prefillProgress: [UUID: PrefillStatus] = [:]
    @Published var loadedModel = ""
    @Published var serverRunning = false
    @Published var serverBusy = false
    @Published var apiAddress = ""
    @Published var error: String?
    @Published var artifactsVisible=false
    @Published var search = ""
    @Published var showArchived = false
    @Published var latestMetrics: Metrics?
    @Published var latestIsCurrent = false
    @Published var activePhase = "Pronto"
    private var activeRequestID = ""
    @Published var apiRequests = Set<String>()
    @Published var operationJournal=OperationJournal()
    @Published var modelIOLog=ModelIOLog()
    let commandJobs:CommandJobs
    let mcpJobs=MCPJobs()
    let actionLoopGuard=ActionLoopGuard()
    @Published var terminals:[UUID:InteractiveTerminal]=[:]
    @Published var logs: [String] = []
    @Published var completed = 0
    @Published var engineFootprint: Double?
    @Published var loadedContext = StudioPlatform.defaultContext
    @Published var memoryPlan = ""
    @Published var expertStats = ""
    @Published var mcpTools: [MCPTool] = []
    @Published var mcpStatus: [UUID: String] = [:]
    @Published var mcpRuns = Set<UUID>()
    var mcpConnections: [UUID: MCPConnection] = [:]
    @Published var agentRuns = Set<UUID>()
    @Published var approvals: [ActionApproval] = []
    var approvalWaiters: [UUID: CheckedContinuation<Bool, Never>] = [:]
    var agentTasks: [UUID: Task<Void, Never>] = [:]
    var activeSkillInstructions:[UUID:String]=[:]
    var agentTurnStarts: [UUID:Int] = [:]
    var agentIterations: [UUID: Int] = [:]
    var agentRepairs: [UUID:Int] = [:]
    var agentRepairMessages: [UUID:Set<UUID>] = [:]
    var subtaskCounts: [UUID: Int] = [:]
    var subagentWaiters: [UUID: CheckedContinuation<String, Error>] = [:]
    var activeRoutines: [UUID: UUID] = [:]
    var webBrowser: WebBrowser?
    @Published var webActivity: WebActivity? // globe button in the chat toolbar (1.4.5_Final)
    var webBrowserAllowed = false // ds4 browser_allowed: valid until the app quits
    var schedulingEnabled = true
    var hookSessions = Set<UUID>()
    var hookTurnIDs: [UUID:String] = [:]
    var hookToolIDs: [UUID:String] = [:]
    var hookContexts: [UUID:[String]] = [:]
    var backgroundHooks: [UUID:[UUID:Task<Void,Never>]] = [:]
    var stopHookActive = Set<UUID>()
    var requestHookTasks: [String: Task<Void, Never>] = [:]
    @Published var attachmentLoads = Set<UUID>()
    var selectedAttachmentsLoading: Bool { archive.selected.map { attachmentLoads.contains($0) } ?? false }
    var canSendSelected: Bool { chatReady && !(archive.selected.map { editingSessions.contains($0) } ?? false) && !selectedAttachmentsLoading && selected.map { !$0.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !($0.draftAttachments ?? []).isEmpty } == true }
    var queueDrainScheduled = Set<UUID>()
    @Published var compacting = Set<UUID>()
    @Published var compactionProgress: [UUID:CompactionProgress] = [:]
    var compactionTasks: [UUID:Task<Void,Never>] = [:]
    var compactionClients: [UUID:RemoteInference] = [:]
    var summaryWaiters: [String:SummaryWaiter] = [:]
    var compactionRequestSessions: [String:UUID] = [:]
    var pendingContextTools: [UUID:(text:String,isTool:Bool)] = [:]
    @Published var editingSessions = Set<UUID>()
    private var retiredRequests = Set<String>()
    var remoteClients: [String: RemoteInference] = [:]
    var remoteTasks: [String: Task<Void, Never>] = [:]
    var usesRemote: Bool { StudioPlatform.effectiveBackend(archive.settings.chatBackend) == "remote" }
    var chatModel: String { usesRemote ? (archive.settings.remote?.model ?? "") : loadedModel }
    var chatStateLabel: String { usesRemote ? (chatReady ? "Server remoto · " + chatModel : "Configura server remoto") : stateLabel }
    var chatReady: Bool { usesRemote ? !(archive.settings.remote?.model.isEmpty ?? true) : modelReady }
    private var statusTimer: Timer?
    @Published var saveError = false
    private(set) var requests: [String: (session: UUID, message: UUID, start: Date)] = [:]
    let power = PowerController()
    @Published var launchAtLogin = false
    @Published var loginStatus = ""
    @Published var powerStatus = "Standby gestito da macOS"
    let engine = EngineClient()
    let disk: ArchiveDisk
    let archiveURL: URL
    private var saveRevision = 0
    private var saveTask: Task<Void, Never>?
    private var loadAfterStart = false
    private var loadID = ""
    var quitting = false
    var selected: ChatSession? { archive.sessions.first { $0.id == archive.selected } }
    var hasWork: Bool { mcpJobs.running() || commandJobs.running() ||  !pendingContextTools.isEmpty || !compacting.isEmpty || terminals.values.contains(where:{$0.running}) || !agentRuns.isEmpty || !mcpRuns.isEmpty || !attachmentLoads.isEmpty || !requests.isEmpty || !apiRequests.isEmpty || modelState == "loading" }
    var modelReady: Bool { modelState == "ready" }
    var stateLabel: String { ["ready": "Modello pronto", "loading": "Caricamento…", "unloaded": "Nessun modello", "error": "Errore motore"][modelState] ?? modelState }
    init(storageURL: URL? = nil, commandHelperURL: URL? = nil) {
        commandJobs=CommandJobs(helperURL:commandHelperURL)
        let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent(StudioPlatform.dataDirectory)
        archiveURL = storageURL ?? root.appendingPathComponent("sessions.json"); disk = ArchiveDisk(url: archiveURL)
        do { archive = try ArchiveDisk.load(archiveURL) }
        catch {
            archive = Archive(); saveError = true
            self.error = "Archivio non leggibile. I dati originali sono preservati. Usa Ripristina backup in Settings. \(error.localizedDescription)"
        }
        if archive.sessions.isEmpty { var s = ChatSession();s.title=archive.settings.effectiveLanguage == "it" ? "Nuova chat":"New chat"; archive.sessions = [s]; archive.selected = s.id }
        if !archive.sessions.contains(where: { $0.id == archive.selected }) { archive.selected = archive.sessions.first?.id }
        if !StudioPlatform.localInferenceAvailable { archive.settings.chatBackend = "remote" }
        // Restored queues require explicit consent to resume after relaunch.
        for i in archive.sessions.indices where !(archive.sessions[i].queuedMessages ?? []).isEmpty { archive.sessions[i].queuePaused = true }
        StudioL10n.language=archive.settings.effectiveLanguage; StudioTypography.apply(archive.settings)
        commandJobs.onChange={ [weak self] in self?.objectWillChange.send() }
        commandJobs.onFinish={ [weak self] job in self?.commandJobFinished(job) }
        mcpJobs.onFinish={ [weak self] job in self?.mcpJobFinished(job) }
        if storageURL == nil { refreshLoginStatus(); if archive.settings.keepAwake == true { setAwake(true) } }
        statusTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.engine.send(["command": "status"]); self?.tickRoutines() }
        }
        Task { for configuration in archive.settings.mcpServers ?? [] where configuration.enabled { await connectMCP(configuration) } }
        engine.onEvent = { [weak self] in self?.receive($0) }
        engine.onLog = { [weak self] text in self?.appendLog(text) }
        engine.onExit = { [weak self] code in
            guard let self else { return }
            for waiter in self.summaryWaiters.values{waiter.continuation.resume(throwing:MCPConnection.failure("Motore locale terminato."))};self.summaryWaiters.removeAll();self.compactionRequestSessions.removeAll()
            self.modelState = "unloaded"; self.loadedModel = ""; self.serverRunning = false; self.serverBusy = false
            for (id, target) in self.requests where self.remoteClients[id] == nil { self.mutateMessage(target.session, target.message) { $0.state = "interrupted" }; self.requests.removeValue(forKey: id);self.operationJournal.update(id:id,state:"Interrotta"); self.completeSubagent(target.session,result:.failure(MCPConnection.failure("Motore locale terminato."))); self.finishAgentReport(target.session,reason:"Motore locale terminato.") };for id in self.apiRequests {self.operationJournal.update(id:id,state:"Interrotta")};if !self.loadID.isEmpty{self.operationJournal.update(id:self.loadID,state:"Interrotta")}; self.apiRequests.removeAll(); self.scheduleSave()
            if !self.quitting && code != 0 { self.error = "Il motore si è chiuso (\(code)). Puoi ricaricare il modello." }
        }
    }
    func appendLog(_ value: String) { logs.append(String(value.prefix(1500))); if logs.count > 150 { logs.removeFirst(logs.count - 150) } }
    func setModelIOLogEnabled(_ enabled:Bool) {
        archive.settings.modelIOLogEnabled=enabled
        if !enabled {modelIOLog.clear()}
        engine.send(["command":"set_model_log","enabled":enabled])
        scheduleSave()
    }
    private func appendModelIO(_ id:String,_ direction:String,_ text:String,coalesce:Bool=false) {
        guard archive.settings.ioLogEnabled else{return}
        modelIOLog.append(requestID:id,direction:direction,text:text,coalesce:coalesce)
    }
    func setLanguage(_ value:String) {
        StudioL10n.language=value
        archive.settings.language=StudioL10n.language
        scheduleSave()
    }
    func scheduleSave() {
        guard !saveError && !quitting else { return }
        saveRevision += 1
        guard saveTask == nil else { return }
        saveTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 400_000_000)
            guard !Task.isCancelled, let self else { return }
            let revision = self.saveRevision
            do { try await self.disk.save(self.archive) } catch { self.error = "Salvataggio: \(error.localizedDescription)" }
            self.saveTask = nil
            if revision != self.saveRevision { self.scheduleSave() }
        }
    }
    func mutateSession(_ id: UUID, _ action: (inout ChatSession) -> Void) {
        guard let i = archive.sessions.firstIndex(where: { $0.id == id }) else { return }; action(&archive.sessions[i]); archive.sessions[i].updated = Date(); scheduleSave()
    }
    func mutateSelected(_ action: (inout ChatSession) -> Void) {
        guard let i = archive.sessions.firstIndex(where: { $0.id == archive.selected }) else { return }
        action(&archive.sessions[i]); archive.sessions[i].updated = Date(); scheduleSave()
    }
    func mutateMessage(_ session: UUID, _ message: UUID, _ action: (inout ChatMessage) -> Void) {
        guard let i = archive.sessions.firstIndex(where: { $0.id == session }), let j = archive.sessions[i].messages.firstIndex(where: { $0.id == message }) else { return }
        action(&archive.sessions[i].messages[j]); archive.sessions[i].updated = Date()
    }
    func markSessionUnreadIfNeeded(_ id: UUID) {
        guard archive.selected != id || tab != "chat" else{return}
        guard let i=archive.sessions.firstIndex(where:{$0.id==id}) else{return}
        archive.sessions[i].hasUnreadMessages=true;scheduleSave()
    }
    func markSessionRead(_ id: UUID) {
        guard archive.sessions.first(where:{$0.id==id})?.hasUnreadMessages==true else{return}
        if let i=archive.sessions.firstIndex(where:{$0.id==id}) {archive.sessions[i].hasUnreadMessages=false;scheduleSave()}
    }
    func selectSession(_ id: UUID) {
        archive.selected=id
        if tab=="chat" {markSessionRead(id)}
        scheduleSave()
    }
    func newSession() { var s = ChatSession(); s.title=L("Nuova chat"); archive.sessions.insert(s, at: 0); archive.selected = s.id; showArchived = false; search = ""; tab = "chat"; scheduleSave() }
    func duplicate(_ session: ChatSession, through index: Int? = nil) {
        var s = session; s.contextSummary=nil; s.contextThroughID=nil; s.compactRequested=nil; s.queuedMessages=nil; s.queuePaused=nil; s.id = UUID(); s.title += L(" · copia"); s.created = Date(); s.updated = Date(); s.archived = false
        if let index { s.messages = Array(s.messages.prefix(index)) }
        s.messages = s.messages.map { var m = $0; m.id = UUID(); if m.state != "complete" { m.state = "interrupted" }; return m }
        archive.sessions.insert(s, at: 0); archive.selected = s.id; showArchived = false; scheduleSave()
    }
    func deleteSession(_ s: ChatSession) {
        guard !mcpJobs.running(s.id), !commandJobs.running(s.id), !compacting.contains(s.id), terminals[s.id]?.running != true, !agentRuns.contains(s.id), !mcpRuns.contains(s.id), !requests.values.contains(where: { $0.session == s.id }) else { error = "Interrompi la generazione di questa sessione prima di eliminarla."; return }
        let alert = NSAlert(); alert.messageText = L("Eliminare “\(s.title)”?"); alert.informativeText = L("Questa azione elimina la sessione e i suoi messaggi."); alert.addButton(withTitle: L("Elimina")); alert.addButton(withTitle: L("Annulla"))
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        Task{_ = await invokeHooks("SessionEnd",session:s,extra:["reason":"other"])}
        for task in backgroundHooks.removeValue(forKey:s.id)?.values ?? [:].values {task.cancel()}
        hookSessions.remove(s.id);hookContexts.removeValue(forKey:s.id)
        actionLoopGuard.reset(s.id);mcpJobs.stopAll(s.id);commandJobs.stopAll(s.id);terminals.removeValue(forKey:s.id)?.stop();archive.sessions.removeAll { $0.id == s.id }; if archive.selected == s.id { archive.selected = archive.sessions.first?.id }; if archive.sessions.isEmpty { newSession() }; scheduleSave()
    }
    func pickFile(model: Bool) {
        let p = NSOpenPanel(); p.canChooseDirectories = false; p.allowsMultipleSelection = false
        p.allowedContentTypes = model ? [UTType(filenameExtension: "suc1") ?? .data] : [.json]
        guard p.runModal() == .OK, let url = p.url else { return }
        selectFile(url, model: model)
    }
    func selectFile(_ url: URL, model: Bool) {
        let bookmark = try? url.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil)
        if model { archive.settings.tokenizerPath = ""; archive.settings.tokenizerBookmark = nil; archive.settings.modelPath = url.path; archive.settings.modelBookmark = bookmark
            let tok = url.deletingLastPathComponent().appendingPathComponent("tokenizer.json"); if FileManager.default.fileExists(atPath: tok.path) { archive.settings.tokenizerPath = tok.path; archive.settings.tokenizerBookmark = try? tok.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil) }
        } else { archive.settings.tokenizerPath = url.path; archive.settings.tokenizerBookmark = bookmark }
        scheduleSave()
    }
    func loadModel() {
        guard StudioPlatform.localInferenceAvailable else { error = StudioPlatform.localUnavailableMessage; tab = "server"; return }
        guard !hasWork && !serverRunning else { error = "Arresta il server e attendi le richieste prima di cambiare modello."; return }
        func resolve(_ data: Data?, _ fallback: String) -> String {
            guard let data else { return fallback }; var stale = false
            return (try? URL(resolvingBookmarkData: data, options: .withoutUI, relativeTo: nil, bookmarkDataIsStale: &stale).path) ?? fallback
        }
        archive.settings.modelPath = resolve(archive.settings.modelBookmark, archive.settings.modelPath)
        archive.settings.tokenizerPath = resolve(archive.settings.tokenizerBookmark, archive.settings.tokenizerPath)
        guard !archive.settings.modelPath.isEmpty && !archive.settings.tokenizerPath.isEmpty else { error = "Seleziona modello SUC1 e tokenizer.json."; tab = "settings"; return }
        guard FileManager.default.isReadableFile(atPath: archive.settings.modelPath), FileManager.default.isReadableFile(atPath: archive.settings.tokenizerPath) else { error = "Modello o tokenizer non accessibile. Seleziona nuovamente il file."; return }
        if modelReady { loadAfterStart = true; engine.send(["command": "unload"]); return }
        scheduleSave()
        do { try engine.start();engine.send(["command":"set_model_log","enabled":archive.settings.ioLogEnabled]); memoryPlan = ""; expertStats = ""; modelState = "loading"; loadMemory = nil; loadPhase = "Preparazione del caricamento"; loadID = UUID().uuidString;operationJournal.update(id:loadID,title:"Caricamento · "+URL(fileURLWithPath:archive.settings.modelPath).lastPathComponent,state:"In corso")
            let adaptive=archive.settings.prefill=="adaptive"
            var args = ["--prefill", adaptive ? "batch":archive.settings.prefill,"--history-prefill",adaptive ? "auto":"off"]
            if archive.settings.effectiveMemoryMode == "manual" { args += ["--expert-budget", archive.settings.expertBudget] }
            engine.send(["command": "load", "id": loadID, "path": archive.settings.modelPath, "tokenizer": archive.settings.tokenizerPath, "context": archive.settings.context, "memory_mode": archive.settings.effectiveMemoryMode, "runtime_args": args])
        } catch { modelState = "error"; self.error = error.localizedDescription }
    }
    func unload() { guard !hasWork && !serverRunning else { error = "Arresta server e richieste prima di scaricare il modello."; return }; engine.send(["command": "unload"]) }
    func pickAttachments() {
        guard let sessionID = archive.selected else { return }
        let panel = NSOpenPanel(); panel.canChooseDirectories = false; panel.allowsMultipleSelection = true
        panel.message = L("Testo con qualsiasi estensione, Word, OpenDocument e PDF · massimo 5 file, 10 MiB per file")
        guard panel.runModal() == .OK else { return }
        attachFiles(panel.urls, to: sessionID)
    }
    func attachFiles(_ urls: [URL], to sessionID: UUID? = nil) {
        guard !quitting, let id = sessionID ?? archive.selected, let session = archive.sessions.first(where: { $0.id == id }) else { return }
        guard !mcpRuns.contains(id) else { error = "Attendi il risultato dello strumento MCP."; return }
        guard !attachmentLoads.contains(id) else { error = "Attendi la lettura degli allegati in questa sessione."; return }
        guard !urls.isEmpty, urls.count + (session.draftAttachments ?? []).count <= AttachmentReader.maxFiles else { error = "Massimo 5 allegati per messaggio."; return }
        attachmentLoads.insert(id)
        Task {
            let results = await Task.detached(priority: .userInitiated) {
                urls.map { url -> Result<ChatAttachment, Error> in Result { try AttachmentReader.read(url) } }
            }.value
            defer { attachmentLoads.remove(id); scheduleChatQueue(id) }
            guard !quitting, let index = archive.sessions.firstIndex(where: { $0.id == id }) else { return }
            var files = archive.sessions[index].draftAttachments ?? [], failures: [String] = []
            for (url, result) in zip(urls, results) {
                do { let file = try result.get(); try AttachmentReader.validate(files + [file]); files.append(file) }
                catch { failures.append("\(url.lastPathComponent): \(error.localizedDescription)") }
            }
            archive.sessions[index].draftAttachments = files; archive.sessions[index].updated = Date(); scheduleSave()
            if !failures.isEmpty { error = failures.joined(separator: "\n\n") }
        }
    }
    func removeAttachment(_ id: UUID) { mutateSelected { $0.draftAttachments?.removeAll { $0.id == id } } }
    func sendMessage(sessionID: UUID? = nil, toolResult: String? = nil, isToolResult: Bool = true, queuedInput: ChatMessage? = nil, bypassQueue: Bool = false, atBoundary: Bool = false) {
        if toolResult == nil, queuedInput == nil, !bypassQueue, let id=sessionID ?? archive.selected, handleChatCommand(id) {return}
        if toolResult == nil, queuedInput == nil, !bypassQueue, let id=sessionID ?? archive.selected,
           sessionTurnBusy(id) || !(archive.sessions.first(where:{$0.id==id})?.queuedMessages ?? []).isEmpty {
            enqueueDraft(id); return
        }
        guard let s = archive.sessions.first(where: { $0.id == (sessionID ?? archive.selected) }), chatReady, !attachmentLoads.contains(s.id), (toolResult != nil || atBoundary || !agentRuns.contains(s.id)) else { return }
        let followup=toolResult != nil && s.queuePaused != true ? s.queuedMessages?.first:nil
        let steering=atBoundary || followup != nil
        let userTurn=toolResult == nil || !isToolResult || followup != nil
        let newTurn=userTurn && !steering
        let promptText=followup?.text ?? toolResult ?? (queuedInput?.text ?? s.draft.trimmingCharacters(in:.whitespacesAndNewlines))
        let firstHookTurn=userTurn && !hookSessions.contains(s.id)
        if userTurn {hookSessions.insert(s.id)}
        if newTurn { actionLoopGuard.reset(s.id);hookTurnIDs[s.id]=UUID().uuidString;stopHookActive.remove(s.id);agentIterations[s.id] = 0; subtaskCounts[s.id] = 0;agentRepairs[s.id]=0;agentRepairMessages[s.id]=[] }
        guard !mcpRuns.contains(s.id), !requests.values.contains(where: { $0.session == s.id }) else { return }
        if toolResult == nil, let previous = s.messages.last(where: { $0.role == "assistant" && !$0.isUIReport }), !previous.model.isEmpty, previous.model != chatModel {
            let a = NSAlert(); a.messageText = L("Modello diverso per questa risposta"); a.informativeText = L("La sessione usava \(previous.model). La nuova risposta userà \(chatModel)."); a.addButton(withTitle: L("Continua")); a.addButton(withTitle: L("Annulla")); guard a.runModal() == .alertFirstButtonReturn else { return }
        }
        let text = toolResult ?? queuedInput?.text ?? s.draft.trimmingCharacters(in: .whitespacesAndNewlines)
        // Tool results include a small harness header; keep the user-input limit unchanged.
        let messageLimit = toolResult != nil && isToolResult ? MCPConnection.maxResultBytes + 64 * 1024 : 1_000_000
        guard text.utf8.count < messageLimit else { error = "Messaggio troppo grande."; return }
        let files = toolResult == nil ? (queuedInput.map{$0.attachments ?? []} ?? s.draftAttachments ?? []) : []
        guard !text.isEmpty || !files.isEmpty else { return }
        do { try AttachmentReader.validate(files) } catch { self.error = error.localizedDescription; return }
        let historical=s.activeContextMessages
        if let followup {
            guard followup.text.utf8.count<1_000_000 else{error="Messaggio troppo grande.";pauseChatQueue(s.id);return}
            do {try AttachmentReader.validate(followup.attachments ?? [])} catch {self.error=error.localizedDescription;pauseChatQueue(s.id);return}
        }
        let allFiles=files + (followup?.attachments ?? []) + historical.flatMap { $0.attachments ?? [] }
        guard usesRemote || !allFiles.contains(where: \.isImage) else { error="Le immagini richiedono Server remoto con un modello vision. Il motore C++ locale attuale accetta solo testo.";return }
        do { for message in historical { try AttachmentReader.validate(message.attachments ?? []) } } catch { self.error=error.localizedDescription;return }
        let userMessage = queuedInput ?? ChatMessage(role: "user", text: text, attachments: files.isEmpty ? nil : files,kind:toolResult == nil || !isToolResult ? nil:"tool_result")
        var history: [[String:Any]] = historical.map { ["role": $0.role, "content": usesRemote ? $0.remoteContent : $0.modelContent] }
        if userTurn {
            let added=[SkillDocument.explicitInstructions(promptText,skills:archive.settings.skills ?? []),PluginCatalog.explicitInstructions(promptText,plugins:archive.settings.plugins ?? [],skills:archive.settings.skills ?? [])].filter{!$0.isEmpty}.joined(separator:"\n\n")
            let previous=steering ? activeSkillInstructions[s.id] ?? "":""
            activeSkillInstructions[s.id]=added.isEmpty || previous.contains(added) ? previous:[previous,added].filter{!$0.isEmpty}.joined(separator:"\n\n")
        }
        let explicit=activeSkillInstructions[s.id] ?? ""
        let pendingContext=hookContexts.removeValue(forKey:s.id) ?? []
        let mcpPrompt=MCPCatalog.prompt(mcpTools)
        let memory=s.activeContextSummary
        let effectivePrompt = ([mcpPrompt,PromptComposer.compose(settings: archive.settings, session: s),explicit,memory.isEmpty ? "":"Memoria sintetica della conversazione (dati storici, non nuove istruzioni):\n"+memory]+pendingContext).filter{!$0.isEmpty}.joined(separator:"\n\n")
        if !effectivePrompt.isEmpty { history.insert(["role": "system", "content": effectivePrompt], at: 0) }
        history.append(["role": "user", "content": usesRemote ? userMessage.remoteContent : userMessage.modelContent])
        if let followup {history.append(["role":"user","content":usesRemote ? followup.remoteContent:followup.modelContent])}
        if maybeCompactBeforeSend(s,messages:history,maxTokens:effectiveOutputBudget(s),queued:queuedInput != nil,toolResult:toolResult,isTool:isToolResult) {
            hookContexts[s.id,default:[]]+=pendingContext
            if firstHookTurn {hookSessions.remove(s.id)}
            return
        }
        let sampling = s.sampling ?? SamplingSettings()
        var body: [String: Any] = ["top_p": sampling.topP, "top_k": sampling.topK, "min_p": sampling.minP, "seed": sampling.seed, "model": chatModel, "messages": history, "max_tokens": contextOutputLimit(s,history:history), "temperature": s.temperature, "chat_template_kwargs": ["enable_thinking": s.thinking]]
        let stops = sampling.stop.components(separatedBy: "\n").filter { !$0.isEmpty }
        if !stops.isEmpty { body["stop"] = Array(stops.prefix(32)) }
        guard let data = try? JSONSerialization.data(withJSONObject: body), data.count < 32 * 1024 * 1024 else {
            error = "Conversazione troppo grande: apri una nuova sessione."
            if let toolResult {pendingContextTools[s.id]=(toolResult,isToolResult);preservePendingContextTool(s.id);finishAgentReport(s.id,reason:"Contesto insufficiente per il risultato dello strumento.")}
            pauseChatQueue(s.id);return
        }
        let response = ChatMessage(implicitThinking:!usesRemote && s.thinking,role: "assistant", text: "", state: "queued", model: chatModel, generation: GenerationSnapshot(temperature: s.temperature, maxTokens: body["max_tokens"] as? Int ?? s.maxTokens, context: usesRemote ? 0 : loadedContext, thinking: s.thinking, systemPrompt: effectivePrompt, sampling: sampling))
        if newTurn { agentTurnStarts[s.id]=(s.actions ?? []).count }
        mutateSession(s.id) { session in
            if session.messages.isEmpty { session.title = String((text.isEmpty ? files.first?.name ?? "Allegati" : text).prefix(60)) }
            session.messages.append(userMessage);if let followup {session.messages.append(followup);session.queuedMessages?.removeAll{$0.id==followup.id}}; session.messages.append(response); if let queuedInput { session.queuedMessages?.removeAll{$0.id==queuedInput.id} }
            else if toolResult == nil { session.draft = ""; session.draftAttachments = nil }
        }
        let id = UUID().uuidString; requests[id] = (s.id, response.id, Date());operationJournal.update(id:id,title:"Generazione · "+chatModel,state:"In coda",sessionID:s.id)
        let remote = usesRemote ? archive.settings.remote : nil
        if userTurn && (archive.settings.hooks ?? []).contains(where: { $0.enabled && ["UserPromptSubmit","SessionStart"].contains(HookDefinition.event($0.event)) }) {
            requestHookTasks[id] = Task {
                defer { requestHookTasks.removeValue(forKey:id) }
                do {
                    var contexts=[String]()
                    if firstHookTurn {
                        let start=await invokeHooks("SessionStart",session:s,extra:["source":s.messages.isEmpty ? "startup":"resume"])
                        contexts+=start.context
                    }
                    let submitted=await invokeHooks("UserPromptSubmit",session:s,extra:["prompt":promptText])
                    if let deny=submitted.deny {throw ExtensionFiles.failure(deny)}
                    contexts+=submitted.context
                    var request=body
                    if !contexts.isEmpty,var messages=request["messages"] as? [[String:Any]] {
                        let extra=contexts.joined(separator:"\n\n")
                        let combined: String
                        if let system=messages.firstIndex(where:{$0["role"] as? String=="system"}) {
                            combined=(messages[system]["content"] as? String ?? "")+"\n\n"+extra
                            messages[system]["content"]=combined
                        } else {combined=extra;messages.insert(["role":"system","content":combined],at:0)}
                        request["messages"]=messages
                        mutateMessage(s.id,response.id){$0.generation?.systemPrompt=combined}
                    }
                    guard let encoded=try? JSONSerialization.data(withJSONObject:request),encoded.count<32*1024*1024 else{throw ExtensionFiles.failure("Contesto degli hook troppo grande per la richiesta.")}
                    try Task.checkCancellation();dispatchGeneration(id:id,body:request,remote:remote)
                }
                catch { receive(["event":Task.isCancelled ? "cancelled" : "error","id":id,"payload":["message":error.localizedDescription]]) }
            }
        } else { dispatchGeneration(id:id, body:body, remote:remote) }
    }
    // Freeze the target session, cancel its old work and retire IDs before accepting new events.
    func reviseMessage(_ messageID: UUID, regenerate: Bool) async {
        guard let session=selected,!editingSessions.contains(session.id),!mcpRuns.contains(session.id),!attachmentLoads.contains(session.id),
              let index=session.messages.firstIndex(where:{$0.id==messageID}) else{return}
        let userIndex:Int
        if regenerate { guard let previous=session.messages[..<index].lastIndex(where:{$0.role=="user" && !$0.isOperational}) else{return};userIndex=previous }
        else { guard session.messages[index].role=="user" else{return};userIndex=index }
        guard !compacting.contains(session.id) else{error="Interrompi prima la compattazione.";return}
        pauseChatQueue(session.id)
        if let boundary=session.contextThroughID,let compactIndex=session.messages.firstIndex(where:{$0.id==boundary}),userIndex<=compactIndex {mutateSession(session.id){$0.contextSummary=nil;$0.contextThroughID=nil}}
        editingSessions.insert(session.id)
        defer{editingSessions.remove(session.id)}
        let oldTask=agentTasks[session.id]
        stopAgent(session.id)
        for (id,target) in requests.filter({$0.value.session==session.id}) {
            retiredRequests.insert(id);requests.removeValue(forKey:id);operationJournal.update(id:id,state:"Interrotta")
            requestHookTasks[id]?.cancel();remoteTasks[id]?.cancel();remoteClients[id]?.stop()
            engine.send(["command":"cancel","id":id]);finishActivity(id)
            mutateMessage(target.session,target.message){$0.state="interrupted"}
        }
        await oldTask?.value
        guard archive.sessions.contains(where:{$0.id==session.id}),!quitting else{return}
        let original=session.messages[userIndex]
        mutateSession(session.id) { value in
            value.messages=Array(session.messages.prefix(userIndex))
            value.draft=original.text;value.draftAttachments=original.attachments
        }
        if regenerate { sendMessage(sessionID:session.id,bypassQueue:true) }
    }

    func cancelSelected() {
        guard let session = archive.selected else { return }
        pauseChatQueue(session)
        if let target=archive.sessions.first(where:{$0.id==session}) {Task{_ = await invokeHooks("Interrupt",session:target)}}
        compactionTasks[session]?.cancel();compactionClients[session]?.stop();mutateSession(session){$0.compactRequested=nil};if preservePendingContextTool(session) {finishAgentReport(session,reason:"Richiesta interrotta.")}
        stopAgent(session)
        for (id, target) in requests where target.session == session { requestHookTasks[id]?.cancel(); if let client = remoteClients[id] { remoteTasks[id]?.cancel(); client.stop() } else { engine.send(["command": "cancel", "id": id]) }; operationJournal.update(id:id,state:"Interruzione richiesta");mutateMessage(target.session, target.message) { $0.state = "cancelling" } }
    }
    func dispatchGeneration(id: String, body: [String:Any], remote: RemoteConfiguration?) {
        if archive.settings.ioLogEnabled {appendModelIO(id,"INPUT",ModelIOLog.request(body))}
        if let remote { sendRemote(remote,id:id,body:body) } else { engine.send(["command":"generate","id":id,"body":body]) }
    }
    func sendRemote(_ configuration: RemoteConfiguration, id: String, body: [String: Any]) {
        do {
            let client = RemoteInference(configuration, key: try MCPSecrets.read(configuration.id)["key"] ?? "")
            remoteClients[id] = client
            remoteTasks[id] = Task {
                defer { client.stop(); remoteClients.removeValue(forKey: id); remoteTasks.removeValue(forKey: id) }
                do {
                    try Task.checkCancellation()
                    try await client.generate(body) { event in var value = event; value["id"] = id; self.receive(value) }
                    receive(["event":"finished", "id":id, "payload":[:]])
                } catch {
                    if Task.isCancelled { receive(["event":"cancelled", "id":id, "payload":[:]]) }
                    else { receive(["event":"error", "id":id, "payload":["message":error.localizedDescription]]) }
                }
            }
        } catch { receive(["event":"error", "id":id, "payload":["message":error.localizedDescription]]) }
    }
    func startServer() {
        guard StudioPlatform.localInferenceAvailable else { error = StudioPlatform.localUnavailableMessage; tab = "server"; return }
        guard modelReady else { error = "Carica un modello prima di avviare il server."; return }
        do { let key = try KeyStore.read(); serverBusy = true
            engine.send(["command": "start_server", "host": archive.settings.lan ? "0.0.0.0" : "127.0.0.1", "port": archive.settings.port, "key": key])
        } catch { self.error = error.localizedDescription }
    }
    func stopServer() { serverBusy = true; engine.send(["command": "stop_server"]) }
    func copyKey() { do { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(try KeyStore.read(), forType: .string) } catch { self.error = error.localizedDescription } }
    func rotateKey() { guard !serverRunning else { error = "Arresta il server prima di ruotare la chiave."; return }; do { _ = try KeyStore.rotate(); appendLog("API key ruotata.") } catch { self.error = error.localizedDescription } }
    func receive(_ e: [String: Any]) {
        let name = e["event"] as? String ?? "", id = e["id"] as? String ?? "", p = e["payload"] as? [String: Any] ?? [:]
        if consumeCompactionEvent(e) {return}
        guard !retiredRequests.contains(id) else { return }
        switch name {
        case "queued":operationJournal.update(id:id,state:"In coda")
        case "started":operationJournal.update(id:id,state:"In corso")
        case "prefill":operationJournal.update(id:id,state:"Decode")
        case "finished","idle":if !id.isEmpty { let cancelling=requests[id].map{t in archive.sessions.first(where:{$0.id==t.session})?.messages.first(where:{$0.id==t.message})?.state=="cancelling"} ?? false;operationJournal.update(id:id,state:cancelling ? "Interrotta":"Completata") }
        case "cancelled":operationJournal.update(id:id,state:"Interrotta")
        case "loaded":operationJournal.update(id:id.isEmpty ? loadID:id,state:"Completata")
        case "unloaded":operationJournal.update(id:UUID().uuidString,title:"Scaricamento modello",state:"Completata")
        case "server":operationJournal.update(id:UUID().uuidString,title:(p["running"] as? Bool ?? false) ? "Avvio server locale":"Arresto server locale",state:"Completata")
        case "error":if !id.isEmpty {operationJournal.update(id:id,state:"Errore")}
        default:break
        }
        switch name {
        case "expert_stats":
            guard activeRequestID == id else { return }
            let read = (p["bytes_read"] as? Double ?? 0) / 1_073_741_824
            let hit = (p["hit_rate"] as? Double ?? 0) * 100
            let wait = p["wait_s"] as? Double ?? 0
            let phase = p["phase"] as? String == "prefill" ? "Fine prefill" : "Fine richiesta"
            expertStats = String(format: "%@ · SSD %.2f GiB · cache hit %.1f%% · attesa I/O %.2f s", phase, read, hit, wait)
        case "memory_plan":
            func gib(_ key: String) -> String { String(format: "%.2f GiB", (p[key] as? Double ?? 0) / 1_073_741_824) }
            let mode = p["mode"] as? String == "expert_ssd" ? "Streaming SSD expert" : "Residente"
            memoryPlan = "\(mode) · limite prudenziale \(gib("limit_bytes"))\nPesi residenti stimati \(gib("resident_bytes")) · riserva stimata \(gib("reserve_bytes")) · cache expert \(gib("expert_budget_bytes"))"
            appendLog(memoryPlan)
        case "load_progress":
            if id == loadID, modelState == "loading" { loadPhase = p["phase"] as? String ?? loadPhase; if let measured=LoadMemoryStatus(p) {loadMemory=measured} }; return
        case "prefill_progress":
            guard let t=requests[id], let value=PrefillStatus(p),
                  let state=archive.sessions.first(where:{$0.id==t.session})?.messages.first(where:{$0.id==t.message})?.state,
                  ["queued","prefill","remote_wait"].contains(state) else { return }
            if let previous=prefillProgress[t.message],previous.total==value.total,previous.processed>value.processed { return }
            prefillProgress[t.message]=value
            mutateMessage(t.session,t.message) { $0.state="prefill" }
            if id==activeRequestID {activePhase="Prefill"}; return
        case "loaded": loadedContext = p["context"] as? Int ?? archive.settings.context; modelState = "ready"; loadedModel = p["model"] as? String ?? ""; appendLog("Modello pronto: \(loadedModel)")
        case "unloaded": memoryPlan = ""; expertStats = ""; modelState = "unloaded"; loadedModel = ""; if loadAfterStart { loadAfterStart = false; loadModel() }
        case "server": serverRunning = p["running"] as? Bool ?? false; serverBusy = false; if serverRunning { apiAddress = "http://\(p["host"] as? String ?? "127.0.0.1"):\(p["port"] as? Int ?? 8080)/v1" }; appendLog(serverRunning ? "Server avviato: \(apiAddress)" : "Server arrestato")
        case "status": engineFootprint = (p["footprint_bytes"] as? Double).map { $0 / 1_073_741_824 }; return
        case "health":
            guard let target=requests[id],let state=archive.sessions.first(where:{$0.id==target.session})?.messages.first(where:{$0.id==target.message})?.state,["queued","prefill","remote_wait"].contains(state) else{return}
            switch p["status"] as? String {
            case "ok":activePhase="Server raggiungibile · in attesa della risposta"
            case "unsupported":activePhase="Elaborazione remota · health check non disponibile"
            case "unreachable":activePhase="Elaborazione remota · health check senza risposta"
            default:break
            }
            return
        case "model_input":appendModelIO(id,"INPUT API",p["text"] as? String ?? "");return
        case "model_output":appendModelIO(id,"OUTPUT API",p["text"] as? String ?? "",coalesce:true);return
        case "queued": if p["origin"] as? String == "api" { apiRequests.insert(id) }
        case "started": activeRequestID = id; activePhase = remoteClients[id] == nil ? "Prefill" : "Attesa remota"; expertStats = ""; latestMetrics = nil; latestIsCurrent = true; if let t = requests[id] { mutateMessage(t.session, t.message) { if $0.state != "cancelling" { $0.state = remoteClients[id] == nil ? "prefill" : "remote_wait" } } }
        case "reasoning": if let t=requests[id] {
            appendModelIO(id,"REASONING",p["text"] as? String ?? "",coalesce:true)
            mutateMessage(t.session,t.message){message in
                if message.text.isEmpty && (message.reasoning ?? "").isEmpty {var m=message.metrics ?? Metrics();m.firstTextSeconds=Date().timeIntervalSince(t.start);message.metrics=m}
                message.reasoning=(message.reasoning ?? "")+(p["text"] as? String ?? "");if message.state != "cancelling" {message.state="generating"}
            };if id==activeRequestID {activePhase="Ragionamento"}
        }
        case "delta": if let t = requests[id] { appendModelIO(id,"OUTPUT",p["text"] as? String ?? "",coalesce:true);mutateMessage(t.session, t.message) { message in
            if message.text.isEmpty && (message.reasoning ?? "").isEmpty { var m = message.metrics ?? Metrics(); m.firstTextSeconds = Date().timeIntervalSince(t.start); message.metrics = m }
            message.text += p["text"] as? String ?? ""; if message.state != "cancelling" { message.state = "generating" }
        }; if id==activeRequestID {activePhase = "Decode"} }
        case "prefill", "metrics":
            var m = latestMetrics ?? Metrics(); m.promptTokens = p["prompt_tokens"] as? Int ?? 0; m.prefillSeconds = p["prefill_s"] as? Double ?? 0
            m.completionTokens = p["completion_tokens"] as? Int ?? 0; m.reasoningTokens = p["reasoning_tokens"] as? Int ?? m.reasoningTokens; m.decodeSeconds = p["decode_s"] as? Double ?? 0
            m.cachedPromptTokens=RemoteTelemetry.count(p["cached_tokens"]);m.evaluatedPromptTokens=RemoteTelemetry.count(p["prefill_tokens"])
            m.reportedPrefillRate=RemoteTelemetry.positive(p["prefill_tps"]);m.reportedDecodeRate=RemoteTelemetry.positive(p["decode_tps"])
            m.prefillSource=p["prefill_source"] as? String;m.decodeSource=p["decode_source"] as? String;m.remote=p["remote"] as? Bool
            latestMetrics = m
            // Live usage may omit completion_tokens until the final SSE frame.
            // Metrics cannot move an already streaming request back to prefill.
            if id==activeRequestID {
                if name=="prefill" || m.completionTokens>0 {
                    if activePhase != "Ragionamento" {activePhase="Decode"}
                } else if !["Decode","Ragionamento"].contains(activePhase),remoteClients[id]==nil {activePhase="Prefill"}
            }
            if name == "prefill",let target=requests[id] { mutateMessage(target.session,target.message){if $0.state != "cancelling" {$0.state="generating"}} }
            if let t = requests[id] { mutateMessage(t.session, t.message) { message in var value = m; value.firstTextSeconds = message.metrics?.firstTextSeconds; message.metrics = value } }
        case "finish_reason":if let t=requests[id] {mutateMessage(t.session,t.message){$0.finishReason=p["reason"] as? String}}
        case "finished", "cancelled":
            appendModelIO(id,"STATO",name == "finished" ? "Richiesta completata.":"Richiesta interrotta.")
            if let t = requests.removeValue(forKey: id) { let cancelled=archive.sessions.first(where:{$0.id==t.session})?.messages.first(where:{$0.id==t.message})?.state == "cancelling"; mutateMessage(t.session, t.message) { $0.state = name == "finished" && !cancelled ? "complete" : "interrupted"; if let reason=p["reason"] as? String {$0.finishReason=reason} }; if name == "finished" && !cancelled { finishGeneration(sessionID:t.session, messageID:t.message) } else { completeSubagent(t.session, result:.failure(CancellationError()));finishAgentReport(t.session,reason:"Generazione interrotta.") } }; finishActivity(id); completed += 1
        case "idle": apiRequests.remove(id); if id.hasPrefix("api-") { completed += 1;appendModelIO(id,"STATO","Richiesta API completata.") }; finishActivity(id)
        case "error":
            appendModelIO(id,"ERRORE",p["message"] as? String ?? "Errore sconosciuto")
            finishActivity(id)
            let message = p["message"] as? String ?? "Errore sconosciuto"
            if let t = requests.removeValue(forKey: id) { mutateMessage(t.session, t.message) { $0.state = "error"; if $0.text.isEmpty { $0.text = message } }; markSessionUnreadIfNeeded(t.session);completeSubagent(t.session, result:.failure(MCPConnection.failure(message)));finishAgentReport(t.session,reason:"La generazione è terminata con un errore.") }
            else { error = message }; if id == loadID { modelState = "error" }; serverBusy = false; appendLog(message)
        default: break
        }
        scheduleSave()
    }
    private func finishActivity(_ id: String) {
        if let t=requests[id] { prefillProgress.removeValue(forKey:t.message) }
        // Completed requests may already have been removed from the request map.
        let live=Set(requests.values.map { $0.message })
        prefillProgress = prefillProgress.filter { live.contains($0.key) }
        guard id == activeRequestID else { return }
        activeRequestID = ""; latestIsCurrent = false; activePhase = "Pronto"
    }
    func exportSession(json: Bool) {
        guard let session = selected else { return }
        let panel = NSSavePanel(); panel.allowedContentTypes = [json ? .json : .plainText]; panel.nameFieldStringValue = "Chat-\(session.id.uuidString.prefix(8)).\(json ? "json" : "md")"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let data: Data
            if json { let e = JSONEncoder(); e.outputFormatting = [.prettyPrinted, .sortedKeys]; data = try e.encode(SessionExport(session: session)) }
            else { data = Data(("# \(session.title)\n\n" + session.messages.map { "## \($0.role)\n\n\($0.modelContent)" }.joined(separator: "\n\n")).utf8) }
            try data.write(to: url, options: .atomic)
        } catch { self.error = error.localizedDescription }
    }
    func importSession() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0; guard size < 20 * 1024 * 1024 else { throw NSError(domain: "Import", code: 1, userInfo: [NSLocalizedDescriptionKey: "File troppo grande (max 20 MB)."] ) }
            var value = try JSONDecoder().decode(SessionExport.self, from: Data(contentsOf: url)); guard value.version == 1, value.session.messages.count < 10000, (1...32768).contains(value.session.maxTokens), (0...2).contains(value.session.temperature) else { throw NSError(domain: "Import", code: 2, userInfo: [NSLocalizedDescriptionKey: "Formato o parametri non validi."]) }
            try AttachmentReader.validate(value.session.draftAttachments ?? [])
            for message in value.session.messages { try AttachmentReader.validate(message.attachments ?? []) }
            value.session.workspacePath=nil;value.session.workspaceBookmark=nil;value.session.agentEnabled=false;value.session.approvalMode="manual";value.session.allowCommands=false;value.session.allowSubagents=false;value.session.parentSessionID=nil;value.session.subtasks=nil
            duplicate(value.session)
        } catch { self.error = error.localizedDescription }
    }
    func restoreBackup() {
        do { guard FileManager.default.fileExists(atPath: archiveURL.appendingPathExtension("backup").path) else { throw NSError(domain: "Backup", code: 1, userInfo: [NSLocalizedDescriptionKey: "Nessun backup disponibile."]) }; var value = try ArchiveDisk.load(archiveURL.appendingPathExtension("backup")); for i in value.sessions.indices where !(value.sessions[i].queuedMessages ?? []).isEmpty {value.sessions[i].queuePaused=true}; for id in Array(mcpConnections.keys) { disconnectMCP(id) }; archive = value; StudioL10n.language=archive.settings.effectiveLanguage; StudioTypography.apply(archive.settings); setAwake(archive.settings.keepAwake ?? false); saveError = false; scheduleSave(); Task { for configuration in archive.settings.mcpServers ?? [] where configuration.enabled { await connectMCP(configuration) } } } catch { self.error = error.localizedDescription }
    }
    func quit(completion: @escaping (Bool) -> Void) {
        if hasWork { let alert = NSAlert(); alert.messageText = L("Interrompere il lavoro e uscire?"); alert.addButton(withTitle: L("Interrompi ed esci")); alert.addButton(withTitle: L("Annulla")); guard alert.runModal() == .alertFirstButtonReturn else { completion(false); return } }
        quitting = true;mcpJobs.stopAll();commandJobs.stopAll(); for task in compactionTasks.values{task.cancel()};for client in compactionClients.values{client.stop()}; for terminal in terminals.values {terminal.stop()}; schedulingEnabled = false; try? power.set(false); saveTask?.cancel(); for id in Array(agentTasks.keys) { stopAgent(id) }; for task in requestHookTasks.values { task.cancel() }; for task in remoteTasks.values { task.cancel() }; for client in remoteClients.values { client.stop() }; engine.shutdown(); for id in Array(mcpConnections.keys) { disconnectMCP(id) }
        Task {
            await commandJobs.shutdown()
            for tasks in backgroundHooks.values {for task in tasks.values{task.cancel()}}
            for session in archive.sessions where hookSessions.contains(session.id) {_ = await invokeHooks("SessionEnd",session:session,extra:["reason":"other"])}
            do { if !saveError { try await disk.save(archive) } } catch { self.error = error.localizedDescription; quitting = false; completion(false); return }; completion(true) }
    }
}
