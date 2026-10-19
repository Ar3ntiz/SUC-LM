import SwiftUI
import AppKit
import UniformTypeIdentifiers

let studioAccent = Color(red: 0.30, green: 0.43, blue: 0.94)
func copyText(_ text: String) { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(text, forType: .string) }
func rate(_ value: Double?) -> String { value.map { String(format: "%.1f tok/s", $0) } ?? "—" }

struct StudioRoot: View {
    @EnvironmentObject var store: StudioStore
    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 18) {
                Button { store.archive.settings.sectionSidebarCollapsed = !(store.archive.settings.sectionSidebarCollapsed ?? true); store.scheduleSave() } label: {
                    HStack { Image(systemName: "sidebar.left"); if !(store.archive.settings.sectionSidebarCollapsed ?? true) { Text(StudioPlatform.name).font(StudioTypography.ui(.headline)) } }
                }.buttonStyle(.plain).help("Espandi/comprimi sezioni").accessibilityLabel("Espandi o comprimi la barra delle sezioni")
                ForEach(["chat", "server", "settings"], id: \.self) { item in
                    Button { store.tab = item } label: {
                        HStack(spacing: 12) {
                            Image(systemName: item == "chat" ? "bubble.left.and.bubble.right" : item == "server" ? "server.rack" : "slider.horizontal.3").frame(width: 22)
                            if !(store.archive.settings.sectionSidebarCollapsed ?? true) { Text(item.capitalized); Spacer(minLength: 0) }
                        }.padding(10).frame(maxWidth: .infinity, alignment: .leading)
                            .background(store.tab == item ? studioAccent.opacity(0.18) : Color.clear, in: RoundedRectangle(cornerRadius: 10))
                    }.buttonStyle(.plain).help(item.capitalized).accessibilityLabel(item.capitalized)
                }
                Spacer()
                HStack { Circle().fill(store.chatReady ? Color.green : Color.secondary).frame(width: 7, height: 7); if !(store.archive.settings.sectionSidebarCollapsed ?? true) { Text(store.chatStateLabel).font(StudioTypography.ui(.caption)).lineLimit(2) } }.help(store.chatStateLabel)
            }.padding(10).frame(width: (store.archive.settings.sectionSidebarCollapsed ?? true) ? 58 : 154).background(.bar)
            Divider()
            Group {
                switch store.tab {
                case "server": ServerView()
                case "settings": SettingsView()
                default: ChatView()
                }
            }.frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(minWidth: 820, minHeight: 560)
        .background(Color(nsColor: .windowBackgroundColor))
        .font(StudioTypography.ui())
        .tint(studioAccent)
        .preferredColorScheme(store.archive.settings.appearance == "dark" ? .dark : store.archive.settings.appearance == "light" ? .light : nil)
        .safeAreaInset(edge:.top,spacing:0) { GlobalActivityView().environmentObject(store) }
        .alert(StudioPlatform.name, isPresented: Binding(get: { store.error != nil }, set: { if !$0 { store.error = nil } })) {
            Button("OK") { store.error = nil }
        } message: { Text(store.error ?? "") }
    }
}
struct ChatView: View {
    @EnvironmentObject var store: StudioStore
    @State private var sidebar = true
    @State private var availableHeight:CGFloat=560
    @State private var composerHeight:CGFloat=82
    @State private var dragStartHeight:CGFloat?
    @State private var codeInsertion:CodeInsertion?
    @State private var settings = false
    @State private var scrollPolicy=ChatScrollPolicy()
    @State private var bottomTask:Task<Void,Never>?
    var follow:Bool{scrollPolicy.following}
    @State private var dropTarget = false
    @State private var toolSession: ToolSession?
    @State private var preview: ChatAttachment?
    @FocusState private var searchFocused: Bool
    var body: some View {
        HStack(spacing: 0) {
            if sidebar { sessionList.frame(width: 240); Divider() }
            VStack(spacing: 0) {
                HStack {
                    Button { sidebar.toggle() } label: { Image(systemName: "sidebar.left") }.help("Mostra/nascondi sessioni")
                    VStack(alignment: .leading, spacing: 3) {
                        Text(store.selected?.title ?? "Chat").font(StudioTypography.ui(.headline)).lineLimit(1)
                        Text(store.chatModel.isEmpty ? "Scegli un modello in Settings o Server" : (store.usesRemote ? "Remoto · " : "") + store.chatModel).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary).lineLimit(1)
                    }
                    Spacer()
                    if store.hasWork { Text(store.archive.selected.flatMap{store.terminals[$0]}?.running == true ? "Terminale attivo":store.activePhase).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary); ProgressView().controlSize(.small) }
                    if let activity = store.webActivity, activity.sessionID == store.archive.selected { WebActivityButton(activity: activity) { store.showWebBrowser() } }
                    Button { settings.toggle() } label: { Image(systemName: "slider.horizontal.3") }.help("Impostazioni della sessione").popover(isPresented: $settings) { ScrollView { sessionSettings.padding(20) }.frame(width: 410, height: 620) }
                    Menu { Button("Esporta Markdown") { store.exportSession(json: false) }; Button("Esporta JSON") { store.exportSession(json: true) }; Button("Importa sessione…") { store.importSession() } } label: { Image(systemName: "ellipsis.circle") }.menuStyle(.borderlessButton).frame(width: 28)
                }.buttonStyle(.plain).padding(18)
                Divider()
                transcript
                composer
            }
            .dropDestination(for: URL.self) { urls, _ in
                store.attachFiles(urls); return !urls.isEmpty
            } isTargeted: { dropTarget = $0 }
            .overlay { if dropTarget { RoundedRectangle(cornerRadius: 16).fill(studioAccent.opacity(0.12)).overlay(RoundedRectangle(cornerRadius: 16).stroke(studioAccent, style: StrokeStyle(lineWidth: 2, dash: [8]))).overlay(Label("Rilascia i file per allegarli alla chat", systemImage: "doc.badge.plus").padding().background(.regularMaterial, in: Capsule())).padding(8).allowsHitTesting(false) } }
        }.sheet(item: $preview) { AttachmentPreview(file: $0) }.sheet(item: $toolSession) { MCPToolSheet(sessionID: $0.id).environmentObject(store) }
        .onGeometryChange(for:CGFloat.self){$0.size.height} action:{availableHeight=$0}
        .sheet(item:$codeInsertion){target in CodeInsertSheet {code in
            store.mutateSession(target.sessionID){session in
                let original=session.draft as NSString;let offset=min(original.length,max(0,target.offset))
                session.draft=original.replacingCharacters(in:NSRange(location:offset,length:0),with:"\n"+code+"\n")
            }
        }}
    }
    var sessionList: some View {
        VStack(spacing: 12) {
            HStack { Text("Sessioni").font(StudioTypography.ui(.headline)); Spacer(); Button { store.newSession() } label: { Image(systemName: "square.and.pencil") }.buttonStyle(.plain).help("Nuova sessione ⌘N") }.padding(.top, 18)
            TextField("Cerca nelle chat", text: $store.search).textFieldStyle(.roundedBorder).focused($searchFocused).onReceive(NotificationCenter.default.publisher(for: .init("SUCStudioSearch"))) { _ in searchFocused = true }
            Picker("Cronologia", selection: $store.showArchived) { Text("Recenti").tag(false); Text("Archivio").tag(true) }.pickerStyle(.segmented).labelsHidden()
            ScrollViewReader { sessionReader in
            ScrollView {
                LazyVStack(spacing: 5) {
                    ForEach(filteredSessions) { session in
                        Button { store.selectSession(session.id) } label: {
                            HStack(alignment: .top, spacing: 9) {
                                Image(systemName: session.pinned ? "pin.fill" : "bubble.left").foregroundStyle(.secondary).padding(.top, 3)
                                VStack(alignment: .leading, spacing: 5) {
                                    HStack(spacing:6) {
                                        Text(session.title).font(StudioTypography.ui(size:13,weight:.medium)).lineLimit(2)
                                        if session.hasUnreadMessages==true {Circle().fill(Color.yellow).frame(width:7,height:7).accessibilityLabel("Messaggi non letti").help("Nuovi messaggi o approvazioni nella chat")}
                                    }
                                    Text(session.updated, style: .date).font(StudioTypography.ui(.caption2)).foregroundStyle(.secondary)
                                }
                                Spacer(minLength: 0)
                                if store.requests.values.contains(where: { $0.session == session.id }) { ProgressView().controlSize(.mini) }
                            }.padding(11).frame(maxWidth: .infinity, alignment: .leading).background(store.archive.selected == session.id ? studioAccent.opacity(0.14) : Color.clear, in: RoundedRectangle(cornerRadius: 10))
                        }.id(session.id).buttonStyle(.plain).contextMenu {
                            Button("Rinomina…") { rename(session) }
                            Button("Duplica") { store.duplicate(session) }
                            Button(session.pinned ? "Rimuovi preferita" : "Fissa tra preferite") { change(session) { $0.pinned.toggle() } }
                            Button(session.archived ? "Ripristina" : "Archivia") { change(session) { $0.archived.toggle() } }
                            Divider(); Button("Elimina…", role: .destructive) { store.deleteSession(session) }
                        }
                    }
                }
            }
            .scrollIndicators(.visible).accessibilityIdentifier("session-list-scroll")
            .task(id:store.archive.selected) { await Task.yield();guard !Task.isCancelled,let id=store.archive.selected else{return};sessionReader.scrollTo(id) }
            }.frame(minHeight:0,maxHeight:.infinity).layoutPriority(1)
            Text("Privato. Sul tuo Mac.").font(StudioTypography.ui(.caption2)).foregroundStyle(.tertiary).padding(.bottom, 12)
        }.padding(.horizontal, 12).background(Color(nsColor: .controlBackgroundColor).opacity(0.65))
    }
    var filteredSessions: [ChatSession] {
        store.archive.sessions.filter { s in s.archived == store.showArchived && (store.search.isEmpty || s.title.localizedCaseInsensitiveContains(store.search) || s.messages.contains { $0.text.localizedCaseInsensitiveContains(store.search) }) }.sorted { a, b in a.pinned != b.pinned ? a.pinned : a.updated > b.updated }
    }
    func change(_ session: ChatSession, _ action: (inout ChatSession) -> Void) { if let i = store.archive.sessions.firstIndex(where: { $0.id == session.id }) { action(&store.archive.sessions[i]); store.scheduleSave() } }
    func rename(_ session: ChatSession) {
        let alert = NSAlert(); alert.messageText = "Nome della sessione"; let field = NSTextField(string: session.title); field.frame = NSRect(x: 0, y: 0, width: 300, height: 24); alert.accessoryView = field; alert.addButton(withTitle: "Salva"); alert.addButton(withTitle: "Annulla")
        if alert.runModal() == .alertFirstButtonReturn { change(session) { $0.title = String(field.stringValue.prefix(200)) } }
    }
    var transcript: some View {
        ScrollViewReader { reader in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 24) {
                    if store.selected?.messages.isEmpty != false {
                        VStack(spacing: 16) {
                            Image(systemName: "sparkles").font(StudioTypography.ui(size: 38)).foregroundStyle(studioAccent)
                            Text("Uno spazio per pensare.").font(StudioTypography.ui(size: 26, weight: .semibold, design: .rounded))
                            Text("Conversazioni locali, con il tuo modello.\nInizia un messaggio o scegli il modello nei Settings.").multilineTextAlignment(.center).foregroundStyle(.secondary)
                        }.frame(maxWidth: .infinity).padding(.top, 80).padding(.bottom, 50)
                    }
                    ForEach((store.selected?.messages ?? []).filter { $0.visibleInChat(agent:store.selected?.agentEnabled == true,web:store.selected?.webToolsEnabled == true) }) { message in
                        MessageCard(message: message, edit: { edit(message) }, regenerate: { regenerate(message) })
                    }
                    if let last=store.selected?.messages.last,!last.visibleInChat(agent:store.selected?.agentEnabled==true,web:store.selected?.webToolsEnabled==true),["queued","prefill","remote_wait","generating"].contains(last.state) { GenerationActivity(state:last.state) }
                    ForEach(store.approvals.filter{$0.sessionID==store.archive.selected}) { approval in ActionApprovalView(approval:approval) }
                    if let id=store.archive.selected,let terminal=store.terminals[id] {
                        TerminalSlot(terminal:terminal).id(terminal.id)
                    }
                    if let artifacts=store.selected?.artifacts,!artifacts.isEmpty {ArtifactCards(artifacts:artifacts)}
                    Color.clear.frame(height: 1).id("bottom")
                }.frame(maxWidth: 800).padding(24).frame(maxWidth: .infinity)
            }
            .accessibilityIdentifier("chat-transcript-scroll")
            .defaultScrollAnchor(.bottom,for:.initialOffset)
            .onScrollPhaseChange { _, phase in
                if phase == .tracking || phase == .interacting || phase == .decelerating {
                    scrollPolicy.userScrolled();bottomTask?.cancel()
                }
            }
            .onScrollGeometryChange(for:ChatScrollExtent.self) { geometry in
                ChatScrollExtent(contentHeight:geometry.contentSize.height,viewportHeight:geometry.containerSize.height,viewportWidth:geometry.containerSize.width)
            } action: { _, _ in requestBottom(reader) }
            .onChange(of:store.selected?.messages.last?.text) { _,_ in requestBottom(reader) }
            .onChange(of:store.selected?.messages.count) { _,_ in requestBottom(reader) }
            .onChange(of:store.approvals.map(\.id)) { _,_ in requestBottom(reader) }
            .onChange(of:store.archive.selected.flatMap{store.terminals[$0]?.output}) { _,_ in requestBottom(reader) }
            .onChange(of:store.archive.selected) { _,id in if let id {store.markSessionRead(id)};scrollPolicy.resume();requestBottom(reader) }
            .onChange(of:follow) { _,value in if value {requestBottom(reader)} }
            .onAppear { if let id=store.archive.selected {store.markSessionRead(id)};requestBottom(reader) }
            .onDisappear {bottomTask?.cancel()}
            .overlay(alignment:.bottomTrailing) { if !follow {
                Button {scrollPolicy.resume();requestBottom(reader)} label:{Label("Vai alla fine",systemImage:"arrow.down")}.buttonStyle(.bordered).padding()
            } }
            .frame(minHeight:0,maxHeight:.infinity)

        }
    }
    func requestBottom(_ reader:ScrollViewProxy) {
        guard follow else{return}
        bottomTask?.cancel()
        let sessionID=store.archive.selected
        bottomTask=Task { @MainActor in
            // Wait for the new text/Markdown layout; a gesture cancels this pending move.
            do {try await Task.sleep(nanoseconds:16_000_000)} catch{return}
            guard !Task.isCancelled,follow,store.archive.selected==sessionID else{return}
            var transaction=Transaction();transaction.disablesAnimations=true
            withTransaction(transaction){reader.scrollTo("bottom",anchor:.bottom)}
        }
    }
    func edit(_ message: ChatMessage) { Task { await store.reviseMessage(message.id,regenerate:false) } }
    func regenerate(_ message: ChatMessage) { scrollPolicy.resume();Task { await store.reviseMessage(message.id,regenerate:true) } }
    var composer: some View {
        VStack(spacing: 8) {
            VStack(spacing: 6) {
                if !(store.selected?.draftAttachments ?? []).isEmpty {
                    ScrollView(.horizontal) { HStack {
                        ForEach(store.selected?.draftAttachments ?? []) { file in
                            HStack(spacing: 8) {
                                Button { preview = file } label: { AttachmentLabel(file:file) }.buttonStyle(.plain).help("Anteprima del contenuto che leggerà il modello")
                                Button { store.removeAttachment(file.id) } label: { Image(systemName: "xmark.circle.fill") }.buttonStyle(.plain).help("Rimuovi \(file.name)")
                            }.font(StudioTypography.ui(.caption)).padding(9).background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
                        }
                    } }
                }
                if store.archive.selected.map({ store.mcpRuns.contains($0) }) == true { HStack { ProgressView().controlSize(.small); Text("Strumento MCP in esecuzione…").font(StudioTypography.ui(.caption)); Spacer() } }
                if store.archive.selected.map({ store.agentRuns.contains($0) }) == true { HStack { ProgressView().controlSize(.small); Text("Azione del modello in corso…").font(StudioTypography.ui(.caption)); Spacer(); Button("Interrompi") { store.cancelSelected() } } }
                if store.selectedAttachmentsLoading { HStack { ProgressView().controlSize(.small); Text("Lettura degli allegati…").font(StudioTypography.ui(.caption)); Spacer() } }
                Capsule().fill(Color.secondary.opacity(0.35)).frame(width:36,height:4).frame(maxWidth:.infinity).frame(height:14)
                    .contentShape(Rectangle()).help("Trascina verso l’alto per ampliare il messaggio")
                    .gesture(DragGesture(minimumDistance:1).onChanged{value in
                        if dragStartHeight==nil{dragStartHeight=composerHeight}
                        composerHeight=min(max(82,availableHeight*0.55),max(82,(dragStartHeight ?? 82)-value.translation.height))
                    }.onEnded{_ in dragStartHeight=nil})
                ComposeEditor(text: Binding(get: { store.selected?.draft ?? "" }, set: { value in store.mutateSelected { $0.draft = value } }), onSend: { store.sendMessage(); scrollPolicy.resume() }, onFiles: { store.attachFiles($0) },onCode:{offset in if let id=store.archive.selected{codeInsertion=CodeInsertion(sessionID:id,offset:offset)}}).frame(height:min(composerHeight,max(82,availableHeight*0.55)))
                HStack {
                    Button { store.pickAttachments() } label: { Image(systemName: "paperclip") }.buttonStyle(.plain).help("Allega immagini, testo, Word, OpenDocument o PDF").disabled(store.selectedAttachmentsLoading)
                    Menu {ForEach((store.archive.settings.skills ?? []).filter{$0.enabled}){skill in
                        Button(skill.displayName ?? skill.name){store.mutateSelected{$0.draft += ($0.draft.isEmpty ? "":" ")+"@"+skill.name+" "+(skill.defaultPrompt ?? "")}}
                    }} label:{Image(systemName:"sparkles")}.menuStyle(.borderlessButton).fixedSize().help("Richiama una skill attiva")
                    Button { if let id = store.archive.selected { toolSession = ToolSession(id: id) } } label: { Image(systemName: "wrench.and.screwdriver") }.buttonStyle(.plain).help("Strumenti MCP")
                    Text("Invio per inviare · ⇧ Invio a capo").font(StudioTypography.ui(.caption2)).foregroundStyle(.tertiary)
                    Spacer()
                    if store.requests.values.contains(where: { $0.session == store.archive.selected }) {
                        Button { store.cancelSelected() } label: { Label("Interrompi", systemImage: "stop.fill") }.buttonStyle(.bordered)
                    } else {
                        Button { store.sendMessage(); scrollPolicy.resume() } label: { Image(systemName: "arrow.up").fontWeight(.semibold).frame(width: 24, height: 24) }.buttonStyle(.borderedProminent).clipShape(Circle()).disabled(!store.canSendSelected).help("Invia messaggio")
                    }
                }
            }.padding(12).background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 16)).overlay(RoundedRectangle(cornerRadius: 16).stroke(.quaternary))
            if !(store.selected?.draftAttachments ?? []).isEmpty { Text("Il testo degli allegati occupa il contesto del modello. Nessun taglio automatico.").font(StudioTypography.ui(.caption2)).foregroundStyle(.secondary) }
            MetricsStrip(metrics: store.selected?.messages.last?.metrics, current: store.requests.values.contains { $0.session == store.archive.selected },workspace:store.selected?.workspacePath ?? "")
        }.frame(maxWidth: 830).padding(.horizontal, 20).padding(.bottom, 14).frame(maxWidth: .infinity)
    }
    func samplingBinding<T>(_ key: WritableKeyPath<SamplingSettings, T>) -> Binding<T> {
        Binding(get: { (store.selected?.sampling ?? SamplingSettings())[keyPath: key] }, set: { value in store.mutateSelected { session in var s = session.sampling ?? SamplingSettings(); s[keyPath: key] = value; session.sampling = s } })
    }
    var sessionSettings: some View {
        VStack(alignment: .leading, spacing: 15) {
            WorkspaceSettingsView()
            Divider()
            Text("Questa sessione").font(StudioTypography.ui(.headline))
            Text("System prompt extra della sessione").font(StudioTypography.ui(.caption))
            Text("Si aggiunge al prompt generale e alle skill attive nei Settings.").font(StudioTypography.ui(.caption2)).foregroundStyle(.secondary)
            TextEditor(text: Binding(get: { store.selected?.systemPrompt ?? "" }, set: { v in store.mutateSelected { $0.systemPrompt = v } })).frame(height: 100).overlay(RoundedRectangle(cornerRadius: 4).stroke(.quaternary))
            HStack { Text("Temperatura"); Spacer(); Text(String(format: "%.2f", store.selected?.temperature ?? 0.7)).monospacedDigit() }
            Slider(value: Binding(get: { store.selected?.temperature ?? 0.7 }, set: { v in store.mutateSelected { $0.temperature = v } }), in: 0...2)
            Stepper("Output max: \(store.selected?.maxTokens ?? 4000)", value: Binding(get: { store.selected?.maxTokens ?? 4000 }, set: { v in store.mutateSelected { $0.maxTokens = v } }), in: 1...32768, step: 128)
            Toggle("Thinking (se supportato)", isOn: Binding(get: { store.selected?.thinking ?? false }, set: { v in store.mutateSelected { $0.thinking = v } }))
            DisclosureGroup("Sampling avanzato") {
                VStack(alignment:.leading,spacing:9) {
                    TextField("Top-p", value: samplingBinding(\.topP), format: .number)
                    Text("0–1: mantiene il nucleo di token con probabilità cumulativa indicata. Valori più bassi rendono la risposta più focalizzata; 1 non applica il filtro.")
                    TextField("Top-k (0 = disattivo)", value: samplingBinding(\.topK), format: .number)
                    Text("Limita la scelta ai K token più probabili. Parti da 20–50 per contenere la varietà; 0 disattiva il filtro.")
                    TextField("Min-p", value: samplingBinding(\.minP), format: .number)
                    Text("0–1: scarta token molto meno probabili del migliore. 0 disattiva; 0,05–0,1 è un intervallo prudente.")
                    TextField("Seed (-1 = casuale)", value: samplingBinding(\.seed), format: .number.grouping(.never))
                    Text("Usa lo stesso intero per risultati ripetibili a parità di modello e parametri. −1 sceglie un seed casuale.")
                    TextField("Stop (una per riga)", text: samplingBinding(\.stop), axis: .vertical).lineLimit(1...3)
                    Text("La generazione termina quando compare una delle sequenze indicate. Temperatura, Top-p, Top-k e Min-p si combinano: modifica un parametro alla volta.")
                }.font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            }
            Text("Il contesto viene ricostruito da questa sessione. Se supera il limite, il motore restituisce un errore senza troncarlo.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
        }
    }
}
struct MessageCard: View {
    @EnvironmentObject var store: StudioStore
    @State private var preview: ChatAttachment?
    let message: ChatMessage
    var edit: () -> Void
    var regenerate: () -> Void
    var body: some View {
        Group {
            if message.isUIReport {ActivityMessageBox(message:message)}
            else {regularMessage}
        }
    }
    private var regularMessage: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                Label(message.isUIReport ? "Attività" : message.role == "user" ? "Tu" : "SUC", systemImage: message.isUIReport ? "list.bullet.clipboard" : message.role == "user" ? "person.crop.circle" : "sparkles").font(StudioTypography.ui(size: 12, weight: .semibold)).foregroundStyle(.secondary)
                Spacer()
                if !["complete"].contains(message.state) { Text(["queued":"In coda", "prefill":"Prefill…", "remote_wait":"Attesa remota…", "generating":"Generazione…", "cancelling":"Interruzione in attesa del motore…", "interrupted":"Interrotta", "error":"Errore"][message.state] ?? message.state).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary) }
            }
            ForEach(message.attachments ?? []) { file in
                Button { preview = file } label: { AttachmentLabel(file:file) }.buttonStyle(.bordered)
            }
            if message.role == "assistant",["queued","prefill","remote_wait","generating"].contains(message.state) { GenerationActivity(state:message.state) }
            if !message.text.isEmpty || !(message.reasoning ?? "").isEmpty { ResponseBody(text:message.text,reasoning:message.role=="assistant" ? message.reasoning:nil,implicit:message.role=="assistant" && message.implicitThinking==true,assistant:message.role=="assistant",generating:message.state != "complete") }

            HStack(spacing: 14) {
                Button { copyText(message.text) } label: { Image(systemName: "doc.on.doc") }.help("Copia")
                if message.isUIReport { EmptyView() }
                else if message.role == "user" { Button(action: edit) { Image(systemName: "pencil") }.help("Modifica nella stessa sessione") }
                else { Button(action: regenerate) { Image(systemName: "arrow.clockwise") }.help("Rigenera nella stessa sessione") }
                Spacer()
                if let m = message.metrics, message.state == "complete" { Text("Prefill \(rate(m.prefillRate)) · Decode \(m.decodeSource?.hasPrefix("client_estimate") == true ? "≈ " : "")\(rate(m.decodeRate))").font(StudioTypography.ui(.caption2)).foregroundStyle(.tertiary) }
            }.buttonStyle(.plain).foregroundStyle(.secondary).font(StudioTypography.ui(.caption))
        }.padding(message.role == "user" ? 16 : 0).background(message.role == "user" ? Color.secondary.opacity(0.07) : Color.clear, in: RoundedRectangle(cornerRadius: 14)).sheet(item: $preview) { AttachmentPreview(file: $0) }
    }
}
struct MetricsStrip: View {
    var metrics: Metrics?
    var current = false
    var workspace=""
    var body: some View {
        HStack(spacing: 16) {
            Text("Prefill \(rate(metrics?.prefillRate))")
            Text("Decode \(metrics?.decodeSource?.hasPrefix("client_estimate") == true ? "≈ " : "")\(rate(metrics?.decodeRate))")
            if !workspace.isEmpty {Text("work directory: \(workspace)").lineLimit(1).truncationMode(.middle).help(workspace)}
            if let metrics { Text("\(metrics.completionTokens) token"); if let ttft = metrics.firstTextSeconds { Text(String(format: "Primo testo %.2f s", ttft)) } }
            Spacer(minLength: 0)
            Text(metrics?.remote == true ? (metrics?.decodeSource?.hasPrefix("client_estimate") == true ? "stima client" : "misure server · — non disponibile") : (current ? "misure a fine fase" : "ultima risposta"))
        }.font(StudioTypography.ui(size: 10, design: .monospaced)).foregroundStyle(.secondary).lineLimit(1).minimumScaleFactor(0.75).accessibilityElement(children: .combine)
    }
}
struct ComposeEditor: NSViewRepresentable {
    @EnvironmentObject var store: StudioStore
    @Binding var text: String
    var onSend: () -> Void
    var onFiles: ([URL]) -> Void
    var onCode: (Int)->Void = {_ in}
    func makeCoordinator() -> Coordinator { Coordinator(self) }
    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSScrollView(); scroll.hasVerticalScroller = true; scroll.drawsBackground = false
        let view = SendTextView(); view.isRichText = false; view.drawsBackground = false; view.font = StudioTypography.nsChat; view.textColor = .textColor; view.isVerticallyResizable = true; view.isHorizontallyResizable = false; view.autoresizingMask = [.width]; view.textContainer?.widthTracksTextView = true; view.textContainerInset = NSSize(width: 3, height: 6); view.delegate = context.coordinator; view.onSend = onSend; view.onFiles = onFiles;view.onCode=onCode;view.isAutomaticLinkDetectionEnabled=false;view.isAutomaticDataDetectionEnabled=false;view.isContinuousSpellCheckingEnabled=false;view.isGrammarCheckingEnabled=false;view.isAutomaticQuoteSubstitutionEnabled=false;view.isAutomaticDashSubstitutionEnabled=false;view.layoutManager?.allowsNonContiguousLayout=true; view.registerForDraggedTypes([.fileURL]); view.setAccessibilityLabel("Messaggio")
        scroll.documentView = view; return scroll
    }
    func updateNSView(_ scroll: NSScrollView, context: Context) { context.coordinator.parent = self; if let view = scroll.documentView as? SendTextView { if view.font != StudioTypography.nsChat {view.font=StudioTypography.nsChat}; if view.string != text { view.string = text }; view.onSend = onSend; view.onFiles = onFiles;view.onCode=onCode } }
    final class Coordinator: NSObject, NSTextViewDelegate { var parent: ComposeEditor; init(_ p: ComposeEditor) { parent = p }; func textDidChange(_ n: Notification) { if let v = n.object as? NSTextView { parent.text = v.string } } }
    final class SendTextView: NSTextView {
        var onSend: (() -> Void)?
        var onFiles: (([URL]) -> Void)?
        var onCode:((Int)->Void)?
        override func insertText(_ insertString:Any,replacementRange:NSRange){
            super.insertText(insertString,replacementRange:replacementRange)
            guard insertString as? String == "`",!hasMarkedText() else{return}
            let offset=selectedRange().location
            guard offset>=3,(string as NSString).substring(with:NSRange(location:offset-3,length:3))=="```" else{return}
            let range=NSRange(location:offset-3,length:3)
            guard shouldChangeText(in:range,replacementString:"") else{return}
            textStorage?.replaceCharacters(in:range,with:"");setSelectedRange(NSRange(location:range.location,length:0));didChangeText()
            let insertion=range.location,callback=onCode;DispatchQueue.main.async{callback?(insertion)}
        }
        private func files(_ sender: NSDraggingInfo) -> [URL] { (sender.draggingPasteboard.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL]) ?? [] }
        override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation { files(sender).isEmpty ? super.draggingEntered(sender) : .copy }
        override func draggingUpdated(_ sender: NSDraggingInfo) -> NSDragOperation { files(sender).isEmpty ? super.draggingUpdated(sender) : .copy }
        override func prepareForDragOperation(_ sender: NSDraggingInfo) -> Bool { !files(sender).isEmpty || super.prepareForDragOperation(sender) }
        override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
            let urls = files(sender); guard !urls.isEmpty else { return super.performDragOperation(sender) }
            onFiles?(urls); return true
        }
        override func keyDown(with event: NSEvent) { if event.keyCode == 36 && !event.modifierFlags.contains(.shift) && !hasMarkedText() { onSend?() } else { super.keyDown(with: event) } }
    }
}

func highlight(_ source: String) -> AttributedString {
    let value = NSMutableAttributedString(string: source)
    guard source.utf8.count < 100_000 else { return AttributedString(source) }
    let rules: [(String, NSColor)] = [
        (#"\b(import|from|class|struct|func|def|let|var|const|return|if|else|for|while|async|await|try|throw|true|false|null|nil|public|private|static|void|int|float|double)\b"#, .systemPurple),
        (#"\b[0-9]+(?:\.[0-9]+)?\b"#, .systemOrange),
        (#""(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'"#, .systemGreen),
        (#"(?m)//[^\n]*|#[^\n]*"#, .secondaryLabelColor)
    ]
    for (pattern, color) in rules { if let regex = try? NSRegularExpression(pattern: pattern) { for match in regex.matches(in: source, range: NSRange(location: 0, length: (source as NSString).length)) { value.addAttribute(.foregroundColor, value: color, range: match.range) } } }
    return (try? AttributedString(value, including: \.appKit)) ?? AttributedString(source)
}

struct AttachmentPreview: View {
    let file: ChatAttachment
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack { Label(file.name, systemImage: "doc.text").font(StudioTypography.ui(.headline)); Spacer(); Button("Chiudi") { dismiss() }.keyboardShortcut(.cancelAction) }
            if let data=file.imageData,let image=NSImage(data:data) {
                Text("Immagine JPEG · \(file.imageWidth ?? 0) × \(file.imageHeight ?? 0) · \(ByteCountFormatter.string(fromByteCount:Int64(data.count),countStyle:.binary)) · richiede un modello vision remoto").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                Image(nsImage:image).resizable().scaledToFit().frame(maxWidth:.infinity,maxHeight:420)
            } else {
                Text("Testo estratto localmente · \(file.textBytes) byte" + (file.pages.map { " · \($0) pagine PDF" } ?? "")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                ScrollView { Text(file.text).font(StudioTypography.ui(.body, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
            }
        }.padding(24).frame(minWidth: 500, idealWidth: 680, minHeight: 350, idealHeight: 520)
    }
}

struct ToolSession: Identifiable { let id: UUID }

/// Globe button next to the session settings: visible only while a web action runs in this session or a Google verification waits (1.4.5_Final).
struct WebActivityButton: View {
    let activity: WebActivity
    let open: () -> Void
    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { context in
            Button(action: open) {
                HStack(spacing: 5) {
                    Image(systemName: activity.verification ? "exclamationmark.shield" : "globe")
                    if !activity.verification { ProgressView().controlSize(.mini) }
                    Text(activity.verification ? WebTools.verificationLabel : "\(activity.phase) · \(activity.elapsed(context.date)) s").font(StudioTypography.ui(.caption)).lineLimit(1)
                }
                .padding(.horizontal, 8).padding(.vertical, 3)
                .foregroundStyle(activity.verification ? Color.orange : Color.secondary)
                .background(Capsule().fill(activity.verification ? Color.orange.opacity(0.15) : Color.secondary.opacity(0.1)))
            }.buttonStyle(.plain).help(activity.label(now: context.date) + " — clicca per guardare il browser").accessibilityLabel(activity.label(now: context.date))
        }
    }
}
