import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct ServerView: View {
    @EnvironmentObject var store: StudioStore
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                BackendSettingsView()
                if !store.usesRemote {
                HStack {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Il tuo motore. La tua API.").font(StudioTypography.ui(size: 28, weight: .semibold, design: .rounded))
                        Text("Chat e client condividono un solo modello in memoria.").foregroundStyle(.secondary)
                    }
                    Spacer()
                    Label(store.serverRunning ? "Online" : "Spento", systemImage: "circle.fill").font(StudioTypography.ui(.caption)).foregroundStyle(store.serverRunning ? .green : .secondary).padding(10).background(.quaternary, in: Capsule())
                }
                GroupBox {
                    VStack(alignment: .leading, spacing: 18) {
                        LabeledContent("Modello", value: store.loadedModel.isEmpty ? "Nessuno" : store.loadedModel)
                        Picker("Accesso", selection: $store.archive.settings.lan) { Text("Questo Mac").tag(false); Text("Rete locale").tag(true) }.pickerStyle(.segmented).disabled(store.serverRunning)
                        HStack { Text("Porta"); TextField("8080", value: $store.archive.settings.port, format: .number.grouping(.never)).frame(width: 100).textFieldStyle(.roundedBorder).disabled(store.serverRunning); Spacer(); Button(store.serverRunning ? "Arresta server" : "Avvia server") { store.serverRunning ? store.stopServer() : store.startServer() }.buttonStyle(.borderedProminent).disabled(!store.modelReady || store.serverBusy) }
                        if store.archive.settings.lan { Label("HTTP locale non cifrato: usa solo una rete fidata. API key obbligatoria.", systemImage: "network").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary) }
                        if store.serverRunning { HStack { Text(store.apiAddress).font(StudioTypography.ui(.body, design: .monospaced)).textSelection(.enabled); Button { copyText(store.apiAddress) } label: { Image(systemName: "doc.on.doc") } }; if store.archive.settings.lan { Text("Da un altro dispositivo sostituisci 0.0.0.0 con l’indirizzo IPv4 di questo Mac.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary) } }
                    }.padding(12)
                }
                HStack(spacing: 18) { StatTile(title: "Richieste API", value: "\(store.apiRequests.count)"); StatTile(title: "Completate", value: "\(store.completed)"); StatTile(title: "Prefill", value: rate(store.latestMetrics?.prefillRate)); StatTile(title: "Decode", value: rate(store.latestMetrics?.decodeRate)) }
                Text(store.latestIsCurrent ? "Misure della richiesta corrente, disponibili a fine fase." : "Velocità dell’ultima richiesta; decode = media dei passi successivi al primo token.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                GroupBox("API key") { HStack { Text("Salvata nel Keychain. Mai inclusa negli export.").foregroundStyle(.secondary); Spacer(); Button("Copia chiave") { store.copyKey() }; Button("Ruota chiave") { store.rotateKey() }.disabled(store.serverRunning) }.padding(12) }
                GroupBox("Client compatibili OpenAI") {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("POST /v1/chat/completions · POST /v1/completions\nGET /v1/models · GET /health").font(StudioTypography.ui(.caption, design: .monospaced)).textSelection(.enabled)
                        Text("Streaming SSE e usage supportati. Tools, embeddings e Responses non sono implementati.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                        Button("Copia esempio curl") { copyText("curl http://127.0.0.1:\(store.archive.settings.port)/v1/chat/completions -H 'Authorization: Bearer YOUR_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"\(store.loadedModel)\",\"messages\":[{\"role\":\"user\",\"content\":\"Ciao\"}],\"max_tokens\":128,\"stream\":true}'") }
                    }.frame(maxWidth: .infinity, alignment: .leading).padding(12)
                }
                }
                OperationLogView()
                DisclosureGroup("Registro eventi (ultimi 150)") { VStack(alignment: .leading, spacing: 5) { ForEach(Array(store.logs.enumerated()), id: \.offset) { _, text in Text(text).font(StudioTypography.ui(.caption, design: .monospaced)).textSelection(.enabled) } }.frame(maxWidth: .infinity, alignment: .leading) }
            }.padding(30).frame(maxWidth: 1050).frame(maxWidth: .infinity)
        }.onChange(of: store.archive.settings.lan) { _, _ in store.scheduleSave() }.onChange(of: store.archive.settings.port) { _, _ in store.scheduleSave() }
    }
}
struct StatTile: View {
    let title: String
    let value: String
    var body: some View { VStack(alignment: .leading, spacing: 8) { Text(title).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary); Text(value).font(StudioTypography.ui(size: 21, weight: .medium, design: .rounded)).monospacedDigit().lineLimit(1).minimumScaleFactor(0.7) }.frame(maxWidth: .infinity, alignment: .leading).padding(18).background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 12)) }
}
struct SettingsView: View {
    @State private var showHooks = false
    @State private var showPlugins = false
    @State private var showRoutines = false
    @EnvironmentObject var store: StudioStore
    var body: some View {
        Form {
            Section("Modello locale") {
                HStack { VStack(alignment: .leading, spacing: 5) { Text("Modello SUC1").fontWeight(.medium); Text(store.archive.settings.modelPath.isEmpty ? "Nessun file selezionato" : store.archive.settings.modelPath).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary).textSelection(.enabled) }; Spacer(); Button("Scegli…") { store.pickFile(model: true) } }
                HStack { VStack(alignment: .leading, spacing: 5) { Text("Tokenizer JSON").fontWeight(.medium); Text(store.archive.settings.tokenizerPath.isEmpty ? "Seleziona tokenizer.json del modello" : store.archive.settings.tokenizerPath).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary).textSelection(.enabled) }; Spacer(); Button("Scegli…") { store.pickFile(model: false) } }
                HStack { Label(store.stateLabel, systemImage: store.modelReady ? "checkmark.circle.fill" : "cube"); Spacer(); if store.modelState == "loading" { ProgressView().controlSize(.small) }; Button("Scarica dalla memoria") { store.unload() }.disabled(!store.modelReady || store.hasWork || store.serverRunning); Button(store.modelReady ? "Ricarica selezione" : "Carica modello") { store.loadModel() }.buttonStyle(.borderedProminent).disabled(store.hasWork || store.serverRunning) }
                Text("I file restano nella loro posizione originale. Il tokenizer viene verificato contro l’hash del modello, quando presente. Nessun download automatico.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            }
            Section("Memoria e prestazioni · applicate al prossimo caricamento") {
                TextField("Contesto massimo (token)", value: $store.archive.settings.context, format: .number.grouping(.never))
                Picker("Prefill", selection: $store.archive.settings.prefill) { Text("Batch (veloce)").tag("batch"); Text("Exact (se supportato)").tag("exact"); Text("Token").tag("token") }
                Picker("Gestione memoria", selection: Binding(get: { store.archive.settings.effectiveMemoryMode }, set: { store.archive.settings.memoryMode = $0; store.scheduleSave() })) {
                    Text("Automatico").tag("auto")
                    Text("Residente").tag("resident")
                    Text("Streaming expert · budget manuale").tag("manual")
                }
                if store.archive.settings.effectiveMemoryMode == "manual" { TextField("Cache expert, es. 8G", text: $store.archive.settings.expertBudget) }
                if !store.memoryPlan.isEmpty { Text("Ultimo piano di caricamento\n" + store.memoryPlan).font(StudioTypography.ui(.caption)).textSelection(.enabled) }
                if !store.expertStats.isEmpty { Text(store.expertStats).font(StudioTypography.ui(.caption)).textSelection(.enabled) }
                Text("Automatico sceglie residente o streaming SSD per gli expert MoE. I modelli densi sono supportati se rientrano nella memoria stimata; lo streaming dei loro pesi non è disponibile. La stima include contesto e margine, ma non garantisce memoria libera durante l’uso.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                Text("Il contesto e gli expert consumano memoria oltre ai pesi. Prefill batch può differire numericamente dal percorso exact. Il modello deve rientrare nelle risorse del Mac.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            }
            Section("System prompt generale") {
                TextEditor(text: Binding(get: { store.archive.settings.globalSystemPrompt ?? "" }, set: { store.archive.settings.globalSystemPrompt = $0; store.scheduleSave() })).frame(minHeight: 110)
                Text("Usato in tutte le chat dell’app, insieme alle skill attive e al prompt extra di ogni sessione. Le modifiche si applicano dal prossimo messaggio.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            }
            ExtensionSettingsView()
            Section("Plugin e routine") { Button("Gestisci plugin…") { showPlugins = true }.sheet(isPresented:$showPlugins) { PluginManagerView().environmentObject(store) }; Button("Gestisci routine…") { showRoutines = true }.sheet(isPresented:$showRoutines) { RoutinesView().environmentObject(store) } }
            Section("Azioni del modello") {
                Stepper("Azioni consecutive: \(store.archive.settings.actionLimit)",value:Binding(get:{store.archive.settings.actionLimit},set:{store.archive.settings.maxConsecutiveActions=$0;store.scheduleSave()}),in:1...100)
                Text("Massimo per turno e per subagent, da 1 a 100; predefinito 12. Le modifiche si applicano dalla prossima azione. Le approvazioni restano indipendenti da questo limite.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            }
            Section("Automazione e hook") { Button("Gestisci hook…") { showHooks = true }.sheet(isPresented:$showHooks) { HooksSettingsView().environmentObject(store) } }
            LifecycleSettingsView()
            Section("Aspetto") {
                TypographySettingsView()
                Picker("Tema", selection: $store.archive.settings.appearance) { Text("Sistema").tag("system"); Text("Chiaro").tag("light"); Text("Scuro").tag("dark") } }
            Section("Sessioni e dati") {
                LabeledContent("Sessioni salvate", value: "\(store.archive.sessions.count)")
                HStack { Button("Importa sessione…") { store.importSession() }; Button("Apri cartella dati") { NSWorkspace.shared.open(store.archiveURL.deletingLastPathComponent()) }; Button("Ripristina backup…") { let a = NSAlert(); a.messageText = "Ripristinare l’ultimo backup?"; a.informativeText = "Sostituisce le sessioni attualmente aperte."; a.addButton(withTitle: "Ripristina"); a.addButton(withTitle: "Annulla"); if a.runModal() == .alertFirstButtonReturn { store.restoreBackup() } }.disabled(store.hasWork) }
                Text("Salvataggio locale automatico con backup. Una chiusura interrotta conserva il testo parziale; le richieste non vengono reinviate da sole.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            }
            Section("Ricerca web") {
                Button("Cancella dati del browser web") { Task { await store.clearWebData() } }.disabled(!store.agentRuns.isEmpty)
                Text("web_search e visit_page usano un browser integrato nascosto (visibile dal pulsante globo nella barra della chat) con un profilo dedicato e persistente, separato da Safari. Il pulsante chiude il browser e cancella cookie, consenso e cache.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            }
            Section("SUC Studio 1.4.5_Final") {
                Text("SwiftUI · motore SUC C++/Metal · Apple Silicon\nDistribuzione locale, senza Developer ID o notarizzazione.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                Button("Guida setup e utilizzo") { store.openHelp() }
            }
        }.formStyle(.grouped).onDrop(of: [UTType.fileURL], isTargeted: nil) { providers in
            guard let provider = providers.first else { return false }
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url, ["suc1", "json"].contains(url.pathExtension.lowercased()) else { return }
                Task { @MainActor in store.selectFile(url, model: url.pathExtension.lowercased() == "suc1") }
            }
            return true
        }.onChange(of: store.archive.settings.context) { _, _ in store.scheduleSave() }.onChange(of: store.archive.settings.prefill) { _, _ in store.scheduleSave() }.onChange(of: store.archive.settings.expertBudget) { _, _ in store.scheduleSave() }.onChange(of: store.archive.settings.appearance) { _, _ in store.scheduleSave() }
    }
}
struct MenuBarView: View {
    @EnvironmentObject var store: StudioStore
    var show: () -> Void
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack { Text("SUC Studio").font(StudioTypography.ui(.headline)); Spacer(); Circle().fill(store.serverRunning ? .green : .secondary).frame(width: 8, height: 8) }
            Text(store.serverRunning ? (store.archive.settings.lan ? "API · Rete locale" : "API · Questo Mac") : "API spenta").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            Text(store.loadedModel.isEmpty ? "Nessun modello caricato" : store.loadedModel).font(StudioTypography.ui(.callout)).lineLimit(2)
            if store.hasWork { HStack { ProgressView().controlSize(.small); Text(store.activePhase).font(StudioTypography.ui(.caption)) } }
            Divider()
            LabeledContent("Prefill", value: rate(store.latestMetrics?.prefillRate))
            LabeledContent("Decode", value: rate(store.latestMetrics?.decodeRate))
            Text(store.latestIsCurrent ? "Misure disponibili a fine fase" : "Ultima richiesta").font(StudioTypography.ui(.caption2)).foregroundStyle(.secondary)
            if let memory = store.engineFootprint { LabeledContent("Memoria motore", value: String(format: "%.2f GB", memory)) }
            if !store.expertStats.isEmpty { Text(store.expertStats).font(StudioTypography.ui(.caption2)) }
            Divider()
            Button("Apri SUC Studio", action: show).keyboardShortcut("o")
            Button(store.serverRunning ? "Arresta API" : "Avvia API") { store.serverRunning ? store.stopServer() : store.startServer() }.disabled(!store.modelReady || store.serverBusy)
            Button("Esci") { NSApplication.shared.terminate(nil) }.keyboardShortcut("q")
        }.padding(18).frame(width: 300)
    }
}
