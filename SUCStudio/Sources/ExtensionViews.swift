import SwiftUI

struct ExtensionSettingsView: View {
    @EnvironmentObject var store: StudioStore
    @State private var skill: StudioSkill?
    @State private var server: MCPConfiguration?
    var body: some View {
        Section("Skill") {
            Text("Istruzioni riutilizzabili: quelle attive vengono aggiunte al system prompt delle chat. Puoi scriverle o importare un file SKILL.md.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            ForEach(store.archive.settings.skills ?? []) { item in
                HStack {
                    Toggle(item.name, isOn: Binding(get: { (store.archive.settings.skills ?? []).first { $0.id == item.id }?.enabled ?? false }, set: { var value = item; value.enabled = $0; store.saveSkill(value) }))
                    Button("Modifica") { skill = item }
                    Button(role: .destructive) { store.archive.settings.skills?.removeAll { $0.id == item.id }; store.scheduleSave() } label: { Image(systemName: "trash") }.help("Elimina skill")
                }
            }
            HStack { Button("Nuova skill") { skill = StudioSkill() }; Button("Importa SKILL.md…") { store.importSkill() } }
        }.sheet(item: $skill) { SkillEditor(skill: $0).environmentObject(store) }
        Section("Server MCP") {
            Text("Attiva i server da usare. Nella chat trovi gli strumenti disponibili: scegli lo strumento, eseguilo e allega il risultato al modello.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            ForEach(store.archive.settings.mcpServers ?? []) { item in
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Toggle(item.name, isOn: Binding(get: { (store.archive.settings.mcpServers ?? []).first { $0.id == item.id }?.enabled ?? false }, set: { store.toggleMCP(item, enabled: $0) }))
                        Button("Modifica") { server = item }.disabled(item.enabled)
                        Button("Riconnetti") { Task { await store.connectMCP(item) } }.disabled(!item.enabled)
                        Button(role: .destructive) { store.removeMCP(item.id) } label: { Image(systemName: "trash") }.help("Elimina server MCP")
                    }
                    Text(store.mcpStatus[item.id] ?? "Spento").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary).textSelection(.enabled)
                }
            }
            Button("Aggiungi MCP") { server = MCPConfiguration() }
        }.sheet(item: $server) { MCPServerEditor(configuration: $0).environmentObject(store) }
    }
}
struct SkillEditor: View {
    @EnvironmentObject var store: StudioStore
    @Environment(\.dismiss) private var dismiss
    @State var skill: StudioSkill
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Skill").font(StudioTypography.ui(.title2).bold())
            TextField("Nome", text: $skill.name).textFieldStyle(.roundedBorder)
            TextEditor(text: $skill.instructions).font(StudioTypography.ui(.body, design: .monospaced)).frame(minHeight: 260)
            Toggle("Attiva", isOn: $skill.enabled)
            Text("Le skill aggiungono istruzioni; script e file citati nel testo non vengono eseguiti o importati automaticamente.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            HStack { Spacer(); Button("Annulla") { dismiss() }; Button("Salva") { store.saveSkill(skill); dismiss() }.buttonStyle(.borderedProminent).disabled(skill.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || skill.instructions.utf8.count > 64 * 1024) }
        }.padding(24).frame(width: 620)
    }
}
struct MCPServerEditor: View {
    @EnvironmentObject var store: StudioStore
    @Environment(\.dismiss) private var dismiss
    @State var configuration: MCPConfiguration
    @State private var arguments = "[]"
    @State private var credentials = "{}"
    @State private var credentialsLoaded = false
    @State private var error: String?
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Server MCP").font(StudioTypography.ui(.title2).bold())
            TextField("Nome", text: $configuration.name).textFieldStyle(.roundedBorder)
            Picker("Connessione", selection: $configuration.transport) { Text("Locale · stdio").tag("stdio"); Text("Remoto · HTTP").tag("http") }.pickerStyle(.segmented)
            if configuration.transport == "stdio" {
                TextField("Eseguibile (percorso assoluto)", text: $configuration.executable).textFieldStyle(.roundedBorder)
                Text("Argomenti come array JSON, per esempio [\"/percorso/server.js\"]").font(StudioTypography.ui(.caption))
                TextEditor(text: $arguments).font(StudioTypography.ui(.caption, design: .monospaced)).frame(height: 60)
                Text("Variabili d’ambiente come oggetto JSON").font(StudioTypography.ui(.caption))
            } else {
                TextField("https://server.example/mcp", text: $configuration.endpoint).textFieldStyle(.roundedBorder)
                Text("Header HTTP come oggetto JSON, per esempio {\"Authorization\":\"Bearer …\"}").font(StudioTypography.ui(.caption))
                Text("Streamable HTTP con risposte JSON/SSE. OAuth interattivo e trasporto SSE legacy non inclusi.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            }
            TextEditor(text: $credentials).font(StudioTypography.ui(.caption, design: .monospaced)).frame(height: 90)
            Text("Variabili/header salvati nel Portachiavi, esclusi dagli export delle chat.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            Toggle("Attiva dopo il salvataggio", isOn: $configuration.enabled)
            if let error { Text(error).font(StudioTypography.ui(.caption)).foregroundStyle(.red) }
            HStack { Spacer(); Button("Annulla") { dismiss() }; Button("Salva") {
                do {
                    guard arguments.utf8.count < 64 * 1024, credentials.utf8.count < 64 * 1024 else { throw MCPConnection.failure("Configurazione troppo grande.") }
                    configuration.arguments = try JSONDecoder().decode([String].self, from: Data(arguments.utf8))
                    let secrets = try JSONDecoder().decode([String:String].self, from: Data(credentials.utf8))
                    try store.saveMCP(configuration, credentials: secrets); dismiss()
                } catch { self.error = error.localizedDescription }
            }.buttonStyle(.borderedProminent).disabled(!credentialsLoaded) }
        }.padding(24).frame(width: 640)
        .onAppear {
            arguments = String(decoding: (try? JSONEncoder().encode(configuration.arguments)) ?? Data("[]".utf8), as: UTF8.self)
            do { credentials = String(decoding: try JSONEncoder().encode(MCPSecrets.read(configuration.id)), as: UTF8.self); credentialsLoaded = true } catch { self.error = error.localizedDescription }
        }
    }
}
struct MCPToolSheet: View {
    @EnvironmentObject var store: StudioStore
    @Environment(\.dismiss) private var dismiss
    let sessionID: UUID
    @State private var selectedID = ""
    @State private var arguments = "{}"
    @State private var working = false
    @State private var error: String?
    private var selected: MCPTool? { store.mcpTools.first { $0.id == selectedID } }
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Strumenti MCP").font(StudioTypography.ui(.title2).bold())
            if store.mcpTools.isEmpty { Text("Attiva un server MCP nei Settings per vedere i suoi strumenti.") }
            else {
                Picker("Strumento", selection: $selectedID) { Text("Seleziona…").tag(""); ForEach(store.mcpTools) { tool in Text("\(tool.serverName) · \(tool.name)").tag(tool.id) } }.disabled(working)
                if let selected {
                    Text(selected.description).font(StudioTypography.ui(.caption)).lineLimit(5)
                    DisclosureGroup("Schema degli argomenti") { ScrollView { Text(selected.schema).font(StudioTypography.ui(.caption, design: .monospaced)).textSelection(.enabled) }.frame(maxHeight: 130) }
                }
                Text("Argomenti JSON").font(StudioTypography.ui(.caption))
                TextEditor(text: $arguments).font(StudioTypography.ui(.body, design: .monospaced)).frame(height: 150).disabled(working)
                Text("Il risultato viene aggiunto agli allegati di questa chat. Premi poi Invio per farlo leggere al modello.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            }
            if let error { ScrollView { Text(error).foregroundStyle(.red).textSelection(.enabled) }.frame(maxHeight: 100) }
            HStack { if working { ProgressView().controlSize(.small); Text("Esecuzione…") }; Spacer(); Button("Chiudi") { dismiss() }.disabled(working); Button("Esegui e allega risultato") {
                guard let tool = selected else { return }; working = true; error = nil
                Task { do { try await store.runMCP(tool, argumentsJSON: arguments, sessionID: sessionID); dismiss() } catch { self.error = error.localizedDescription }; working = false }
            }.buttonStyle(.borderedProminent).disabled(selected == nil || working) }
        }.padding(24).frame(width: 660).interactiveDismissDisabled(working)
    }
}
