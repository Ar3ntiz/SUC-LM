import SwiftUI

struct ExtensionSettingsView: View {
    @EnvironmentObject var store: StudioStore
    @State private var skill: StudioSkill?
    @State private var server: MCPConfiguration?
    var body: some View {
        Section(L("Skill")) {
            Text(L("Titolo e descrizione delle skill attive sono visibili al modello; istruzioni e risorse vengono lette quando servono. Richiamo esplicito: @nome-skill oppure $nome-skill.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            ForEach(store.archive.settings.skills ?? []) { item in
                HStack {
                    Toggle(item.name, isOn: Binding(get: { (store.archive.settings.skills ?? []).first { $0.id == item.id }?.enabled ?? false }, set: { var value = item; value.enabled = $0; store.saveSkill(value) }))
                    Button(L("Modifica")) { skill = item }
                    Button(role: .destructive) { store.archive.settings.skills?.removeAll { $0.id == item.id }; store.scheduleSave() } label: { Image(systemName: "trash") }.help(L("Elimina skill"))
                }
            }
            HStack { Button(L("Nuova skill")) { skill = StudioSkill() }; Button(L("Importa skill / cartella…")) { store.importSkill() };Button(L("Scopri skill locali")){store.discoverSkills()} }
        }.sheet(item: $skill) { SkillEditor(skill: $0).environmentObject(store) }
        Section(L("Server MCP")) {
            Text(L("Attiva i server da usare. Nella chat trovi gli strumenti disponibili: scegli lo strumento, eseguilo e allega il risultato al modello.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            ForEach(store.archive.settings.mcpServers ?? []) { item in
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Toggle(item.name, isOn: Binding(get: { (store.archive.settings.mcpServers ?? []).first { $0.id == item.id }?.enabled ?? false }, set: { store.toggleMCP(item, enabled: $0) }))
                        Button(L("Modifica")) { server = item }.disabled(item.enabled)
                        Button(L("Riconnetti")) { Task { await store.connectMCP(item) } }.disabled(!item.enabled)
                        Button(role: .destructive) { store.removeMCP(item.id) } label: { Image(systemName: "trash") }.help(L("Elimina server MCP"))
                    }
                    Text(LT(store.mcpStatus[item.id] ?? L("Spento"))).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary).textSelection(.enabled)
                }
            }
            Button(L("Aggiungi MCP")) { server = MCPConfiguration() }
        }.sheet(item: $server) { MCPServerEditor(configuration: $0).environmentObject(store) }
    }
}
struct SkillEditor: View {
    @EnvironmentObject var store: StudioStore
    @Environment(\.dismiss) private var dismiss
    @State var skill: StudioSkill
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(L("Skill")).font(StudioTypography.ui(.title2).bold())
            Menu(L("Suggerimenti / esempi…")) {
                Button(L("Revisione codice")){applySample("review-code",L("Usa quando viene chiesta una revisione del codice o dei bug."),L("Leggi i file pertinenti. Cerca bug concreti e regressioni; riporta file, righe e motivazione. Non modificare file senza richiesta."))}
                Button(L("Ricerca con fonti")){applySample("research",L("Usa per ricerche web con confronto tra fonti e dati aggiornati."),L("Cerca fonti primarie, verifica date e divergenze. Cita link diretti; distingui fatti, stime e ipotesi. Usa gli strumenti web solo se abilitati."))}
                Button(L("Documentazione")){applySample("document-project",L("Usa quando occorre spiegare un progetto o aggiornare la sua guida."),L("Consulta README e sorgenti pertinenti. Scrivi passaggi riproducibili, esempi minimi e requisiti. Non dichiarare verifiche non eseguite."))}
            }
            TextField(L("Nome · usato con @ o $"), text: $skill.name).textFieldStyle(.roundedBorder)
            Text(L("Descrizione · quando il modello deve usare questa skill")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            TextEditor(text:Binding(get:{skill.description ?? ""},set:{skill.description=$0})).frame(minHeight:65,maxHeight:100)
            Text(L("Istruzioni della skill")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            TextEditor(text: $skill.instructions).font(StudioTypography.ui(.body, design: .monospaced)).frame(minHeight: 260)
            Toggle(L("Consenti selezione implicita da parte del modello"),isOn:Binding(get:{skill.allowImplicitInvocation ?? true},set:{skill.allowImplicitInvocation=$0}))
            if let path=skill.resourceRoot {DisclosureGroup(L("Risorse del pacchetto")){Text(path).font(.caption).textSelection(.enabled);ScrollView{Text(((try? ExtensionFiles.files(URL(fileURLWithPath:path))) ?? []).joined(separator:"\n")).font(.caption.monospaced()).textSelection(.enabled)}.frame(maxHeight:80)}}
            Toggle(L("Attiva"), isOn: $skill.enabled)
            Text(L("Il modello legge inizialmente titolo e descrizione delle skill attive. Carica il testo completo con load_skill quando è pertinente. Puoi includere scripts/, references/ e assets/ nella cartella importata. I file vengono letti su richiesta; gli script richiedono i normali permessi dei comandi.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            HStack { Spacer(); Button(L("Annulla")) { dismiss() }; Button(L("Salva")) { store.saveSkill(skill); dismiss() }.buttonStyle(.borderedProminent).disabled(skill.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || skill.instructions.utf8.count > 64 * 1024 || (skill.description ?? "").utf8.count > 4 * 1024) }
        }.padding(24).frame(width: 620)
    }
    private func applySample(_ name: String,_ description: String,_ instructions: String) {skill.name=name;skill.description=description;skill.instructions=instructions;skill.enabled=false}
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
            Text(L("Server MCP")).font(StudioTypography.ui(.title2).bold())
            TextField(L("Nome"), text: $configuration.name).textFieldStyle(.roundedBorder)
            Picker(L("Connessione"), selection: $configuration.transport) { Text(L("Locale · stdio")).tag("stdio"); Text(L("Remoto · HTTP")).tag("http") }.pickerStyle(.segmented)
            if configuration.transport == "stdio" {
                TextField(L("Eseguibile (percorso assoluto)"), text: $configuration.executable).textFieldStyle(.roundedBorder)
                Text(L("Argomenti come array JSON, per esempio [\"/percorso/server.js\"]")).font(StudioTypography.ui(.caption))
                TextEditor(text: $arguments).font(StudioTypography.ui(.caption, design: .monospaced)).frame(height: 60)
                Text(L("Variabili d’ambiente come oggetto JSON")).font(StudioTypography.ui(.caption))
            } else {
                TextField("https://server.example/mcp", text: $configuration.endpoint).textFieldStyle(.roundedBorder)
                Text(L("Header HTTP come oggetto JSON, per esempio {\"Authorization\":\"Bearer …\"}")).font(StudioTypography.ui(.caption))
                Text(L("Streamable HTTP con risposte JSON/SSE. OAuth interattivo e trasporto SSE legacy non inclusi.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            }
            TextEditor(text: $credentials).font(StudioTypography.ui(.caption, design: .monospaced)).frame(height: 90)
            Text(L("Variabili/header salvati nel Portachiavi, esclusi dagli export delle chat.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            if configuration.transport == "stdio" {
                Text(L("I provider come Ghidra richiedono variabili proprie, per esempio GHIDRA_INSTALL_DIR e JAVA_HOME. L’automazione di altre app può richiedere il consenso in Privacy e sicurezza → Automazione; la connessione MCP da sola non lo concede.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary).fixedSize(horizontal:false,vertical:true)
            }
            Toggle(L("Attiva dopo il salvataggio"), isOn: $configuration.enabled)
            if let error { Text(error).font(StudioTypography.ui(.caption)).foregroundStyle(.red) }
            HStack { Spacer(); Button(L("Annulla")) { dismiss() }; Button(L("Salva")) {
                do {
                    guard arguments.utf8.count < 64 * 1024, credentials.utf8.count < 64 * 1024 else { throw MCPConnection.failure(L("Configurazione troppo grande.")) }
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
            Text(L("Strumenti MCP")).font(StudioTypography.ui(.title2).bold())
            if store.mcpTools.isEmpty { Text(L("Attiva un server MCP nei Settings per vedere i suoi strumenti.")) }
            else {
                Picker(L("Strumento"), selection: $selectedID) { Text(L("Seleziona…")).tag(""); ForEach(store.mcpTools) { tool in Text("\(tool.serverName) · \(tool.name)").tag(tool.id) } }.disabled(working)
                if let selected {
                    Text(selected.description).font(StudioTypography.ui(.caption)).lineLimit(5)
                    DisclosureGroup(L("Schema degli argomenti")) { ScrollView { Text(selected.schema).font(StudioTypography.ui(.caption, design: .monospaced)).textSelection(.enabled) }.frame(maxHeight: 130) }
                }
                Text(L("Argomenti JSON")).font(StudioTypography.ui(.caption))
                TextEditor(text: $arguments).font(StudioTypography.ui(.body, design: .monospaced)).frame(height: 150).disabled(working)
                Text(L("Il risultato viene aggiunto agli allegati di questa chat. Premi poi Invio per farlo leggere al modello.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            }
            if let error { ScrollView { Text(error).foregroundStyle(.red).textSelection(.enabled) }.frame(maxHeight: 100) }
            HStack { if working { ProgressView().controlSize(.small); Text(L("Esecuzione…")) }; Spacer(); Button(L("Chiudi")) { dismiss() }.disabled(working); Button(L("Esegui e allega risultato")) {
                guard let tool = selected else { return }; working = true; error = nil
                Task { do { try await store.runMCP(tool, argumentsJSON: arguments, sessionID: sessionID); dismiss() } catch { self.error = error.localizedDescription }; working = false }
            }.buttonStyle(.borderedProminent).disabled(selected == nil || working) }
        }.padding(24).frame(width: 660).interactiveDismissDisabled(working)
    }
}
