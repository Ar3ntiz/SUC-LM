import SwiftUI
import AppKit

struct PluginBuilderView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var draft=PluginBuilderDraft()
    @State private var editor="skill"
    @State private var status=""
    init(draft: PluginBuilderDraft = PluginBuilderDraft()) {_draft=State(initialValue:draft)}
    private var code: Binding<String> {
        switch editor {case "hook":return $draft.hook;case "mcp":return $draft.mcp;default:return $draft.skill}
    }
    private var path: String {editor=="hook" ? "hooks/hooks.json":editor=="mcp" ? "mcp.json":"skills/<nome-skill>/SKILL.md"}
    private func label(_ component: String) -> String {component=="skill" ? "Skill":component=="hook" ? "Hook":"MCP"}
    var body: some View {
        VStack(alignment:.leading,spacing:12) {
            HStack {Text("Crea un plugin").font(StudioTypography.ui(.title2).bold());Spacer();Button("Chiudi"){dismiss()}}
            Text("Seleziona uno o più componenti: skill, hook e MCP possono convivere nello stesso plugin. Ogni editor conserva il proprio codice; l’esportazione non esegue comandi e non attiva componenti.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            TextField("Identificatore · kebab-case",text:$draft.name).textFieldStyle(.roundedBorder)
            TextField("Descrizione",text:$draft.description).textFieldStyle(.roundedBorder)
            GroupBox("Componenti da includere") {
                HStack(spacing:24) {Toggle("Skill",isOn:$draft.includeSkill);Toggle("Hook",isOn:$draft.includeHook);Toggle("MCP",isOn:$draft.includeMCP);Spacer()}.padding(4)
            }
            if !draft.selectedComponents.isEmpty {
                Picker("Editor",selection:$editor){ForEach(draft.selectedComponents,id:\.self){Text(label($0)).tag($0)}}.pickerStyle(.segmented)
                HStack {Text(path).font(StudioTypography.ui(.caption,design:.monospaced));Spacer();Button("Carica esempio "+label(editor)){draft.reset(editor);status="Esempio caricato solo per "+label(editor)+". Gli altri componenti sono conservati."}}
                TextEditor(text:code).font(StudioTypography.ui(.body,design:.monospaced)).frame(minHeight:180)
                Text(editor=="skill" ? "SKILL.md: nome, descrizione di quando usarla e istruzioni. Il nome determina la cartella della skill.":editor=="hook" ? "hooks.json: eventi e comandi. Gli hook richiederanno revisione e autorizzazione dopo l’importazione.":"mcp.json: server stdio o Streamable HTTP. Sostituisci l’URL di esempio con il tuo server; configura i segreti nel Portachiavi.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            } else {ContentUnavailableView("Seleziona i componenti",systemImage:"shippingbox",description:Text("Puoi includere Skill, Hook e MCP insieme oppure scegliere solo quelli necessari.")).frame(minHeight:180)}
            DisclosureGroup("Struttura e suggerimenti") {Text("Un unico plugin.json descrive il pacchetto. skills/ contiene cartelle con SKILL.md, scripts/, references/ e assets/ opzionali; hooks/hooks.json e mcp.json aggiungono gli altri componenti selezionati. Usa ${PLUGIN_ROOT} per il codice e ${PLUGIN_DATA} per i dati scrivibili. Gli editor sono indipendenti: cambiare scheda o disattivare un componente conserva il testo.").font(StudioTypography.ui(.caption))}
            Text(status).font(StudioTypography.ui(.caption)).textSelection(.enabled).lineLimit(3)
            HStack {Text(draft.selectedComponents.count==1 ? "1 componente selezionato":"\(draft.selectedComponents.count) componenti selezionati").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary);Spacer();Button("Esporta plugin…"){export()}.buttonStyle(.borderedProminent).disabled(draft.selectedComponents.isEmpty)}
        }.padding(24).frame(width:780,height:700)
        .onChange(of:draft.selectedComponents){_,selected in if !selected.contains(editor){editor=selected.first ?? "skill"}}
    }
    private func export() {
        do {
            _ = try draft.documents()
            let panel=NSOpenPanel();panel.canChooseFiles=false;panel.canChooseDirectories=true;panel.message="Scegli la cartella in cui creare \(draft.name)"
            guard panel.runModal() == .OK,let parent=panel.url else{return}
            let root=try draft.export(to:parent)
            status="Esportato: \(root.path). Importalo dal gestore Plugin."
        } catch {status=error.localizedDescription}
    }
}
struct PackageMarketplaceView: View {
    @EnvironmentObject var store: StudioStore
    @Environment(\.dismiss) private var dismiss
    @State private var entries=[PackageMarketplaceEntry]()
    @State private var address=""
    @State private var temporaryRoot: URL?
    @State private var status=""
    @State private var busy=false
    var body: some View {
        VStack(alignment:.leading,spacing:16) {
            HStack {Text("Marketplace di pacchetti").font(.title2.bold());Spacer();Button("Chiudi"){dismiss()}.disabled(busy)}
            Text("Apri un repository locale con .agents/plugins/marketplace.json o clona un repository HTTPS. Le sorgenti local, url e git-subdir sono supportate. Installazione sempre disattivata: revisiona poi i componenti.").font(.caption).foregroundStyle(.secondary)
            HStack {Button("Apri marketplace locale…"){choose()}.disabled(busy);TextField("https://host/repository.git",text:$address);Button("Carica repository"){loadRemote()}.disabled(busy || address.isEmpty)}
            List(entries){entry in HStack{Text(entry.name);Spacer();Button("Installa"){busy=true;Task{defer{busy=false};do{try await store.installMarketplacePackage(entry);status="Installato disattivato: "+entry.name}catch{status=error.localizedDescription}}}.disabled(entry.unavailable || busy)}}
            if busy {ProgressView().controlSize(.small)}
            Text(status).font(.caption).textSelection(.enabled)
        }.padding(24).frame(width:760,height:520).interactiveDismissDisabled(busy)
        .onDisappear{if let temporaryRoot{try? FileManager.default.removeItem(at:temporaryRoot)}}
    }
    private func read(_ url: URL) throws {
        var root=url.deletingLastPathComponent()
        if root.lastPathComponent=="plugins",root.deletingLastPathComponent().lastPathComponent==".agents" {root=root.deletingLastPathComponent().deletingLastPathComponent()}
        else if root.lastPathComponent==".claude-plugin" {root=root.deletingLastPathComponent()}
        entries=try PackageMarketplace.decode(Data(ExtensionFiles.read(url,limit:1024*1024).utf8),root:root)
        status="\(entries.count) pacchetti disponibili"
    }
    private func choose() {let panel=NSOpenPanel();panel.message="Scegli marketplace.json";guard panel.runModal() == .OK,let url=panel.url else{return};do{try read(url)}catch{status=error.localizedDescription}}
    private func loadRemote() {
        busy=true;Task {defer{busy=false};do{
            let root=try await PackageMarketplace.checkout(address)
            if let old=temporaryRoot {try? FileManager.default.removeItem(at:old)};temporaryRoot=root
            let candidates=[".agents/plugins/marketplace.json",".claude-plugin/marketplace.json","marketplace.json"]
            guard let file=candidates.map({root.appendingPathComponent($0)}).first(where:{FileManager.default.fileExists(atPath:$0.path)}) else{throw ExtensionFiles.failure("marketplace.json non trovato nel repository.")}
            try read(file)
        }catch{status=error.localizedDescription}}
    }
}
