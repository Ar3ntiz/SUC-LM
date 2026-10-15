import SwiftUI
struct PluginManagerView: View {
    @EnvironmentObject var store: StudioStore
    @Environment(\.dismiss) private var dismiss
    @State private var address=""
    @State private var entries:[MarketplaceEntry]=[]
    @State private var preview:InstalledPlugin?
    @State private var candidate:PluginManifest?
    @State private var candidateSource=""
    @State private var busy=false
    @State private var status=""
    var body: some View {
        VStack(alignment:.leading,spacing:14){
            HStack{Text("Plugin").font(StudioTypography.ui(.title2).bold());Spacer();Button("Chiudi"){dismiss()}}
            HStack{Button("Installa da locale…"){store.importPlugin()};Text("Formato SUC Studio v1 · installati disattivati").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)}
            List{ForEach(store.archive.settings.plugins ?? []){plugin in
                HStack{Toggle(plugin.manifest.name+" · "+plugin.manifest.version,isOn:Binding(get:{store.archive.settings.plugins?.first{$0.id==plugin.id}?.enabled ?? false},set:{store.setPlugin(plugin,enabled:$0)}));Button("Dettagli"){preview=plugin};Button("Rimuovi",role:.destructive){store.removePlugin(plugin)}}
            }}.frame(minHeight:150)
            Text("Marketplace HTTPS").font(StudioTypography.ui(.headline))
            HStack{TextField("https://example.com/marketplace.json",text:$address).textFieldStyle(.roundedBorder);Button("Carica indice"){
                busy=true;Task{defer{busy=false};do{let data=try await PluginValidation.download(address);let index=try JSONDecoder().decode(MarketplaceIndex.self,from:data);guard index.entries.count<=200 else{throw MCPConnection.failure("Indice oltre 200 plugin.")};entries=index.entries;store.archive.settings.marketplaceURL=address;store.scheduleSave();status="\(entries.count) plugin disponibili"}catch{status=error.localizedDescription}}
            }.disabled(busy)}
            ScrollView{VStack(alignment:.leading){ForEach(entries){entry in HStack{Text(entry.name);Spacer();Button("Anteprima / installa"){
                busy=true;Task{defer{busy=false};do{candidate=try PluginValidation.verified(await PluginValidation.download(entry.manifestURL),sha256:entry.sha256);candidateSource=entry.manifestURL}catch{status=error.localizedDescription}}
            }.disabled(busy)}}}}.frame(height:120)
            if busy{ProgressView().controlSize(.small)}
            Text(status).font(StudioTypography.ui(.caption)).textSelection(.enabled)
            if let candidate{GroupBox("\(candidate.name) · \(candidate.version)"){VStack(alignment:.leading){Text(candidateSource).font(StudioTypography.ui(.caption));Text("\(candidate.skills?.count ?? 0) skill · \(candidate.mcpServers?.count ?? 0) MCP · \(candidate.hooks?.count ?? 0) hook");Button("Installa disattivato"){do{try store.installPlugin(candidate,source:candidateSource);self.candidate=nil;status="Installato: consulta Dettagli prima di attivarlo."}catch{status=error.localizedDescription}}}}}
        }.padding(24).frame(width:760,height:650).onAppear{address=store.archive.settings.marketplaceURL ?? ""}
        .sheet(item:$preview){plugin in PluginDetailsView(plugin:plugin)}
    }
}
struct PluginDetailsView:View{
    @Environment(\.dismiss) private var dismiss
    let plugin:InstalledPlugin
    var body:some View{VStack(alignment:.leading){HStack{Text(plugin.manifest.name).font(StudioTypography.ui(.title2));Spacer();Button("Chiudi"){dismiss()}};Text(plugin.source).font(StudioTypography.ui(.caption));ScrollView{Text(String(decoding:(try? JSONEncoder.pretty.encode(plugin.manifest)) ?? Data(),as:UTF8.self)).font(StudioTypography.ui(.body,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading)}}.padding(24).frame(width:700,height:550)}
}
extension JSONEncoder {static var pretty:JSONEncoder{let encoder=JSONEncoder();encoder.outputFormatting=[.prettyPrinted,.sortedKeys];return encoder}}
struct RoutinesView:View{
    @EnvironmentObject var store:StudioStore
    @Environment(\.dismiss) private var dismiss
    @State private var editing:StudioRoutine?
    var body:some View{
        VStack(alignment:.leading,spacing:16){HStack{Text("Routine").font(StudioTypography.ui(.title2).bold());Spacer();Button("Chiudi"){dismiss()}}
            Text("Si eseguono con l’app aperta, il backend selezionato e un modello disponibile. Dopo sleep: una sola esecuzione, senza recuperare tutte le scadenze. Ereditano directory e approvazioni della sessione.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            List{ForEach(store.archive.settings.routines ?? []){routine in VStack(alignment:.leading){HStack{Toggle(routine.name,isOn:Binding(get:{store.archive.settings.routines?.first{$0.id==routine.id}?.enabled ?? false},set:{enabled in var copy=routine;copy.enabled=enabled;do{try store.saveRoutine(copy)}catch{store.error=error.localizedDescription}}));Button("Modifica"){editing=routine}.disabled(store.activeRoutines[routine.id] != nil);Button(role:.destructive){store.archive.settings.routines?.removeAll{$0.id==routine.id};store.scheduleSave()}label:{Image(systemName:"trash")}.disabled(store.activeRoutines[routine.id] != nil)}
                Text("\(routine.status) · prossima: \(routine.nextRun.formatted())").font(StudioTypography.ui(.caption))
                DisclosureGroup("Storico (\(routine.history.count))"){ForEach(routine.history.reversed()){run in Text("\(run.date.formatted()) · \(run.state)\n\(run.detail)").font(StudioTypography.ui(.caption)).textSelection(.enabled)}}
            }}}
            Button("Nuova routine"){var value=StudioRoutine();value.sessionID=store.archive.selected;value.backend=store.usesRemote ? "remote":"local";editing=value}
        }.padding(24).frame(width:760,height:560).sheet(item:$editing){RoutineEditor(value:$0).environmentObject(store)}
    }
}
struct RoutineEditor:View{
    @EnvironmentObject var store:StudioStore
    @Environment(\.dismiss) private var dismiss
    @State var value:StudioRoutine
    @State private var error=""
    var body:some View{VStack(alignment:.leading,spacing:14){TextField("Nome",text:$value.name);Picker("Sessione",selection:$value.sessionID){ForEach(store.archive.sessions.filter{$0.parentSessionID==nil}){Text($0.title).tag(Optional($0.id))}};Picker("Backend richiesto",selection:$value.backend){Text("Server locale").tag("local");Text("Server remoto").tag("remote")};TextEditor(text:$value.prompt).frame(height:150);DatePicker("Prossima esecuzione",selection:$value.nextRun);TextField("Intervallo in minuti",value:$value.intervalMinutes,format:.number.grouping(.never));Toggle("Attiva",isOn:$value.enabled);Text(error).foregroundStyle(.red);HStack{Spacer();Button("Annulla"){dismiss()};Button("Salva"){do{try store.saveRoutine(value);dismiss()}catch{self.error=error.localizedDescription}}.buttonStyle(.borderedProminent)}}.padding(24).frame(width:620)}
}
