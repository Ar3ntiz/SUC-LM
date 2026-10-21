import SwiftUI
struct PluginManagerView: View {
    @EnvironmentObject var store: StudioStore
    @Environment(\.dismiss) private var dismiss
    @State private var builder=false
    @State private var packages=false
    @State private var address=""
    @State private var entries:[MarketplaceEntry]=[]
    @State private var preview:InstalledPlugin?
    @State private var candidate:PluginManifest?
    @State private var candidateSource=""
    @State private var busy=false
    @State private var status=""
    var body: some View {
        VStack(alignment:.leading,spacing:14){
            HStack{Text(L("Plugin")).font(StudioTypography.ui(.title2).bold());Spacer();Button(L("Chiudi")){dismiss()}}
            HStack{Button(L("Installa da locale…")){store.importPlugin()};Button(L("Crea con esempi…")){builder=true};Button(L("Marketplace pacchetti…")){packages=true};Text("Portable / Codex / SUC v1").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)}
            List{ForEach(store.archive.settings.plugins ?? []){plugin in
                HStack{Toggle(plugin.manifest.name+" · "+plugin.manifest.version,isOn:Binding(get:{store.archive.settings.plugins?.first{$0.id==plugin.id}?.enabled ?? false},set:{store.setPlugin(plugin,enabled:$0)}));Button(L("Dettagli")){preview=plugin};Button(L("Rimuovi"),role:.destructive){store.removePlugin(plugin)}}
            }}.frame(minHeight:150)
            Text("Marketplace HTTPS").font(StudioTypography.ui(.headline))
            HStack{TextField("https://example.com/marketplace.json",text:$address).textFieldStyle(.roundedBorder);Button(L("Carica indice")){
                busy=true;Task{defer{busy=false};do{let data=try await PluginValidation.download(address);let index=try JSONDecoder().decode(MarketplaceIndex.self,from:data);guard index.entries.count<=200 else{throw MCPConnection.failure(L("Indice oltre 200 plugin."))};entries=index.entries;store.archive.settings.marketplaceURL=address;store.scheduleSave();status="\(entries.count) plugin disponibili"}catch{status=error.localizedDescription}}
            }.disabled(busy)}
            ScrollView{VStack(alignment:.leading){ForEach(entries){entry in HStack{Text(entry.name);Spacer();Button(L("Anteprima / installa")){
                busy=true;Task{defer{busy=false};do{candidate=try PluginValidation.verified(await PluginValidation.download(entry.manifestURL),sha256:entry.sha256);candidateSource=entry.manifestURL}catch{status=error.localizedDescription}}
            }.disabled(busy)}}}}.frame(height:120)
            if busy{ProgressView().controlSize(.small)}
            Text(LT(status)).font(StudioTypography.ui(.caption)).textSelection(.enabled)
            if let candidate{GroupBox("\(candidate.name) · \(candidate.version)"){VStack(alignment:.leading){Text(candidateSource).font(StudioTypography.ui(.caption));Text("\(candidate.skills?.count ?? 0) skill · \(candidate.mcpServers?.count ?? 0) MCP · \(candidate.hooks?.count ?? 0) hook");Button(L("Installa disattivato")){do{try store.installPlugin(candidate,source:candidateSource);self.candidate=nil;status=L("Installato: consulta Dettagli prima di attivarlo.")}catch{status=error.localizedDescription}}}}}
        }.padding(24).frame(width:760,height:650).onAppear{address=store.archive.settings.marketplaceURL ?? ""}
        .sheet(item:$preview){plugin in PluginDetailsView(plugin:plugin).environmentObject(store)}
        .sheet(isPresented:$builder){PluginBuilderView()}
        .sheet(isPresented:$packages){PackageMarketplaceView().environmentObject(store)}
    }
}
struct PluginDetailsView:View {
    @EnvironmentObject var store: StudioStore
    @Environment(\.dismiss) private var dismiss
    let plugin:InstalledPlugin
    var body:some View {
        VStack(alignment:.leading,spacing:14) {
            HStack{Text(plugin.manifest.name).font(.title2);Spacer();Button(L("Chiudi")){dismiss()}}
            Text(plugin.manifest.description ?? plugin.source).font(.caption).textSelection(.enabled)
            Text(L("Formato: \(plugin.packageFormat ?? "SUC v1") · \(plugin.manifest.version)")).font(.caption)
            if let path=plugin.bundleRoot {Text(L("Codice: ")+path).font(.caption).textSelection(.enabled)}
            if let path=plugin.dataRoot {Text(L("Dati: ")+path).font(.caption).textSelection(.enabled)}
            List {
                Section(L("Skill · abilita il plugin per gestire i componenti")) {ForEach(plugin.skills){skill in
                    HStack{Text(skill.name);Spacer();Toggle(L("Attiva"),isOn:Binding(get:{store.archive.settings.skills?.first{$0.id==skill.id}?.enabled ?? false},set:{value in if var copy=store.archive.settings.skills?.first(where:{$0.id==skill.id}){copy.enabled=value;store.saveSkill(copy)}})).disabled(!plugin.enabled)}
                }}
                Section("MCP") {ForEach(plugin.servers){server in
                    HStack{Text(server.name);Spacer();Toggle(L("Attivo"),isOn:Binding(get:{store.archive.settings.mcpServers?.first{$0.id==server.id}?.enabled ?? false},set:{store.toggleMCP(server,enabled:$0)})).disabled(!plugin.enabled)}
                }}
                Section(L("Hook · revisiona e autorizza nella sezione Hook")) {ForEach(plugin.hooks){hook in Text(hook.name+" · "+HookDefinition.event(hook.event)).font(.caption)}}
            }
            if let onboarding=plugin.manifest.onboardingSkill,let skill=plugin.skills.first(where:{$0.sourcePath?.hasSuffix(onboarding.replacingOccurrences(of:"./",with:""))==true}) {
                Button(L("Avvia configurazione guidata")) {store.tab="chat";store.mutateSelected{$0.draft="@"+skill.name+" "+(skill.defaultPrompt ?? L("Configura il plugin."))};dismiss()}.disabled(!plugin.enabled || store.selected==nil)
            }
            DisclosureGroup("Manifest JSON") {ScrollView{Text(String(decoding:(try? JSONEncoder.pretty.encode(plugin.manifest)) ?? Data(),as:UTF8.self)).font(.caption.monospaced()).textSelection(.enabled)}.frame(maxHeight:120)}
        }.padding(24).frame(width:740,height:600)
    }
}
extension JSONEncoder {static var pretty:JSONEncoder{let encoder=JSONEncoder();encoder.outputFormatting=[.prettyPrinted,.sortedKeys];return encoder}}
struct RoutinesView:View{
    @EnvironmentObject var store:StudioStore
    @Environment(\.dismiss) private var dismiss
    @State private var editing:StudioRoutine?
    var body:some View{
        VStack(alignment:.leading,spacing:16){HStack{Text(L("Routine")).font(StudioTypography.ui(.title2).bold());Spacer();Button(L("Chiudi")){dismiss()}}
            Text(L("Si eseguono con l’app aperta, il backend selezionato e un modello disponibile. Dopo sleep: una sola esecuzione, senza recuperare tutte le scadenze. Ereditano directory e approvazioni della sessione.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            List{ForEach(store.archive.settings.routines ?? []){routine in VStack(alignment:.leading){HStack{Toggle(routine.name,isOn:Binding(get:{store.archive.settings.routines?.first{$0.id==routine.id}?.enabled ?? false},set:{enabled in var copy=routine;copy.enabled=enabled;do{try store.saveRoutine(copy)}catch{store.error=error.localizedDescription}}));Button(L("Modifica")){editing=routine}.disabled(store.activeRoutines[routine.id] != nil);Button(role:.destructive){store.archive.settings.routines?.removeAll{$0.id==routine.id};store.scheduleSave()}label:{Image(systemName:"trash")}.disabled(store.activeRoutines[routine.id] != nil)}
                Text("\(LT(routine.status)) · \(routine.isRepeating ? L("prossima"):L("esecuzione unica")): \(routine.nextRun.formatted())").font(StudioTypography.ui(.caption))
                DisclosureGroup(L("Storico (\(routine.history.count))")){ForEach(routine.history.reversed()){run in Text("\(run.date.formatted()) · \(run.state)\n\(run.detail)").font(StudioTypography.ui(.caption)).textSelection(.enabled)}}
            }}}
            Button(L("Nuova routine")){var value=StudioRoutine();value.sessionID=store.archive.selected;value.backend=store.usesRemote ? "remote":"local";editing=value}
        }.padding(24).frame(width:760,height:560).sheet(item:$editing){RoutineEditor(value:$0).environmentObject(store)}
    }
}
struct RoutineEditor:View {
    @EnvironmentObject var store:StudioStore
    @Environment(\.dismiss) private var dismiss
    @State var value:StudioRoutine
    @State private var error=""
    @State private var intervalText="1"
    var body:some View {
        VStack(alignment:.leading,spacing:14) {
            TextField(L("Nome"),text:$value.name)
            Picker(L("Sessione"),selection:$value.sessionID) {ForEach(store.archive.sessions.filter{$0.parentSessionID==nil}){Text($0.title).tag(Optional($0.id))}}
            Picker(L("Backend richiesto"),selection:$value.backend) {
                if StudioPlatform.localInferenceAvailable {Text(L("Server locale")).tag("local")}
                Text(L("Server remoto")).tag("remote")
            }
            TextEditor(text:$value.prompt).frame(height:150)
            Text(L("Le routine riservano almeno 4000 token di output, nei limiti del contesto, per completare thinking, strumenti e risposta.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            DatePicker(L("Prima esecuzione"),selection:$value.nextRun)
            Toggle(L("Ripeti nel tempo"),isOn:Binding(get:{value.isRepeating},set:{value.repeatEnabled=$0}))
            if value.isRepeating {
                HStack {
                    Text(L("Ogni"))
                    TextField(L("Numero"),text:$intervalText).textFieldStyle(.roundedBorder).frame(width:95)
                        .onChange(of:intervalText){_,text in intervalText=String(text.filter{$0.isASCII && $0.isNumber}.prefix(6))}
                    Picker(L("Unità"),selection:Binding(get:{value.recurrenceUnit},set:{value.repeatUnit=$0})) {
                        Text(L("minuti")).tag("minutes");Text(L("ore")).tag("hours");Text(L("giorni")).tag("days");Text(L("mesi")).tag("months")
                    }.labelsHidden().frame(width:130)
                }
            }
            Toggle(L("Attiva"),isOn:$value.enabled)
            Text(L("Le routine richiedono l’app aperta e un server disponibile. Le autorizzazioni sono quelle della sessione: con approvazione automatica possono usare anche il browser web; con approvazione manuale attendono conferma nella chat.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            Text(LT(error)).foregroundStyle(.red)
            HStack {
                Spacer();Button(L("Annulla")){dismiss()}
                Button(L("Salva")) {
                    do {
                        if value.isRepeating {
                            guard let count=Int(intervalText),(1...525600).contains(count) else{throw MCPConnection.failure(L("Inserisci un numero intero da 1 a 525600."))}
                            value.repeatCount=count
                            if value.repeatUnit==nil {value.repeatUnit="minutes"}
                        }
                        try store.saveRoutine(value);dismiss()
                    } catch {self.error=error.localizedDescription}
                }.buttonStyle(.borderedProminent)
            }
        }.padding(24).frame(width:620).onAppear{intervalText=String(value.recurrenceCount)}
    }
}
