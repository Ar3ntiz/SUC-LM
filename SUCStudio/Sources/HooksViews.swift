import SwiftUI
struct HooksSettingsView: View {
    @EnvironmentObject var store: StudioStore
    @Environment(\.dismiss) private var dismiss
    @State private var editing: StudioHook?
    var body: some View {
        VStack(alignment:.leading,spacing:16) {
            HStack {Text(L("Hook del ciclo di lavoro")).font(StudioTypography.ui(.title2).bold());Spacer();Button(L("Chiudi")){dismiss()}.keyboardShortcut(.cancelAction)}
            Text(L("Eventi JSON su stdin · output e decisioni JSON · timeout configurabile. Ogni definizione deve essere revisionata: le modifiche ne revocano la fiducia. Gli script hanno i permessi del tuo utente.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            List {ForEach(store.archive.settings.hooks ?? []){hook in
                HStack {
                    Toggle(hook.name,isOn:Binding(get:{store.archive.settings.hooks?.first{$0.id==hook.id}?.enabled ?? false},set:{value in
                        if value && !HookDefinition.trusted(hook) {editing=hook}
                        else if let i=store.archive.settings.hooks?.firstIndex(where:{$0.id==hook.id}) {store.archive.settings.hooks?[i].enabled=value;store.scheduleSave()}
                    }))
                    Text(HookDefinition.event(hook.event)).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                    if !HookDefinition.trusted(hook) {Image(systemName:"exclamationmark.shield").help(L("Da revisionare"))}
                    Button(L("Modifica / revisiona")){editing=hook}
                    Button(role:.destructive){store.archive.settings.hooks?.removeAll{$0.id==hook.id};store.scheduleSave()}label:{Image(systemName:"trash")}
                }
            }}.frame(minHeight:180)
            HStack {Button(L("Aggiungi hook")){editing=StudioHook()};Button(L("Importa hooks.json…")){store.importHooks()}}
        }.padding(24).frame(width:780,height:480).sheet(item:$editing){HookEditor(hook:$0).environmentObject(store)}
    }
}
struct HookEditor: View {
    @EnvironmentObject var store: StudioStore
    @Environment(\.dismiss) private var dismiss
    @State var hook: StudioHook
    @State private var accepted=false
    @State private var issue=""
    var body: some View {
        VStack(alignment:.leading,spacing:12) {
            Text(L("Configura e revisiona hook")).font(StudioTypography.ui(.title2).bold())
            Menu(L("Suggerimenti ed esempi…")) {ForEach(HookSamples.all,id:\.name){sample in Button(LT(sample.name)){hook.name=sample.name;hook.event=sample.event;hook.command=sample.command;hook.handlerType="command";hook.enabled=false;accepted=false}}}
            Text(L("Scegli un evento, filtra gli strumenti con una regex (es. ^Bash$), poi scrivi un comando. Riceve un oggetto JSON su stdin. Exit 2 può bloccare l’azione; altri errori vengono registrati. Gli esempi non vengono eseguiti dall’editor.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            TextField(L("Nome"),text:$hook.name).textFieldStyle(.roundedBorder)
            HStack {
                Picker(L("Evento"),selection:$hook.event){ForEach(HookDefinition.events,id:\.self){Text($0).tag($0)}}
                Picker("Handler",selection:Binding(get:{hook.handlerType ?? "command"},set:{hook.handlerType=$0})){Text(L("Comando")).tag("command");Text(L("Strumento MCP")).tag("mcp_tool")}
            }
            TextField(L("Matcher regex · vuoto = tutti"),text:Binding(get:{hook.matcher ?? ""},set:{hook.matcher=$0})).textFieldStyle(.roundedBorder)
            if hook.handlerType=="mcp_tool" {
                HStack {TextField(L("Nome server MCP attivo"),text:Binding(get:{hook.server ?? ""},set:{hook.server=$0}));TextField(L("Nome tool"),text:Binding(get:{hook.tool ?? ""},set:{hook.tool=$0}))}
                Text(L("Input JSON · supporta ${tool_input.path} e ${prompt}")).font(StudioTypography.ui(.caption))
                TextEditor(text:Binding(get:{hook.inputJSON ?? "{}"},set:{hook.inputJSON=$0})).font(.system(.body,design:.monospaced)).frame(height:140)
            } else {
                Text(L("Comando shell / codice")).font(StudioTypography.ui(.caption))
                TextEditor(text:$hook.command).font(.system(.body,design:.monospaced)).frame(height:170)
            }
            HStack {
                Text(L("Timeout (secondi)"));TextField("600",value:Binding(get:{hook.timeout ?? 600},set:{hook.timeout=$0}),format:.number).frame(width:80)
                Toggle(L("In background"),isOn:Binding(get:{hook.background ?? false},set:{hook.background=$0})).disabled(hook.event=="SessionEnd")
            }
            DisclosureGroup(L("Guida rapida")) {
                Text(L("SessionStart: contesto iniziale. UserPromptSubmit: valida il prompt. PreToolUse: blocca o riscrivi argomenti. PermissionRequest: consenti/nega. PostToolUse: aggiungi feedback. Stop: chiedi una continuazione. Matcher usa Bash per i comandi e apply_patch per la scrittura. Async: solo contesto al turno successivo, nessuna decisione. Pre/PostCompact sono accettati ma questo harness non compatta automaticamente.")).font(StudioTypography.ui(.caption))
            }
            Toggle(L("Ho letto questa definizione e autorizzo la sua esecuzione"),isOn:$accepted)
            Text(issue).font(StudioTypography.ui(.caption)).foregroundStyle(.red)
            HStack {Spacer();Button(L("Annulla")){dismiss()};Button(L("Salva disattivato")){save(trust:false)};Button(L("Autorizza e attiva")){save(trust:true)}.buttonStyle(.borderedProminent).disabled(!accepted)}
        }.padding(24).frame(width:740)
        .onAppear{hook.event=HookDefinition.event(hook.event)}
        .onChange(of:hook.command){_,_ in accepted=false}
    }
    private func save(trust: Bool) {
        do {
            guard !hook.name.isEmpty,(hook.timeout ?? 600)>0,(hook.timeout ?? 600)<=3600,hook.command.utf8.count<=32768 else{throw ExtensionFiles.failure(L("Nome, timeout o comando non valido."))}
            if let matcher=hook.matcher,!matcher.isEmpty,matcher != "*" {_ = try NSRegularExpression(pattern:matcher)}
            if hook.handlerType=="mcp_tool" {
                guard !(hook.server ?? "").isEmpty,!(hook.tool ?? "").isEmpty,try JSONSerialization.jsonObject(with:Data((hook.inputJSON ?? "{}").utf8)) is [String:Any] else{throw ExtensionFiles.failure(L("Server, tool e input JSON obbligatori."))}
            } else if hook.command.isEmpty {throw ExtensionFiles.failure(L("Comando obbligatorio."))}
            if trust {store.reviewHook(hook)} else {
                hook.enabled=false;hook.trustedDigest=nil
                var items=store.archive.settings.hooks ?? [];if let i=items.firstIndex(where:{$0.id==hook.id}){items[i]=hook}else{items.append(hook)}
                store.archive.settings.hooks=items;store.scheduleSave()
            }
            dismiss()
        } catch {issue=error.localizedDescription}
    }
}
