import SwiftUI
struct HooksSettingsView: View {
    @EnvironmentObject var store: StudioStore
    @Environment(\.dismiss) private var dismiss
    @State private var editing: StudioHook?
    var body: some View {
        VStack(alignment:.leading,spacing:16) {
            HStack { Text("Hook del ciclo di lavoro").font(StudioTypography.ui(.title2).bold());Spacer();Button("Chiudi") { dismiss() }.keyboardShortcut(.cancelAction) }
            Text("Script locali per sessioni con directory. Ereditano le approvazioni della sessione. Timeout 30 secondi; un errore prima dello strumento o della richiesta blocca l’operazione.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            List {
                ForEach(store.archive.settings.hooks ?? []){hook in
                    HStack{
                        Toggle(hook.name,isOn:Binding(get:{store.archive.settings.hooks?.first{$0.id==hook.id}?.enabled ?? false},set:{value in if let i=store.archive.settings.hooks?.firstIndex(where:{$0.id==hook.id}){store.archive.settings.hooks?[i].enabled=value;store.scheduleSave()}}))
                        Text(hook.event).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                        Button("Modifica"){editing=hook}
                        Button(role:.destructive){store.archive.settings.hooks?.removeAll{$0.id==hook.id};store.scheduleSave()}label:{Image(systemName:"trash")}
                    }
                }
            }.frame(minHeight:180)
            Button("Aggiungi hook"){editing=StudioHook()}
        }.padding(24).frame(width:700,height:450).sheet(item:$editing){HookEditor(hook:$0).environmentObject(store)}
    }
}
struct HookEditor: View {
    @EnvironmentObject var store: StudioStore
    @Environment(\.dismiss) private var dismiss
    @State var hook: StudioHook
    var body: some View {
        VStack(alignment:.leading,spacing:16){
            Menu("Usa un esempio…") {
                ForEach(HookSamples.all,id:\.name) { sample in Button(sample.name) { hook.name=sample.name;hook.event=sample.event;hook.command=sample.command;hook.enabled=false } }
            }
            Text("Gli esempi sono modificabili e inizialmente disattivati. Scegliere un esempio non esegue il comando.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            TextField("Nome",text:$hook.name).textFieldStyle(.roundedBorder)
            Picker("Evento",selection:$hook.event){Text("Prima della richiesta").tag("request_start");Text("Dopo la risposta").tag("request_end");Text("Prima dello strumento").tag("before_tool");Text("Dopo lo strumento").tag("after_tool")}
            TextEditor(text:$hook.command).font(StudioTypography.ui(.body,design:.monospaced)).frame(height:160)
            Text("Variabili: SUC_HOOK_EVENT, SUC_SESSION_ID, SUC_WORKSPACE, SUC_TOOL_NAME, SUC_TOOL_ARGUMENTS (JSON). Esecuzione zsh con i permessi dell’utente; nessuna shell sandbox.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            Toggle("Attivo",isOn:$hook.enabled)
            HStack{Spacer();Button("Annulla"){dismiss()};Button("Salva"){
                var items=store.archive.settings.hooks ?? [];if let i=items.firstIndex(where:{$0.id==hook.id}){items[i]=hook}else{items.append(hook)}
                store.archive.settings.hooks=items;store.scheduleSave();dismiss()
            }.buttonStyle(.borderedProminent).disabled(hook.name.isEmpty || hook.command.isEmpty || hook.command.utf8.count>32768 || (store.archive.settings.hooks?.count ?? 0)>32)}
        }.padding(24).frame(width:640)
    }
}
