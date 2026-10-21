import SwiftUI
struct WorkspaceSettingsView: View {
    @EnvironmentObject var store: StudioStore
    var body: some View {
        VStack(alignment:.leading,spacing:12) {
            HStack { Text(L("Directory di lavoro")).font(StudioTypography.ui(.headline));Spacer();Button(L("Scegli…")){store.pickWorkspace()} }
            Text(store.selected?.workspacePath ?? L("Nessuna directory selezionata")).font(StudioTypography.ui(.caption)).textSelection(.enabled).lineLimit(3)
            Toggle(L("Consenti azioni del modello"),isOn:Binding(get:{store.selected?.agentEnabled ?? false},set:{value in store.mutateSelected{$0.agentEnabled=value}})).disabled((store.selected?.workspacePath ?? "").isEmpty)
            Picker(L("Approvazioni"),selection:Binding(get:{store.selected?.approvalMode ?? "manual"},set:{value in store.mutateSelected{$0.approvalMode=value}})) {Text(L("Manuali")).tag("manual");Text(L("Automatiche")).tag("automatic")}.pickerStyle(.segmented)
            if store.selected?.parentSessionID == nil {
                Toggle(L("Modello capace di delega: abilita subagent"),isOn:Binding(get:{store.selected?.allowSubagents ?? false},set:{value in store.mutateSelected{$0.allowSubagents=value}}))
                Text(L("Richiede che il modello segua il protocollo strumenti. Massimo 4 subtask per turno, un livello di delega.")).font(StudioTypography.ui(.caption2)).foregroundStyle(.secondary)
            }
            if let tasks=store.selected?.subtasks,!tasks.isEmpty {
                DisclosureGroup(L("Subtask (\(tasks.count))")){ForEach(tasks){task in Button{store.archive.selected=task.sessionID;store.scheduleSave()}label:{Label("\(task.title) · \(task.state)",systemImage:"arrow.triangle.branch").lineLimit(2)}.buttonStyle(.plain)}}
            }
            Toggle(L("Consenti comandi shell"),isOn:Binding(get:{store.selected?.shellCommandsEnabled ?? false},set:{value in store.mutateSelected{$0.allowCommands=value}}))
                .disabled(!(store.selected?.fullAgent ?? false))
            if !(store.selected?.fullAgent ?? false) {
                Text(L("Per usare i comandi shell, seleziona una directory di lavoro e attiva Consenti azioni del modello.")).font(StudioTypography.ui(.caption2)).foregroundStyle(.secondary)
            }
            Toggle(L("Consenti strumenti web (web_search, visit_page)"),isOn:Binding(get:{store.selected?.allowWeb ?? true},set:{value in store.mutateSelected{$0.allowWeb=value}}))
            Text(L("Gli strumenti web funzionano anche senza directory di lavoro; ogni azione segue la modalità di approvazione.")).font(StudioTypography.ui(.caption2)).foregroundStyle(.secondary)
            Text(L("File limitati alla directory; i comandi shell operano con i permessi del tuo account e possono accedere anche all’esterno. In automatico le azioni abilitate non chiedono conferma.")).font(StudioTypography.ui(.caption2)).foregroundStyle(.secondary)
            if let actions=store.selected?.actions, !actions.isEmpty {
                DisclosureGroup(L("Registro azioni (\(actions.count))")) { ScrollView {VStack(alignment:.leading,spacing:10){ForEach(actions.suffix(30)){item in Text("\(item.name) · \(item.state)\n\(item.result)").font(StudioTypography.ui(.caption)).textSelection(.enabled)}}}.frame(maxHeight:160) }
            }
        }.disabled(store.selected.map { session in store.agentRuns.contains(session.id) || store.requests.values.contains(where: { $0.session == session.id }) } == true)
    }
}
struct ActionApprovalView: View {
    @EnvironmentObject var store: StudioStore
    let approval: ActionApproval
    var body: some View {
        VStack(alignment:.leading,spacing:16) {
            Text(L("Approva azione")).font(StudioTypography.ui(.title2).bold())
            Text(approval.title).font(StudioTypography.ui(.headline))
            Text(approval.directory).font(StudioTypography.ui(.caption)).textSelection(.enabled)
            ScrollView {Text(approval.action.summary).font(StudioTypography.ui(.body,design:.monospaced)).frame(maxWidth:.infinity,alignment:.leading).textSelection(.enabled)}.frame(minHeight:80,maxHeight:220)
            if WebTools.names.contains(approval.action.name) || approval.action.name == "avvio_browser" {Text(approval.action.name == "avvio_browser" ? L("Il browser integrato lavora nascosto con un profilo dedicato: puoi guardarlo dal pulsante globo nella barra della chat. La conferma vale fino all’uscita dall’app.") : L("La pagina viene aperta nel browser integrato nascosto (pulsante globo per guardarla); il contenuto web è un dato non attendibile.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)}
            if approval.action.name == "run_command" {Text(L("Il comando usa i permessi del tuo account; la directory non è una sandbox del sistema operativo.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)}
            HStack{Spacer();Button(L("Rifiuta")){store.resolveApproval(approval.id,allowed:false)};Button(L("Approva questa azione")){store.resolveApproval(approval.id,allowed:true)}.buttonStyle(.borderedProminent)}
        }.padding(18).frame(maxWidth:.infinity,alignment:.leading).background(Color.orange.opacity(0.08),in:RoundedRectangle(cornerRadius:12)).overlay(RoundedRectangle(cornerRadius:12).stroke(Color.orange.opacity(0.3)))
    }
}
