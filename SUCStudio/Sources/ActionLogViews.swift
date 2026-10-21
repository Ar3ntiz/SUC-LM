import SwiftUI

struct ActionLogView:View {
    @EnvironmentObject var store:StudioStore
    let actions:[ActionRecord]
    @State private var expanded=false
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            Button {expanded.toggle()} label:{
                HStack(spacing:8) {
                    Image(systemName:expanded ? "chevron.down":"chevron.right")
                    Label(L("Registro azioni"),systemImage:"list.bullet.rectangle")
                    Spacer()
                    Text("\(actions.count)").monospacedDigit().foregroundStyle(.secondary)
                }.font(StudioTypography.ui(.body)).padding(.vertical,8)
                    .frame(maxWidth:.infinity,minHeight:38,alignment:.leading).contentShape(Rectangle())
            }.buttonStyle(.plain).accessibilityIdentifier("toggle-action-log")
            if expanded {
                ScrollView {
                    LazyVStack(spacing:10) {
                        ForEach(actions.suffix(30).reversed()) {record in
                            ActionLogRow(record:record,job:store.commandJobs.jobs.first{$0.recordID==record.id})
                        }
                    }.padding(.trailing,4)
                }.frame(minHeight:240,maxHeight:380)
                Text(L("Azioni più recenti in alto · ultime 30 voci")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            }
        }
    }
}

struct ActionLogRow:View {
    let record:ActionRecord
    var job:CommandJob?=nil
    @State private var expanded=false
    @State private var showFull=false
    private var title:String {
        LT(["run_command":"Comando shell","run_interactive":"Comando interattivo","command_status":"Stato comando","wait_command":"Attesa comando","stop_command":"Interruzione comando","read_file":"Lettura file","write_file":"Scrittura file","list_files":"Elenco directory","make_directory":"Creazione directory","web_search":"Ricerca web","visit_page":"Lettura pagina","mcp_call":"Strumento MCP","load_skill":"Caricamento skill","load_plugin":"Caricamento plugin"][record.name] ?? record.name)
    }
    private var color:Color {record.state=="completed" ? .green:["error","denied"].contains(record.state) ? .red:record.state=="running" ? .orange:.secondary}
    private var status:String {LT(["completed":"Completata","running":"In esecuzione","error":"Errore","denied":"Rifiutata","cancelled":"Interrotta","interrupted":"Interrotta"][record.state] ?? record.state)}
    private var preview:String {String(record.details.split(separator:"\n").dropFirst().joined(separator:" · ").prefix(180))}
    var body:some View {
        VStack(alignment:.leading,spacing:0) {
            Button {expanded.toggle()} label:{
                VStack(alignment:.leading,spacing:7) {
                    HStack(spacing:8) {
                        Image(systemName:record.state=="completed" ? "checkmark.circle.fill":record.state=="running" ? "hourglass":"exclamationmark.circle").foregroundStyle(color)
                        Text(title).fontWeight(.semibold)
                        Spacer(minLength:8)
                        Image(systemName:expanded ? "chevron.down":"chevron.right").foregroundStyle(.secondary)
                    }.font(StudioTypography.ui(.body))
                    HStack {Text(status).foregroundStyle(color);Spacer();Text(record.date,style:.time).foregroundStyle(.secondary)}.font(StudioTypography.ui(.caption))
                    if !preview.isEmpty {Text(preview).font(StudioTypography.ui(.callout)).foregroundStyle(.secondary).lineLimit(2).multilineTextAlignment(.leading)}
                }.padding(14).frame(maxWidth:.infinity,alignment:.leading).contentShape(Rectangle())
            }.buttonStyle(.plain).accessibilityIdentifier("toggle-action-details")
            if expanded {
                Divider().padding(.horizontal,14)
                VStack(alignment:.leading,spacing:14) {
                    Text(L("Richiesta")).font(StudioTypography.ui(.subheadline)).fontWeight(.semibold)
                    textBlock(record.details)
                    HStack {Text(L("Risultato")).font(StudioTypography.ui(.subheadline)).fontWeight(.semibold);Spacer();Button(L("Copia risultato completo")){copyText(record.result)}.buttonStyle(.link)}
                    if let job,job.running {
                        TimelineView(.periodic(from:.now,by:1)) {_ in
                            let snapshot=job.snapshot()
                            Text(L("In esecuzione")+" · "+String(format:"%.0f s",snapshot.elapsed)).font(StudioTypography.ui(.body)).foregroundStyle(.orange)
                            textBlock(snapshot.output.isEmpty ? L("Nessun output testuale."):snapshot.output)
                        }
                    } else {textBlock(readableResult)}
                    if record.result.count>16000 {Button(L("Apri risultato completo")){showFull=true}.buttonStyle(.link)}
                }.padding(14)
            }
        }.background(Color.secondary.opacity(0.055),in:RoundedRectangle(cornerRadius:12))
            .overlay(RoundedRectangle(cornerRadius:12).stroke(Color.secondary.opacity(0.14),lineWidth:1))
            .sheet(isPresented:$showFull){FullMessageView(text:record.result)}
    }
    private func textBlock(_ text:String)->some View {
        ScrollView {
            Text(String(text.prefix(16000))).font(StudioTypography.ui(.body,design:.monospaced)).textSelection(.enabled)
                .frame(maxWidth:.infinity,alignment:.leading).padding(10)
        }.frame(maxHeight:220).background(Color.secondary.opacity(0.045),in:RoundedRectangle(cornerRadius:8))
    }
    private var readableResult:String {
        guard !record.result.isEmpty else{return L("Nessun output testuale.")}
        guard record.result.utf8.count<=128*1024,
              let object=try? JSONSerialization.jsonObject(with:Data(record.result.utf8)) as? [String:Any],object["job_id"] != nil else{return record.result}
        var lines=[String]()
        if let code=object["exit_code"] as? Int {lines.append(L("Codice di uscita")+": \(code)")}
        if let seconds=object["elapsed_seconds"] as? Double {lines.append(L("Durata")+": "+String(format:"%.1f s",seconds))}
        if let error=object["error"] as? String,!error.isEmpty {lines.append(error)}
        lines.append((object["output"] as? String).flatMap{$0.isEmpty ? nil:$0} ?? L("Nessun output testuale."))
        if let omitted=object["omitted_bytes"] as? Int,omitted>0 {lines.append(L("Anteprima: \(omitted) byte precedenti non mostrati."))}
        return lines.joined(separator:"\n\n")
    }
}
