import SwiftUI

struct GlobalActivityView:View {
    @EnvironmentObject var store:StudioStore
    var body:some View {
        VStack(spacing:0) {
            if store.modelState=="loading" {
                ModelLoadingActivity(phase:store.loadPhase,memory:store.loadMemory,path:store.archive.settings.modelPath)
            }
            if let job=store.commandJobs.jobs.first(where:{$0.running && ($0.sessionID != store.archive.selected || store.tab != "chat")}) {
                HStack {Label(L("Comando in esecuzione in un’altra chat"),systemImage:"terminal");Spacer();Button(L("Apri chat")){store.archive.selected=job.sessionID;store.tab="chat";store.scheduleSave()}}
                    .font(StudioTypography.ui(.caption)).padding(10).background(studioAccent.opacity(0.06))
            }
            if let item=store.approvals.first(where:{$0.sessionID != store.archive.selected || store.tab != "chat"}) {
                HStack { Label(L("Approvazione richiesta · ")+item.title,systemImage:"hand.raised").lineLimit(1);Spacer();Button(L("Apri chat")) {store.archive.selected=item.sessionID;store.tab="chat";store.scheduleSave()} }.padding(12).background(Color.orange.opacity(0.10))
            }
        }.font(StudioTypography.ui(.callout))
    }
}
private struct ModelLoadingActivity:View {
    let phase:String
    let memory:LoadMemoryStatus?
    let path:String
    var body:some View {
        HStack(spacing:10) {
            VStack(alignment:.leading,spacing:2) {
                Text(L("Caricamento del modello…")).fontWeight(.medium)
                Text(LT(phase)).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                ProgressView().progressViewStyle(.linear).frame(maxWidth:400)
                    .accessibilityLabel(L("Caricamento in corso"))
                memoryReadout
                Text(URL(fileURLWithPath:path).lastPathComponent).font(StudioTypography.ui(.caption)).lineLimit(1)
            }
            Spacer()
            Text("C++ / Metal").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
        }.padding(12).background(studioAccent.opacity(0.08)).accessibilityElement(children:.combine)
    }
    @ViewBuilder private var memoryReadout:some View {
        if let memory {
            Text(LT(memory.label)).monospacedDigit().font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                .help(memory.detail+L(". La residenza può aumentare o diminuire durante il caricamento. Non misura il completamento: il modello è pronto solo quando il motore lo conferma."))
            if let metal=memory.metalLabel {Text(metal).font(StudioTypography.ui(.caption)).foregroundStyle(.tertiary)}
        }
    }
}
struct GenerationActivity:View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @EnvironmentObject var store:StudioStore
    let state:String
    var messageID:UUID? = nil
    var progress:PrefillStatus? { messageID.flatMap { store.prefillProgress[$0] } }
    var rereading:Bool {
        guard let progress,progress.total>=2048,progress.cached==0,let id=messageID,
              let session=store.archive.sessions.first(where:{$0.messages.contains(where:{$0.id==id})}),
              let index=session.messages.firstIndex(where:{$0.id==id}) else{return false}
        return session.messages[..<index].contains { $0.role=="assistant" && (!$0.text.isEmpty || !($0.reasoning ?? "").isEmpty) }
    }
    var decoding:Bool{state=="generating"}
    var label:String{decoding ? L("Generazione della risposta"):state=="queued" ? L("In attesa del modello"):state=="remote_wait" ? L("Attesa remota · prefill non segnalato"):L("Lettura del contesto · prefill")}
    var body:some View {
        HStack(spacing:10) {
            if reduceMotion {Image(systemName:decoding ? "waveform":"sparkle").foregroundStyle(studioAccent)}
            else {
                TimelineView(.animation(minimumInterval:1.0/15)) { context in
                    let t=context.date.timeIntervalSinceReferenceDate
                    if decoding {
                        HStack(spacing:3) {ForEach(0..<4,id:\.self) { i in Capsule().fill(studioAccent.opacity(0.8)).frame(width:3,height:CGFloat(5+9*(0.5+0.5*sin(t*5+Double(i))))) }}.frame(width:22,height:22)
                    } else {
                        Image(systemName:"sparkle").font(.system(size:19)).foregroundStyle(studioAccent).rotationEffect(.degrees(t*45)).scaleEffect(0.9+0.1*sin(t*3)).frame(width:22,height:22)
                    }
                }.accessibilityHidden(true)
            }
            VStack(alignment:.leading,spacing:6) {
                Text(label).font(StudioTypography.ui(.callout)).foregroundStyle(.secondary)
                if state=="prefill" || state=="remote_wait" {
                    if let progress {
                        ProgressView(value:progress.fraction).progressViewStyle(.linear)
                        Text(LT(progress.detail)).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary).monospacedDigit()
                        if rereading { Label(L("Cache non riutilizzata: rilettura completa della sessione."),systemImage:"arrow.clockwise").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary) }
                    } else {
                        ProgressView().progressViewStyle(.linear)
                        Text(L("Avanzamento non ancora disponibile dal motore.")).font(StudioTypography.ui(.caption2)).foregroundStyle(.tertiary)
                    }
                }
            }.frame(maxWidth:420,alignment:.leading)
        }.accessibilityElement(children:.combine).padding(.vertical,4)
    }
}
