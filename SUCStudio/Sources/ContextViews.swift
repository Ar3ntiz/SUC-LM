import SwiftUI

struct ContextIndicator:View {
    @ObservedObject var store:StudioStore
    @State private var open=false
    @State private var sampled=ContextUsage(used:0,limit:nil,estimated:true)
    var body:some View {
        if let session=store.selected {
            let usage=sampled
            Button{open.toggle()} label:{
                ZStack {
                    Circle().stroke(.secondary.opacity(0.2),lineWidth:3)
                    Circle().trim(from:0,to:usage.fraction ?? 0).stroke((usage.fraction ?? 0)<0.9 ? Color.accentColor:Color.orange,style:StrokeStyle(lineWidth:3,lineCap:.round)).rotationEffect(.degrees(-90))
                    Text(usage.fraction.map{(usage.estimated ? "≈":"")+"\(Int($0*100))"} ?? "?").font(.system(size:8,weight:.medium,design:.rounded))
                }.frame(width:25,height:25)
            }.buttonStyle(.plain).help(L("Contesto: \(usage.estimated ? "≈":"")\(usage.used) / \(usage.limit.map(String.init) ?? "?") token")).accessibilityLabel(L("Utilizzo del contesto"))
                .task(id:session.id) {
                    sampled=ContextUsage(used:0,limit:store.contextLimit(session),estimated:true)
                    while !Task.isCancelled {
                        if let value=await store.sampledContextUsage(session.id),!Task.isCancelled {sampled=value}
                        do{try await Task.sleep(nanoseconds:1_000_000_000)}catch{return}
                    }
                }
                .popover(isPresented:$open){ContextControls(store:store,sessionID:session.id).padding(16).frame(width:340)}
        }
    }
}
struct ContextControls:View {
    @ObservedObject var store:StudioStore
    let sessionID:UUID
    @State private var limit=""
    @State private var usage=ContextUsage(used:0,limit:nil,estimated:true)
    var body:some View {
        if let session=store.archive.sessions.first(where:{$0.id==sessionID}) {
            VStack(alignment:.leading,spacing:12) {
                Text(L("Contesto della sessione")).font(.headline)
                Text(L("Contesto: \(usage.estimated ? "≈":"")\(usage.used) / \(usage.limit.map(String.init) ?? "limite sconosciuto") token")).font(.caption).textSelection(.enabled)
                Text(L("≈ indica una stima. I conteggi effettivi del server hanno precedenza. Include istruzioni, storico attivo, strumenti e allegati; le immagini possono avere costi variabili.")).font(.caption).foregroundStyle(.secondary)
                HStack {TextField(L("Limite (token), 0 = automatico"),text:$limit).textFieldStyle(.roundedBorder)
                    Button(L("Applica")) {
                        guard let value=Int(limit),value==0 || (1024...10_000_000).contains(value) else{store.error=L("Limite non valido: usa 0 oppure almeno 1024 token.");return}
                        store.mutateSession(sessionID){$0.contextWindow=value==0 ? nil:value}
                    }
                }
                Toggle(L("Auto-compact al 99%"),isOn:Binding(get:{session.autoCompact != false},set:{value in store.mutateSession(sessionID){$0.autoCompact=value}}))
                Text(L("Una riserva per la risposta può anticipare la compattazione. Il riassunto usa il modello corrente; lo storico originale rimane disponibile.")).font(.caption).foregroundStyle(.secondary)
                Button(L("Compatta ora (/compact)")){store.requestCompaction(sessionID)}.disabled(store.compacting.contains(sessionID))
            }.onAppear{limit=String(session.contextWindow ?? 0)}
            .task(id:sessionID){
                while !Task.isCancelled {
                    if let value=await store.sampledContextUsage(sessionID),!Task.isCancelled{usage=value}
                    do{try await Task.sleep(nanoseconds:1_000_000_000)}catch{return}
                }
            }
        }
    }
}
struct CompactionBanner:View {
    @ObservedObject var store:StudioStore
    var body:some View {
        if let id=store.archive.selected,store.compacting.contains(id) {
            HStack {
                if let progress=store.compactionProgress[id] {ProgressView(value:Double(progress.done),total:Double(max(1,progress.total))).frame(width:90)} else{ProgressView().controlSize(.small)}
                Text(L("Compattazione del contesto…")).font(StudioTypography.ui(.caption))
                if let budget=store.compactionProgress[id]?.outputBudget,budget>0 {Text(L("Budget: \(budget) token")).font(StudioTypography.ui(.caption2)).foregroundStyle(.secondary)}
                Spacer();Button(L("Interrompi")){store.cancelSelected()}
            }
        }
    }
}
