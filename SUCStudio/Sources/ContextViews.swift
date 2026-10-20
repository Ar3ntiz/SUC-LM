import SwiftUI

struct ContextIndicator:View {
    @ObservedObject var store:StudioStore
    @State private var open=false
    var body:some View {
        if let session=store.selected {
            let usage=store.contextUsage(session)
            Button{open.toggle()} label:{
                ZStack {
                    Circle().stroke(.secondary.opacity(0.2),lineWidth:3)
                    Circle().trim(from:0,to:usage.fraction ?? 0).stroke((usage.fraction ?? 0)<0.9 ? Color.accentColor:Color.orange,style:StrokeStyle(lineWidth:3,lineCap:.round)).rotationEffect(.degrees(-90))
                    Text(usage.fraction.map{(usage.estimated ? "≈":"")+"\(Int($0*100))"} ?? "?").font(.system(size:8,weight:.medium,design:.rounded))
                }.frame(width:25,height:25)
            }.buttonStyle(.plain).help(store.contextDescription(session)).accessibilityLabel("Utilizzo del contesto. "+store.contextDescription(session))
                .popover(isPresented:$open){ContextControls(store:store,sessionID:session.id).padding(16).frame(width:340)}
        }
    }
}
struct ContextControls:View {
    @ObservedObject var store:StudioStore
    let sessionID:UUID
    @State private var limit=""
    var body:some View {
        if let session=store.archive.sessions.first(where:{$0.id==sessionID}) {
            VStack(alignment:.leading,spacing:12) {
                Text("Contesto della sessione").font(.headline)
                Text(store.contextDescription(session)).font(.caption).textSelection(.enabled)
                Text("≈ indica una stima. I conteggi effettivi del server hanno precedenza. Include istruzioni, storico attivo, strumenti e allegati; le immagini possono avere costi variabili.").font(.caption).foregroundStyle(.secondary)
                HStack {TextField("Limite (token), 0 = automatico",text:$limit).textFieldStyle(.roundedBorder)
                    Button("Applica") {
                        guard let value=Int(limit),value==0 || (1024...10_000_000).contains(value) else{store.error="Limite non valido: usa 0 oppure almeno 1024 token.";return}
                        store.mutateSession(sessionID){$0.contextWindow=value==0 ? nil:value}
                    }
                }
                Toggle("Auto-compact al 99%",isOn:Binding(get:{session.autoCompact != false},set:{value in store.mutateSession(sessionID){$0.autoCompact=value}}))
                Text("Una riserva per la risposta può anticipare la compattazione. Il riassunto usa il modello corrente; lo storico originale rimane disponibile.").font(.caption).foregroundStyle(.secondary)
                Button("Compatta ora (/compact)"){store.requestCompaction(sessionID)}.disabled(store.compacting.contains(sessionID))
            }.onAppear{limit=String(session.contextWindow ?? 0)}
        }
    }
}
struct CompactionBanner:View {
    @ObservedObject var store:StudioStore
    var body:some View {
        if let id=store.archive.selected,store.compacting.contains(id) {
            HStack {
                if let progress=store.compactionProgress[id] {ProgressView(value:Double(progress.done),total:Double(max(1,progress.total))).frame(width:90)} else{ProgressView().controlSize(.small)}
                Text("Compattazione del contesto…").font(StudioTypography.ui(.caption))
                Spacer();Button("Interrompi"){store.cancelSelected()}
            }
        }
    }
}
