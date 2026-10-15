import SwiftUI
struct OperationLogView:View {
    @EnvironmentObject var store:StudioStore
    @State private var expanded:Bool
    init(initiallyExpanded:Bool=false){_expanded=State(initialValue:initiallyExpanded)}
    var body:some View {
        DisclosureGroup(store.archive.settings.ioLogEnabled ? "Registro log (\(store.modelIOLog.entries.count))":"Registro log · disattivato",isExpanded:$expanded) {
            VStack(alignment:.leading,spacing:10) {
                Toggle("Abilita Registro log I/O",isOn:Binding(get:{store.archive.settings.ioLogEnabled},set:{store.setModelIOLogEnabled($0)}))
                HStack {Text(store.archive.settings.ioLogEnabled ? "Tail in tempo reale degli input e output del modello. Solo memoria, massimo 256 KiB.":"Disattivato: l’app e il motore C++ non raccolgono né inviano eventi I/O.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary);Spacer();Button("Svuota"){store.modelIOLog.clear()}.disabled(store.modelIOLog.entries.isEmpty)}
                if store.modelIOLog.entries.isEmpty {Text(store.archive.settings.ioLogEnabled ? "Nessun input/output registrato.":"Attivalo solo quando serve diagnosticare una richiesta.").foregroundStyle(.secondary)}
                else {
                    ScrollViewReader {reader in ScrollView {
                        LazyVStack(alignment:.leading,spacing:10) {
                            ForEach(store.modelIOLog.entries) { entry in
                                VStack(alignment:.leading,spacing:4) {
                                    HStack {Text(entry.date,format:.dateTime.hour().minute().second()).monospacedDigit();Text(entry.direction).fontWeight(.semibold);Text(String(entry.requestID.prefix(8))).foregroundStyle(.tertiary)}.foregroundStyle(entry.direction=="ERRORE" ? Color.red:Color.secondary)
                                    Text(entry.text).font(StudioTypography.ui(.caption,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading)
                                }.id(entry.id)
                                Divider()
                            }
                            Color.clear.frame(height:1).id("model-log-bottom")
                        }.frame(maxWidth:.infinity,alignment:.leading)
                    }.frame(maxHeight:320).scrollIndicators(.visible).onAppear{reader.scrollTo("model-log-bottom",anchor:.bottom)}.onChange(of:store.modelIOLog.entries.last?.text){_,_ in reader.scrollTo("model-log-bottom",anchor:.bottom)}}
                }
            }.padding(.top,10)
        }.accessibilityIdentifier("model-io-log")
    }
}
