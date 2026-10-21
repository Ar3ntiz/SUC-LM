import SwiftUI

struct ChatQueuePanel: View {
    @ObservedObject var store: StudioStore
    var body: some View {
        if let session=store.selected,let queued=session.queuedMessages,!queued.isEmpty {
            VStack(alignment:.leading,spacing:6) {
                HStack {
                    Label(L("In coda · \(queued.count)"),systemImage:"text.line.first.and.arrowtriangle.forward")
                    if session.queuePaused==true {Text(L("In pausa")).foregroundStyle(.secondary)}
                    Spacer()
                    Button(session.queuePaused==true ? L("Riprendi"):L("Pausa")) {
                        if session.queuePaused==true {store.resumeChatQueue(session.id)} else {store.pauseChatQueue(session.id)}
                    }.buttonStyle(.borderless)
                }
                ScrollView {
                    VStack(alignment:.leading,spacing:6) {
                        ForEach(Array(queued.enumerated()),id:\.element.id) { index,item in
                            HStack(alignment:.top) {
                                Text("\(index+1).").foregroundStyle(.secondary)
                                VStack(alignment:.leading,spacing:2) {
                                    Text(item.text.isEmpty ? L("Allegati"):item.text).lineLimit(2).frame(maxWidth:.infinity,alignment:.leading)
                                    if let files=item.attachments,!files.isEmpty {Text(files.map(\.name).joined(separator:" · ")).lineLimit(1).foregroundStyle(.secondary)}
                                }
                                Button {store.editQueuedMessage(item.id,sessionID:session.id)} label:{Image(systemName:"pencil")}.help(L("Riporta in bozza per modificare"))
                                Button {store.removeQueuedMessage(item.id,sessionID:session.id)} label:{Image(systemName:"xmark")}.help(L("Rimuovi dalla coda"))
                            }.buttonStyle(.borderless)
                        }
                    }
                }.frame(height:min(144,CGFloat(queued.count)*42))
            }.font(StudioTypography.ui(.caption)).padding(10).background(.quaternary,in:RoundedRectangle(cornerRadius:10))
        }
    }
}
