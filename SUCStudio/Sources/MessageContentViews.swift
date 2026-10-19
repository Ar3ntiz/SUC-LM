import SwiftUI
import AppKit

struct ResponseBody:View {
    let text:String
    var reasoning:String?=nil
    var implicit=false
    var assistant=true
    var generating=false
    var body:some View {
        let parts=assistant ? ThinkingParts.split(text,separate:reasoning,implicit:implicit):ThinkingParts(reasoning:"",answer:text,active:false)
        VStack(alignment:.leading,spacing:14){
            if !parts.reasoning.isEmpty || parts.active {
                VStack(alignment:.leading,spacing:8){
                    Label(parts.active && generating ? "Ragionamento in corso":"Ragionamento",systemImage:"brain").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                    BoundedMessageBody(text:parts.reasoning,pointOffset:-1)
                }.padding(14).frame(maxWidth:.infinity,alignment:.leading)
                .background(studioAccent.opacity(0.045),in:RoundedRectangle(cornerRadius:12))
                .overlay(RoundedRectangle(cornerRadius:12).stroke(studioAccent.opacity(0.18),lineWidth:1))
            }
            if !parts.answer.isEmpty {BoundedMessageBody(text:parts.answer)}
        }
    }
}
struct BoundedMessageBody:View {
    let text:String
    var pointOffset:Double=0
    @State private var full=false
    var body:some View {
        Group{
            if MessagePresentation.isLong(text){
                VStack(alignment:.leading,spacing:8){
                    Text(String(text.prefix(2000))).font(StudioTypography.chat(pointOffset:pointOffset)).lineLimit(12).textSelection(.enabled)
                    Button("Apri testo completo · \(ByteCountFormatter.string(fromByteCount:Int64(text.utf8.count),countStyle:.file))"){full=true}.buttonStyle(.link)
                    Text("Anteprima limitata per mantenere la chat reattiva. Il testo integrale resta disponibile.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                }
            } else {MarkdownBody(text:text,pointOffset:pointOffset)}
        }.sheet(isPresented:$full){FullMessageView(text:text)}
    }
}
struct FullMessageView:View {
    let text:String
    @Environment(\.dismiss) private var dismiss
    var body:some View {
        VStack(spacing:12){
            HStack{Text("Testo completo").font(StudioTypography.ui(.headline));Spacer();Button("Copia"){copyText(text)};Button("Chiudi"){dismiss()}.keyboardShortcut(.cancelAction)}
            PlainTextEditor(text:.constant(text),editable:false)
        }.padding(20).frame(minWidth:600,idealWidth:800,minHeight:420,idealHeight:650)
    }
}
struct CodeInsertion:Identifiable {var id=UUID();var sessionID:UUID;var offset:Int}
struct CodeInsertSheet:View {
    @Environment(\.dismiss) private var dismiss
    @State private var code=""
    @State private var language=""
    var insert:(String)->Void
    var body:some View {
        VStack(alignment:.leading,spacing:12){
            HStack{Label("Incolla codice",systemImage:"chevron.left.forwardslash.chevron.right").font(StudioTypography.ui(.headline));Spacer();Button("Annulla"){dismiss()}.keyboardShortcut(.cancelAction)}
            TextField("Linguaggio (facoltativo: html, python, swift…)",text:$language).textFieldStyle(.roundedBorder)
            PlainTextEditor(text:$code)
            HStack{Text("Invio qui aggiunge una riga. Il codice verrà inserito nel messaggio, pronto da inviare.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary);Spacer();Button("Inserisci codice"){insert(MessagePresentation.codeFence(code,language:language));dismiss()}.buttonStyle(.borderedProminent).disabled(code.isEmpty)}
        }.padding(20).frame(minWidth:600,idealWidth:780,minHeight:420,idealHeight:600)
    }
}
struct PlainTextEditor:NSViewRepresentable {
    @Binding var text:String
    var editable=true
    func makeCoordinator()->Coordinator{Coordinator(self)}
    func makeNSView(context:Context)->NSScrollView{
        let scroll=NSScrollView();scroll.hasVerticalScroller=true;scroll.hasHorizontalScroller=true
        let view=NSTextView(frame:NSRect(x:0,y:0,width:600,height:400));view.isRichText=false;view.isEditable=editable;view.isSelectable=true;view.font = .monospacedSystemFont(ofSize:13,weight:.regular)
        view.isAutomaticLinkDetectionEnabled=false;view.isAutomaticDataDetectionEnabled=false;view.isContinuousSpellCheckingEnabled=false;view.isGrammarCheckingEnabled=false;view.isAutomaticQuoteSubstitutionEnabled=false;view.isAutomaticDashSubstitutionEnabled=false
        view.isVerticallyResizable=true;view.isHorizontallyResizable=true;view.minSize = .zero;view.maxSize=NSSize(width:CGFloat.greatestFiniteMagnitude,height:CGFloat.greatestFiniteMagnitude)
        view.textContainer?.containerSize=NSSize(width:CGFloat.greatestFiniteMagnitude,height:CGFloat.greatestFiniteMagnitude);view.textContainer?.widthTracksTextView=false;view.layoutManager?.allowsNonContiguousLayout=true
        view.textContainerInset=NSSize(width:8,height:8);view.delegate=context.coordinator;view.string=text;scroll.documentView=view;return scroll
    }
    func updateNSView(_ scroll:NSScrollView,context:Context){context.coordinator.parent=self;if let view=scroll.documentView as? NSTextView,view.string != text{view.string=text}}
    final class Coordinator:NSObject,NSTextViewDelegate{var parent:PlainTextEditor;init(_ parent:PlainTextEditor){self.parent=parent};func textDidChange(_ notification:Notification){if let view=notification.object as? NSTextView{parent.text=view.string}}}
}

/// Quiet, compact activity summaries; operational payloads remain in the session log.
struct ActivityMessageBox: View {
    let message: ChatMessage
    var body: some View {
        HStack(alignment:.top,spacing:7) {
            Image(systemName:message.kind=="action_notice" ? "exclamationmark.circle":"checklist")
            Text(message.compactActivityText).textSelection(.enabled)
        }
        .font(StudioTypography.ui(.caption2)).foregroundStyle(.secondary)
        .padding(.horizontal,10).padding(.vertical,7)
        .frame(maxWidth:.infinity,alignment:.leading)
        .background(Color.secondary.opacity(0.07),in:RoundedRectangle(cornerRadius:8))
        .contextMenu {Button("Copia"){copyText(message.compactActivityText)}}
        .accessibilityLabel(message.kind=="action_notice" ? "Avviso attività":"Riepilogo attività")
    }
}
