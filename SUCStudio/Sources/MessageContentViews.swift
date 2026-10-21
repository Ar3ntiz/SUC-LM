import SwiftUI
import AppKit

struct ResponseBody:View {
    let text:String
    var reasoning:String?=nil
    var implicit=false
    var assistant=true
    var generating=false
    var prepared:ThinkingParts?=nil
    var body:some View {
        let parts=prepared ?? (assistant ? ThinkingParts.split(text,separate:reasoning,implicit:implicit):ThinkingParts(reasoning:"",answer:text,active:false))
        VStack(alignment:.leading,spacing:14){
            if !parts.reasoning.isEmpty || parts.active {
                ReasoningBody(text:parts.reasoning,generating:generating,active:parts.active)

            }
            if !parts.answer.isEmpty {BoundedMessageBody(text:parts.answer)}
        }
    }
}
/// Historical reasoning starts collapsed; a live block opens and never auto-collapses
/// on completion, avoiding height jumps. Users can toggle it at any time.
struct ReasoningBody:View {
    let text:String
    let generating:Bool
    let active:Bool
    @State private var expanded=false
    var body:some View {
        DisclosureGroup(isExpanded:$expanded) {
            BoundedMessageBody(text:text,pointOffset:-1).padding(.top,8)
        } label:{
            Label(active && generating ? L("Ragionamento in corso"):L("Ragionamento"),systemImage:"brain")
                .font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
        }
        .padding(14).frame(maxWidth:.infinity,alignment:.leading)
        .background(studioAccent.opacity(0.045),in:RoundedRectangle(cornerRadius:12))
        .overlay(RoundedRectangle(cornerRadius:12).stroke(studioAccent.opacity(0.18),lineWidth:1))
        .onAppear{if generating{expanded=true}}
    }
}
struct BoundedMessageBody:View {
    let text:String
    var pointOffset:Double=0
    @State private var full=false
    var body:some View {
        VStack(alignment:.leading,spacing:8){
            // Keep the same Markdown view and preview height across the long-text threshold.
            MarkdownBody(text:text,pointOffset:pointOffset)
            if MessagePresentation.isLong(text){
                Button(L("Apri testo completo · \(ByteCountFormatter.string(fromByteCount:Int64(text.utf8.count),countStyle:.file))")){full=true}.buttonStyle(.link)
                Text(L("Anteprima fino a 8000 caratteri; il testo integrale resta disponibile.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
            }
        }.sheet(isPresented:$full){FullMessageView(text:text)}
    }
}
struct FullMessageView:View {
    let text:String
    @Environment(\.dismiss) private var dismiss
    var body:some View {
        VStack(spacing:12){
            HStack{Text(L("Testo completo")).font(StudioTypography.ui(.headline));Spacer();Button(L("Copia")){copyText(text)};Button(L("Chiudi")){dismiss()}.keyboardShortcut(.cancelAction)}
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
            HStack{Label(L("Incolla codice"),systemImage:"chevron.left.forwardslash.chevron.right").font(StudioTypography.ui(.headline));Spacer();Button(L("Annulla")){dismiss()}.keyboardShortcut(.cancelAction)}
            TextField(L("Linguaggio (facoltativo: html, python, swift…)"),text:$language).textFieldStyle(.roundedBorder)
            PlainTextEditor(text:$code)
            HStack{Text(L("Invio qui aggiunge una riga. Il codice verrà inserito nel messaggio, pronto da inviare.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary);Spacer();Button(L("Inserisci codice")){insert(MessagePresentation.codeFence(code,language:language));dismiss()}.buttonStyle(.borderedProminent).disabled(code.isEmpty)}
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
            Text(StudioL10n.report(message.kind=="command_help" ? message.text:message.compactActivityText)).textSelection(.enabled)
        }
        .font(StudioTypography.ui(.caption2)).foregroundStyle(.secondary)
        .padding(.horizontal,10).padding(.vertical,7)
        .frame(maxWidth:.infinity,alignment:.leading)
        .background(Color.secondary.opacity(0.07),in:RoundedRectangle(cornerRadius:8))
        .contextMenu {Button(L("Copia")){copyText(message.compactActivityText)}}
        .accessibilityLabel(message.kind=="action_notice" ? L("Avviso attività"):L("Riepilogo attività"))
    }
}

/// Keep execution details in the transcript without exposing wire JSON or terminal input.
struct CommandDisclosure:View {
    let action:WorkspaceAction
    @State private var expanded=false
    var body:some View {
        DisclosureGroup(isExpanded:$expanded) {
            VStack(alignment:.leading,spacing:8) {
                Text(action.arguments["command"] ?? L("Comando non disponibile"))
                    .font(StudioTypography.chat(pointOffset:-1).monospaced())
                    .textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading)
                if action.name=="run_interactive" {
                    Text(L("L’eventuale input riservato viene richiesto nella scheda terminale."))
                        .font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                }
            }.padding(.top,6)
        } label: {
            Label(action.name=="run_interactive" ? L("Comando interattivo"):L("Comando"),systemImage:"terminal")
                .font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
        }.padding(10).background(Color.secondary.opacity(0.06),in:RoundedRectangle(cornerRadius:10))
    }
}
