import SwiftUI
import AppKit

struct MarkdownBody: View {
    let text:String
    var pointOffset:Double=0
    @State private var prepared=PreparedMarkdown(blocks:[],formatted:[:])
    @State private var pendingText=""
    @State private var renderingTask:Task<Void,Never>?
    var body: some View {
        VStack(alignment:.leading,spacing:12) {
            if prepared.blocks.isEmpty,!text.isEmpty {Text(MessagePresentation.preview(text)).textSelection(.enabled)}
            ForEach(Array(prepared.blocks.enumerated()),id:\.offset) { _,block in
                switch block.kind {
                case .paragraph: inline(block.text)
                case .heading(let level): inline(block.text).font(StudioTypography.chat(scale:level==1 ? 1.45:level==2 ? 1.25:1.1,weight:.bold,pointOffset:pointOffset)).padding(.top,6)
                case .list(let marker,let indent): HStack(alignment:.top,spacing:8) { Text(marker).foregroundStyle(.secondary);inline(block.text) }.padding(.leading,CGFloat(indent*16))
                case .quote: HStack(alignment:.top,spacing:12) { Rectangle().fill(studioAccent.opacity(0.5)).frame(width:3);inline(block.text).foregroundStyle(.secondary) }.fixedSize(horizontal:false,vertical:true)
                case .rule: Divider()
                case .code(let language):
                    VStack(alignment:.leading,spacing:0) {
                        HStack { Text(language.isEmpty ? "codice":language);Spacer();Button("Copia") { copyText(block.text) }.buttonStyle(.plain) }.font(StudioTypography.ui(.caption)).padding(10).background(.quaternary)
                        ScrollView(.horizontal) { Text(highlight(block.text)).font(StudioTypography.code(pointOffset:pointOffset)).textSelection(.enabled).padding(12) }
                    }.background(Color(nsColor:.controlBackgroundColor),in:RoundedRectangle(cornerRadius:10)).clipShape(RoundedRectangle(cornerRadius:10))
                case .table(let headers,let alignments):
                    ScrollView(.horizontal) {
                        Grid(alignment:.leading,horizontalSpacing:16,verticalSpacing:10) {
                            GridRow { ForEach(headers.indices,id:\.self) { column in inline(headers[column]).bold().gridColumnAlignment(alignment(column,alignments)) } }
                            Divider().gridCellUnsizedAxes(.horizontal)
                            ForEach(block.rows.indices,id:\.self) { row in
                                GridRow { ForEach(headers.indices,id:\.self) { column in inline(column<block.rows[row].count ? block.rows[row][column]:"") } }
                            }
                        }.padding(12)
                    }.background(.quaternary.opacity(0.3),in:RoundedRectangle(cornerRadius:8))
                }
            }
        }.font(StudioTypography.chat(pointOffset:pointOffset)).lineSpacing(4).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading)
        .onAppear{scheduleRendering()}
        .onChange(of:text){_,_ in scheduleRendering()}
        .onDisappear{renderingTask?.cancel();renderingTask=nil}
        .environment(\.openURL,OpenURLAction { url in
            guard ChatMarkdown.webURL(url) else{return .discarded}
            NSWorkspace.shared.open(url);return .handled
        })
    }
    func scheduleRendering(){
        let source=MessagePresentation.preview(text)
        guard source != pendingText || prepared.blocks.isEmpty else{return}
        pendingText=source
        guard renderingTask==nil else{return}
        renderingTask=Task { @MainActor in
            while !Task.isCancelled {
                let snapshot=pendingText
                let result=await Task.detached(priority:.userInitiated){PreparedMarkdown.make(snapshot)}.value
                guard !Task.isCancelled else{return};prepared=result
                if snapshot==pendingText{break}
                do{try await Task.sleep(nanoseconds:50_000_000)}catch{return}
            }
            renderingTask=nil
        }
    }
    func inline(_ text:String)->Text { Text(prepared.formatted[text] ?? AttributedString(text)) }
    func alignment(_ index:Int,_ values:[String])->HorizontalAlignment {
        guard index<values.count else{return .leading};return values[index]=="center" ? .center:values[index]=="right" ? .trailing:.leading
    }
}
