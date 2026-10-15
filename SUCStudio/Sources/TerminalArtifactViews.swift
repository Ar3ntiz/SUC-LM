import SwiftUI
struct TerminalSlot:View {
    @ObservedObject var terminal:InteractiveTerminal
    var body:some View {
        if terminal.awaitingInput {TerminalPanel(terminal:terminal,outputHeight:180)}
    }
}
struct TerminalPanel:View {
    @ObservedObject var terminal:InteractiveTerminal
    @State private var input=""
    @State private var secret=false
    @State private var follow=true
    var outputHeight:CGFloat=150
    var body:some View {
        VStack(alignment:.leading,spacing:8){
            HStack{
                Label(terminal.running ? "Input richiesto dal modello":"Comando del modello · exit \(terminal.exitStatus ?? 0)",systemImage:"terminal")
                Spacer()
                Button("Ctrl-C"){terminal.control(3)}.disabled(!terminal.running)
                Button("Ctrl-D"){terminal.control(4)}.disabled(!terminal.running)
                Button("Termina"){terminal.stop()}.disabled(!terminal.running)
            }.font(StudioTypography.ui(.caption))
            ScrollViewReader{reader in
                ScrollView{
                    Text(terminal.output.isEmpty ? "In attesa del terminale…":terminal.output).font(.system(size:12,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading)
                    Color.clear.frame(height:1).id("terminal-bottom")
                }.frame(height:outputHeight).onChange(of:terminal.output){_,_ in if follow{reader.scrollTo("terminal-bottom",anchor:.bottom)}}
                .onScrollPhaseChange{_,phase in if phase == .interacting || phase == .tracking {follow=false}}
                .overlay(alignment:.bottomTrailing){if !follow{Button("Fondo"){follow=true;reader.scrollTo("terminal-bottom",anchor:.bottom)}}}
            }.padding(8).background(Color.black.opacity(0.2),in:RoundedRectangle(cornerRadius:6))
            HStack{
                Group{if secret{SecureField("Password o input riservato",text:$input)}else{TextField("Comando o risposta",text:$input)}}.onSubmit{send()}
                Toggle("Riservato",isOn:$secret).toggleStyle(.checkbox)
                Button("Invia",action:send).disabled(!terminal.running)
            }.disabled(!terminal.running)
            if let error=terminal.inputError{Text(error).font(StudioTypography.ui(.caption)).foregroundStyle(.red)}
            Text("Questa scheda compare perché un comando del modello richiede interazione. Input/output non entrano nel contesto; l’input riservato viene mascherato.").font(StudioTypography.ui(.caption2)).foregroundStyle(.secondary)
        }.padding(14).background(Color(nsColor:.controlBackgroundColor),in:RoundedRectangle(cornerRadius:12)).overlay(RoundedRectangle(cornerRadius:12).stroke(.quaternary))
    }
    func send(){if terminal.send(input,secret:secret){input="";follow=true}}
}
struct ArtifactCards:View {
    @EnvironmentObject var store:StudioStore
    let artifacts:[ChatArtifact]
    var body:some View {
        VStack(alignment:.leading,spacing:8){
            Label("Artifact · \(artifacts.count)",systemImage:"doc.richtext").font(StudioTypography.ui(.headline))
            ForEach(artifacts){artifact in
                HStack{
                    Image(systemName:"doc")
                    VStack(alignment:.leading){Text(artifact.name).lineLimit(1);Text(ByteCountFormatter.string(fromByteCount:Int64(artifact.byteCount),countStyle:.file)).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)}
                    Spacer()
                    Button("Mostra nel Finder"){store.openArtifact(artifact)}
                    Button("Salva…"){store.saveArtifact(artifact)}
                }.padding(10).background(.quaternary,in:RoundedRectangle(cornerRadius:8))
            }
        }
    }
}
