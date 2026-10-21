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
                Label(terminal.running ? L("Input richiesto dal modello"):L("Comando del modello · exit \(terminal.exitStatus ?? 0)"),systemImage:"terminal")
                Spacer()
                Button("Ctrl-C"){terminal.control(3)}.disabled(!terminal.running)
                Button("Ctrl-D"){terminal.control(4)}.disabled(!terminal.running)
                Button(L("Termina")){terminal.stop()}.disabled(!terminal.running)
            }.font(StudioTypography.ui(.caption))
            ScrollViewReader{reader in
                ScrollView{
                    Text(terminal.output.isEmpty ? L("In attesa del terminale…"):terminal.output).font(.system(size:12,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading)
                    Color.clear.frame(height:1).id("terminal-bottom")
                }.frame(height:outputHeight).onChange(of:terminal.output){_,_ in if follow{reader.scrollTo("terminal-bottom",anchor:.bottom)}}
                .onScrollPhaseChange{_,phase in if phase == .interacting || phase == .tracking {follow=false}}
                .overlay(alignment:.bottomTrailing){if !follow{Button(L("Fondo")){follow=true;reader.scrollTo("terminal-bottom",anchor:.bottom)}}}
            }.padding(8).background(Color.black.opacity(0.2),in:RoundedRectangle(cornerRadius:6))
            HStack{
                Group{if secret{SecureField(L("Password o input riservato"),text:$input)}else{TextField(L("Comando o risposta"),text:$input)}}.onSubmit{send()}
                Toggle(L("Riservato"),isOn:$secret).toggleStyle(.checkbox)
                Button(L("Invia"),action:send).disabled(!terminal.running)
            }.disabled(!terminal.running)
            if let error=terminal.inputError{Text(error).font(StudioTypography.ui(.caption)).foregroundStyle(.red)}
            Text(L("Questa scheda compare perché un comando del modello richiede interazione. Input/output non entrano nel contesto; l’input riservato viene mascherato.")).font(StudioTypography.ui(.caption2)).foregroundStyle(.secondary)
        }.padding(14).background(Color(nsColor:.controlBackgroundColor),in:RoundedRectangle(cornerRadius:12)).overlay(RoundedRectangle(cornerRadius:12).stroke(.quaternary))
    }
    func send(){if terminal.send(input,secret:secret){input="";follow=true}}
}
struct ArtifactSidebar:View {
    @EnvironmentObject var store:StudioStore
    @Binding var isPresented:Bool
    var body:some View {
        VStack(spacing:0) {
            HStack {
                Label(L("Artefatti"),systemImage:"shippingbox").font(StudioTypography.ui(.headline))
                Text("\(store.selected?.artifacts?.count ?? 0)").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                Spacer()
                ArtifactCloseButton(isPresented:$isPresented).frame(width:28,height:28)

            }.padding(16)
            Divider()
            if let artifacts=store.selected?.artifacts,!artifacts.isEmpty {
                ScrollView {ArtifactCards(artifacts:artifacts).padding(16)}
            } else {
                VStack(spacing:12) {
                    Image(systemName:"shippingbox").font(.system(size:28)).foregroundStyle(.secondary)
                    Text(L("Nessun artefatto in questa sessione")).font(StudioTypography.ui(.callout))
                    Text(L("I file pubblicati dal modello saranno disponibili qui.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary).multilineTextAlignment(.center)
                }.padding(24).frame(maxWidth:.infinity,maxHeight:.infinity)
            }
        }.frame(maxWidth:.infinity,maxHeight:.infinity,alignment:.top)
            .accessibilityIdentifier("chat-artifacts-panel")
    }
}
struct ArtifactCards:View {
    @EnvironmentObject var store:StudioStore
    let artifacts:[ChatArtifact]
    var body:some View {
        LazyVStack(alignment:.leading,spacing:12) {
            ForEach(artifacts) {artifact in
                VStack(alignment:.leading,spacing:10) {
                    HStack(alignment:.top,spacing:10) {
                        Image(systemName:"doc").foregroundStyle(.secondary)
                        VStack(alignment:.leading,spacing:4) {
                            Text(artifact.name).font(StudioTypography.ui(.callout)).lineLimit(3).textSelection(.enabled)
                            Text(ByteCountFormatter.string(fromByteCount:Int64(artifact.byteCount),countStyle:.file)).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                        }
                    }
                    HStack {
                        Button {store.openArtifact(artifact)} label:{Label("Finder",systemImage:"folder")}.help(L("Mostra nel Finder"))
                        Spacer()
                        Button {store.saveArtifact(artifact)} label:{Label(L("Salva…"),systemImage:"square.and.arrow.down")}
                    }.font(StudioTypography.ui(.caption))
                }.padding(12).frame(maxWidth:.infinity,alignment:.leading).background(.quaternary,in:RoundedRectangle(cornerRadius:10))
            }
        }
    }
}

/// An AppKit button gives the sidebar a concrete, full-size click target.
struct ArtifactCloseButton:NSViewRepresentable {
    @Binding var isPresented:Bool
    func makeCoordinator()->Coordinator {Coordinator($isPresented)}
    func makeNSView(context:Context)->NSButton {
        let button=NSButton(image:NSImage(systemSymbolName:"xmark",accessibilityDescription:L("Chiudi artefatti"))!,target:context.coordinator,action:#selector(Coordinator.close))
        button.isBordered=false;button.imagePosition = .imageOnly;button.focusRingType = .none
        button.toolTip=L("Chiudi artefatti");button.identifier=NSUserInterfaceItemIdentifier("close-artifacts")
        button.setAccessibilityIdentifier("close-artifacts");button.setAccessibilityLabel(L("Chiudi artefatti"))
        return button
    }
    func updateNSView(_ button:NSButton,context:Context) {context.coordinator.presentation=$isPresented;button.isEnabled=context.environment.isEnabled}
    final class Coordinator:NSObject {
        var presentation:Binding<Bool>
        init(_ presentation:Binding<Bool>){self.presentation=presentation}
        @objc func close(){presentation.wrappedValue=false}
    }
}
