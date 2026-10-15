import SwiftUI
import AppKit
@main struct RenderPreview {
    @MainActor static func main() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = StudioStore(storageURL: root.appendingPathComponent("preview.json"))
        store.loadedModel = "Qwen3 · locale"; store.modelState = "ready"
        store.mutateSelected {
            $0.title = "Idee, spiegate bene"
            $0.messages = [ChatMessage(role: "user", text: "Spiegami la differenza tra prefill e decode."), ChatMessage(role: "assistant", text: "## Due fasi, un’unica risposta\n\nIl **prefill** legge il contesto della conversazione. Il **decode** genera i nuovi token, uno dopo l’altro.\n\n- Prefill: elaborazione del prompt\n- Decode: generazione della risposta\n\n```python\nclient.chat.completions.create(\n    model=\"model.suc1\",\n    stream=True\n)\n```", model: "Qwen3 · locale", metrics: Metrics(promptTokens: 512, completionTokens: 128, prefillSeconds: 2, decodeSeconds: 3.5))]
        }
        if CommandLine.arguments.contains("attachments") {
            store.mutateSelected {
                $0.draft = "Confronta questi documenti e riassumi le differenze."
                $0.draftAttachments = [ChatAttachment(name: "Relazione.docx", text: "Sintesi della relazione", originalBytes: 24000), ChatAttachment(name: "Note.odt", text: "Note del progetto", originalBytes: 8500)]
                $0.messages[0].attachments = [ChatAttachment(name: "Appunti.estensione-personale", text: "Prefill e decode", originalBytes: 200)]
            }
        }
        if CommandLine.arguments.contains("markdown") { store.mutateSelected { $0.messages = [ChatMessage(role:"assistant",text:"## Markdown leggibile\n\n**Grassetto**, *corsivo* e [un link](https://example.com).\n\n1. Primo elemento\n2. Secondo elemento\n- [x] Verificato\n\n> Una citazione leggibile.\n\n| Fase | Token/s |\n| --- | ---: |\n| Prefill | 142.5 |\n| Decode | 31.8 |\n\n```swift\nlet message = \"Ciao\"\n```") ] } }
        if CommandLine.arguments.contains("activity") {
            store.modelState="loading";store.archive.settings.modelPath="/Models/Qwen3.suc1"
            store.mutateSelected { $0.messages=[ChatMessage(role:"user",text:"Leggi README.md e riassumi il progetto."),ChatMessage(role:"assistant",text:"",state:"prefill")] }
            store.approvals=[ActionApproval(sessionID:store.selected!.id,title:store.selected!.title,directory:"/Users/demo/Progetto",action:WorkspaceAction(name:"read_file",arguments:["path":"README.md"]))]
        }
        if CommandLine.arguments.contains("report") {
            store.mutateSelected {
                $0.agentEnabled=true;$0.workspacePath="/Users/demo/Progetto"
                $0.messages=[ChatMessage(role:"user",text:"Leggi il progetto e prepara una nota."),ChatMessage(role:"assistant",text:"{\"suc_action\":{\"name\":\"read_file\",\"arguments\":{\"path\":\"README.md\"}}}",kind:"tool_call"),ChatMessage(role:"user",text:"Risultato tecnico interno",kind:"tool_result"),ChatMessage(role:"assistant",text:"Ho letto il progetto e salvato la nota in **nota.md**. Puoi trovare i dettagli nel registro della sessione.")]
                $0.actions=[ActionRecord(name:"read_file",details:"README.md",state:"completed",result:"testo"),ActionRecord(name:"write_file",details:"nota.md",state:"completed",result:"salvato")]
            }
            store.agentTurnStarts[store.selected!.id]=0;store.finishAgentReport(store.selected!.id)
        }
        if CommandLine.arguments.contains("operationlog") {
            store.archive.settings.modelIOLogEnabled=true
            store.modelIOLog.append(requestID:"req-01234567",direction:"INPUT",text:"{\n  \"messages\": [{\"role\": \"user\", \"content\": \"Analizza il progetto e prepara un report.\"}],\n  \"temperature\": 0.7\n}")
            store.modelIOLog.append(requestID:"req-01234567",direction:"REASONING",text:"Esamino la struttura e raccolgo i dati necessari.")
            store.modelIOLog.append(requestID:"req-01234567",direction:"OUTPUT",text:"Ho analizzato il progetto. Il report è pronto e contiene risultati, limiti e prossimi passi.")
        }
        if CommandLine.arguments.contains("terminal") {
            let terminal=InteractiveTerminal(command:"printf 'Preparazione report…\\nConfermare? [s/n]\\n'; read answer")
            try terminal.start(root:FileManager.default.temporaryDirectory,helper:URL(fileURLWithPath:ProcessInfo.processInfo.environment["SUC_STUDIO_TERMINAL_HELPER"]!))
            store.terminals[store.selected!.id]=terminal
            store.mutateSelected{$0.workspacePath="/Users/demo/Progetto";$0.artifacts=[ChatArtifact(name:"report.pdf",byteCount:24576)]}
        }
        if CommandLine.arguments.contains("thinking") {store.mutateSelected{$0.messages=[ChatMessage(role:"user",text:"Confronta queste due soluzioni."),ChatMessage(reasoning:"Valuto i vincoli e confronto i risultati.\n\nLa seconda soluzione richiede meno memoria e conserva il comportamento atteso.",role:"assistant",text:"## La soluzione consigliata\n\nUsa la **seconda soluzione**: richiede meno memoria e mantiene le stesse funzionalità.")]}}
        if CommandLine.arguments.contains("longhtml") {store.mutateSelected{$0.messages=[ChatMessage(role:"user",text:"<!doctype html><html>"+String(repeating:"<div><a href=\"https://example.com/test\">test</a></div>\n",count:5000)+"</html>"),ChatMessage(role:"assistant",text:"Ho ricevuto il documento completo. Puoi aprire il testo integrale dall’anteprima.")]}}
        let width = CommandLine.arguments.count > 2 ? Double(CommandLine.arguments[2]) ?? 1180 : 1180
        let height = CommandLine.arguments.count > 3 ? Double(CommandLine.arguments[3]) ?? 780 : 780
        store.tab = CommandLine.arguments.count > 4 ? CommandLine.arguments[4] : "chat"
        store.archive.settings.sectionSidebarCollapsed = !CommandLine.arguments.contains("expanded")
        if CommandLine.arguments.contains("remote") { store.archive.settings.chatBackend = "remote"; store.archive.settings.remote = RemoteConfiguration(endpoint: "http://192.168.1.20:8080/v1", model: "Qwen3-8B") }
        if CommandLine.arguments.contains("largefonts") { store.archive.settings.uiFont="Helvetica Neue";store.archive.settings.uiFontSize=16;store.archive.settings.chatFont="Georgia";store.archive.settings.chatFontSize=22;StudioTypography.apply(store.archive.settings) }
        let dark = CommandLine.arguments.count > 5 && CommandLine.arguments[5] == "dark"
        store.archive.settings.appearance = dark ? "dark" : "light"
        _ = NSApplication.shared
        NSApplication.shared.setActivationPolicy(.prohibited)
        let content = CommandLine.arguments.contains("operationlog") ? AnyView(OperationLogView(initiallyExpanded:true).padding(24).frame(maxWidth:.infinity,maxHeight:.infinity,alignment:.top)):CommandLine.arguments.contains("hooks") ? AnyView(HooksSettingsView()):AnyView(StudioRoot())
        let hosting = NSHostingView(rootView: content.environmentObject(store).background(Color(nsColor:.windowBackgroundColor)))
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: width, height: height), styleMask: [.borderless], backing: .buffered, defer: false)
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        window.contentView = hosting
        hosting.frame = NSRect(x: 0, y: 0, width: width, height: height)
        hosting.layoutSubtreeIfNeeded()
        RunLoop.current.run(until: Date().addingTimeInterval(1.5))
        hosting.layoutSubtreeIfNeeded()
        guard let bitmap = hosting.bitmapImageRepForCachingDisplay(in: hosting.bounds) else { throw NSError(domain: "Preview", code: 1) }
        hosting.cacheDisplay(in: hosting.bounds, to: bitmap)
        guard let png = bitmap.representation(using: .png, properties: [:]) else { throw NSError(domain: "Preview", code: 2) }
        try png.write(to: URL(fileURLWithPath: CommandLine.arguments[1]));for terminal in store.terminals.values{terminal.stop()}
    }
}
