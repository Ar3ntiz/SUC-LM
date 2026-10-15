import Foundation
import Combine

@MainActor final class InteractiveTerminal:ObservableObject,Identifiable {
    let id=UUID()
    @Published private(set) var output=""
    @Published private(set) var running=false
    @Published private(set) var awaitingInput=false
    @Published private(set) var exitStatus:Int32?
    @Published private(set) var inputError:String?
    private var rawOutput=Data()
    let command:String
    var onFinish:((Int32)->Void)?
    private var process:Process?
    private var input:FileHandle?
    private var waiter:CheckedContinuation<Int32,Never>?
    private var secrets:[String]=[]
    private let writeQueue=DispatchQueue(label:"sucstudio.terminal.input")
    init(command:String){self.command=command}
    deinit{if let process,process.isRunning{process.terminate()};try? input?.close()}
    func start(root:URL,helper:URL?=nil) throws {
        guard process==nil else{return}
        let p=Process(),stdin=Pipe(),stdout=Pipe()
        p.executableURL=helper ?? Bundle.main.bundleURL.appendingPathComponent("Contents/Helpers/suc-studio-terminal")
        p.arguments=[command];p.currentDirectoryURL=root;p.standardInput=stdin;p.standardOutput=stdout;p.standardError=stdout
        try p.run();process=p;input=stdin.fileHandleForWriting;running=true
        // run_interactive is the model's explicit request for user input. Delay presentation
        // so commands that finish on their own never flash a terminal card in the chat.
        Task { [weak self] in
            try? await Task.sleep(nanoseconds:250_000_000)
            guard !Task.isCancelled,self?.running==true else{return}
            self?.awaitingInput=true
        }
        // A single reader drains output before publishing the exit status.
        DispatchQueue.global(qos:.userInitiated).async { [weak self] in
            while true {
                let data=stdout.fileHandleForReading.availableData
                if data.isEmpty{break}
                // Backpressure bounds queued output even for commands such as yes.
                DispatchQueue.main.sync{self?.append(data)}
            }
            p.waitUntilExit();let status=p.terminationStatus
            DispatchQueue.main.async{self?.finish(status)}
        }
    }
    private func append(_ data:Data){
        rawOutput.append(data)
        if rawOutput.count>131072 {rawOutput.removeFirst(rawOutput.count-131072)}
        output=String(decoding:rawOutput,as:UTF8.self)
        // Line-oriented console: strip ANSI/OSC sequences, preserve ordinary output.
        output=output.replacingOccurrences(of:"\u{1B}\\][^\u{07}]*\u{07}",with:"",options:.regularExpression)
            .replacingOccurrences(of:"\u{1B}\\[[0-?]*[ -/]*[@-~]",with:"",options:.regularExpression)
        for secret in secrets where !secret.isEmpty {output=output.replacingOccurrences(of:secret,with:"••••")}
        if output.count>131072 {output=String(output.suffix(131072))}
    }
    @discardableResult func send(_ value:String,secret:Bool=false)->Bool {
        guard running,let handle=input else{return false}
        guard value.utf8.count<=4096 else{inputError="Input troppo lungo: massimo 4 KiB per invio.";return false}
        inputError=nil
        if secret && !value.isEmpty {secrets.append(value)}
        writeQueue.async { [weak self] in
            do{try handle.write(contentsOf:Data((value+"\n").utf8))}
            catch{DispatchQueue.main.async{self?.inputError="Invio al terminale non riuscito: "+error.localizedDescription}}
        }
        return true
    }
    func control(_ value:UInt8){guard running else{return};guard let handle=input else{return};writeQueue.async{try? handle.write(contentsOf:Data([value]))}}
    func stop(){guard running else{return};process?.terminate();try? input?.close();input=nil}
    private func finish(_ status:Int32){running=false;awaitingInput=false;exitStatus=status;try? input?.close();input=nil;waiter?.resume(returning:status);waiter=nil;rawOutput.removeAll();secrets.removeAll();onFinish?(status)}
    func wait() async -> Int32 {
        if let exitStatus{return exitStatus}
        return await withTaskCancellationHandler(operation:{await withCheckedContinuation{waiter=$0}},onCancel:{Task{@MainActor in self.stop()}})
    }
}
extension StudioStore {
    func startTerminal(session:ChatSession,command:String) throws -> InteractiveTerminal {
        guard terminals[session.id]?.running != true else{throw MCPConnection.failure("Un terminale è già attivo in questa chat.")}
        guard !command.isEmpty,command.utf8.count<=32768 else{throw MCPConnection.failure("Comando assente o troppo grande.")}
        let terminal=InteractiveTerminal(command:command)
        try terminal.start(root:workspaceURL(session));terminals[session.id]=terminal
        operationJournal.update(id:terminal.id.uuidString,title:"Terminale interattivo",state:"In corso",sessionID:session.id)
        let terminalID=terminal.id.uuidString
        terminal.onFinish={ [weak self] status in self?.operationJournal.update(id:terminalID,state:status==0 ? "Completata":(status>=128 ? "Interrotta":"Errore")) }
        return terminal
    }
    func runInteractive(_ action:WorkspaceAction,session:ChatSession) async throws -> String {
        guard session.allowCommands==true else{throw MCPConnection.failure("Comandi shell disabilitati per questa sessione.")}
        let terminal=try startTerminal(session:session,command:action.arguments["command"] ?? "")
        let status=await terminal.wait();try Task.checkCancellation()
        guard status==0 else{throw MCPConnection.failure("Terminale inline terminato con exit status \(status). Controlla la scheda nella chat.")}
        return "Terminale interattivo inline terminato. Exit status: \(status). Output e input riservati rimangono nella scheda della chat e non sono inclusi nel contesto. Per produrre file scaricabili usa publish_artifact(path)."
    }
}
