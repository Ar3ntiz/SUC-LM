import Foundation
import Darwin

enum WorkspaceTools {
    static let names = ["list_files", "read_file", "write_file", "make_directory", "run_command"]
    static func resolve(_ path: String, root: URL) throws -> URL {
        let base = root.standardizedFileURL.resolvingSymlinksInPath()
        guard !path.hasPrefix("/"), !path.contains("\0") else { throw MCPConnection.failure("Usa un percorso relativo alla directory di lavoro.") }
        let url = base.appendingPathComponent(path.isEmpty ? "." : path).standardizedFileURL.resolvingSymlinksInPath()
        guard url.path == base.path || url.path.hasPrefix(base.path + "/") else { throw MCPConnection.failure("Il percorso esce dalla directory di lavoro (anche tramite symlink).") }
        return url
    }
    static func execute(_ action: WorkspaceAction, root: URL, allowCommands: Bool) async throws -> String {
        guard names.contains(action.name) else { throw MCPConnection.failure("Strumento sconosciuto: \(action.name)") }
        if action.name == "run_command" {
            guard allowCommands else { throw MCPConnection.failure("Comandi shell disabilitati per questa sessione.") }
            guard let command = action.arguments["command"], !command.isEmpty, command.utf8.count <= 32768 else { throw MCPConnection.failure("Comando assente o troppo grande.") }
            return try await WorkspaceCommand().run(command, root:root)
        }
        try Task.checkCancellation()
        let url = try resolve(action.arguments["path"] ?? ".",root:root)
        return try await Task.detached {
            try Task.checkCancellation()
            switch action.name {
            case "list_files":
                let names = try FileManager.default.contentsOfDirectory(atPath:url.path).sorted()
                guard names.count <= 2000 else { throw MCPConnection.failure("Troppi elementi: seleziona una sottodirectory.") }
                return names.joined(separator:"\n")
            case "read_file":
                let values = try url.resourceValues(forKeys:[.isRegularFileKey,.fileSizeKey])
                guard values.isRegularFile == true, (values.fileSize ?? Int.max) <= 256 * 1024 else { throw MCPConnection.failure("Lettura consentita solo per file regolari fino a 256 KiB.") }
                return try AttachmentReader.decodeText(Data(contentsOf:url))
            case "write_file":
                guard let content = action.arguments["content"], content.utf8.count <= 128 * 1024, url.path != root.standardizedFileURL.resolvingSymlinksInPath().path else { throw MCPConnection.failure("Contenuto assente, troppo grande o percorso non valido.") }
                try Data(content.utf8).write(to:url,options:.atomic); return "File scritto: \(action.arguments["path"] ?? "") · \(content.utf8.count) byte"
            case "make_directory":
                try FileManager.default.createDirectory(at:url,withIntermediateDirectories:true); return "Directory creata: \(action.arguments["path"] ?? "")"
            default: throw MCPConnection.failure("Azione non supportata.")
            }
        }.value
    }
}
struct CommandTimeLimitError:LocalizedError {
    let seconds:Int
    var errorDescription:String? {"Durata massima richiesta dal modello raggiunta (\(seconds) secondi). Nessun riavvio automatico."}
}
final class WorkspaceCommand: @unchecked Sendable {
    private let helperURL: URL
    init(helperURL: URL? = nil) { self.helperURL = helperURL ?? Bundle.main.bundleURL.appendingPathComponent("Contents/Helpers/suc-studio-command") }
    private let lock=NSLock()
    private var process:Process?
    private var cancelled=false
    private func cancel() {
        lock.lock();cancelled=true;let p=process;lock.unlock()
        if let p,p.isRunning {kill(-p.processIdentifier,SIGTERM)}
    }
    func run(_ command:String,root:URL,environment:[String:String]=[:],timeout:TimeInterval?=nil,onOutput:((Data)->Void)?=nil) async throws -> String {
        if let timeout, !timeout.isFinite || timeout<=0 {throw MCPConnection.failure("Durata massima del comando non valida.")}
        return try await withTaskCancellationHandler(operation:{
            try await withCheckedThrowingContinuation { continuation in
                DispatchQueue.global(qos:.userInitiated).async {
                    do{continuation.resume(returning:try self.execute(command,root:root,environment:environment,timeout:timeout,onOutput:onOutput))}
                    catch{continuation.resume(throwing:error)}
                }
            }
        },onCancel:{self.cancel()})
    }
    private func execute(_ command:String,root:URL,environment:[String:String],timeout:TimeInterval?,onOutput:((Data)->Void)?) throws -> String {
        let p=Process(),pipe=Pipe()
        p.executableURL=helperURL;p.arguments=[command];p.currentDirectoryURL=root
        p.environment=ProcessInfo.processInfo.environment.merging(environment,uniquingKeysWith:{_,new in new})
        p.standardInput=FileHandle.nullDevice;p.standardOutput=pipe;p.standardError=pipe
        let fd=pipe.fileHandleForReading.fileDescriptor
        guard fcntl(fd,F_SETFL,fcntl(fd,F_GETFL)|O_NONBLOCK)>=0 else{throw POSIXError(.EIO)}
        defer {
            if p.processIdentifier>0 {kill(-p.processIdentifier,SIGKILL)}
            try? pipe.fileHandleForReading.close();try? pipe.fileHandleForWriting.close()
            lock.lock();process=nil;lock.unlock()
        }
        lock.lock();let stopped=cancelled;lock.unlock();if stopped{throw CancellationError()}
        try p.run();try? pipe.fileHandleForWriting.close()
        lock.lock();process=p;lock.unlock()
        let start=ProcessInfo.processInfo.systemUptime
        var output=Data(),buffer=[UInt8](repeating:0,count:16384)
        func drain() throws {
            // Bounded reads keep cancellation responsive even for an endless output producer.
            for _ in 0..<16 {
                let count=Darwin.read(fd,&buffer,16384)
                if count>0 {
                    let data=Data(buffer.prefix(count))
                    if let onOutput {onOutput(data)}
                    else {guard output.count+count<=256*1024 else{throw MCPConnection.failure("Output oltre 256 KiB.")};output.append(data)}
                } else if count==0 || errno==EAGAIN || errno==EWOULDBLOCK {return}
                else if errno != EINTR {throw POSIXError(POSIXErrorCode(rawValue:errno) ?? .EIO)}
            }
        }
        do {
            while true {
                lock.lock();let stop=cancelled;lock.unlock()
                if stop{throw CancellationError()}
                if let timeout,p.isRunning,ProcessInfo.processInfo.systemUptime-start>=timeout{throw CommandTimeLimitError(seconds:Int(timeout))}
                try drain()
                if !p.isRunning {
                    // Shell-backgrounded descendants are not independent harness jobs.
                    kill(-p.processIdentifier,SIGKILL)
                    try drain();break
                }
                Thread.sleep(forTimeInterval:0.02)
            }
        } catch {
            kill(-p.processIdentifier,SIGTERM);Thread.sleep(forTimeInterval:0.1)
            kill(-p.processIdentifier,SIGKILL);if p.isRunning{kill(p.processIdentifier,SIGKILL)}
            p.waitUntilExit();throw error
        }
        lock.lock();let wasCancelled=cancelled;lock.unlock();if wasCancelled{throw CancellationError()}
        return "Exit status: \(p.terminationStatus)\n"+String(decoding:output,as:UTF8.self)
    }
}
