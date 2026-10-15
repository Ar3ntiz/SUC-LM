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
final class WorkspaceCommand: @unchecked Sendable {
    private let helperURL: URL
    init(helperURL: URL? = nil) { self.helperURL = helperURL ?? Bundle.main.bundleURL.appendingPathComponent("Contents/Helpers/suc-studio-command") }
    private let lock = NSLock()
    private var process: Process?
    private var cancelled = false
    private var output = Data()
    private var overflow = false
    private func cancel() { lock.lock(); cancelled = true; let p=process; lock.unlock(); if let p, p.isRunning { kill(-p.processIdentifier, SIGTERM) } }
    func run(_ command: String, root: URL, environment: [String:String] = [:]) async throws -> String {
        try await withTaskCancellationHandler(operation: {
            try await withCheckedThrowingContinuation { continuation in
                DispatchQueue.global(qos:.userInitiated).async {
                    do { continuation.resume(returning: try self.execute(command,root:root,environment:environment)) } catch { continuation.resume(throwing:error) }
                }
            }
        },onCancel:{ self.cancel() })
    }
    private func execute(_ command: String, root: URL, environment: [String:String]) throws -> String {
        let p=Process(), pipe=Pipe()
        p.executableURL=helperURL;p.arguments=[command];p.currentDirectoryURL=root
        p.environment=ProcessInfo.processInfo.environment.merging(environment,uniquingKeysWith:{_,new in new})
        p.standardInput=FileHandle.nullDevice;p.standardOutput=pipe;p.standardError=pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data=handle.availableData
            guard let self else{return};self.lock.lock();defer{self.lock.unlock()}
            if self.output.count + data.count > 256 * 1024 { self.overflow=true } else { self.output.append(data) }
        }
        defer { if p.processIdentifier > 0 { kill(-p.processIdentifier, SIGKILL) }; pipe.fileHandleForReading.readabilityHandler=nil;try? pipe.fileHandleForReading.close();lock.lock();process=nil;lock.unlock() }
        lock.lock();let stopped=cancelled;lock.unlock();if stopped{throw CancellationError()}
        try p.run();lock.lock();process=p;lock.unlock()
        let deadline=Date().addingTimeInterval(30)
        while p.isRunning {
            lock.lock();let stop=cancelled || overflow;lock.unlock()
            if stop || Date() > deadline { kill(-p.processIdentifier,SIGTERM);Thread.sleep(forTimeInterval:0.1);kill(-p.processIdentifier,SIGKILL);if p.isRunning{kill(p.processIdentifier,SIGKILL)};p.waitUntilExit();throw MCPConnection.failure(stop ? "Comando annullato o output oltre 256 KiB." : "Comando oltre 30 secondi.") }
            Thread.sleep(forTimeInterval:0.02)
        }
        // Allow the pipe reader to consume the final buffered bytes without waiting on descendants.
        Thread.sleep(forTimeInterval:0.05)
        lock.lock();let data=output, tooLarge=overflow, wasCancelled=cancelled;lock.unlock()
        if tooLarge { throw MCPConnection.failure("Output oltre 256 KiB.") }; if wasCancelled { throw CancellationError() }
        return "Exit status: \(p.terminationStatus)\n" + String(decoding:data,as:UTF8.self)
    }
}
