import Foundation
import Security
import Darwin

struct MCPTool: Identifiable {
    var serverID: UUID
    var serverName: String
    var name: String
    var description: String
    var schema: String
    var id: String { serverID.uuidString + ":" + name }
}
enum MCPSecrets {
    static func read(_ id: UUID) throws -> [String: String] {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: "local.sucstudio.universal.mcp", kSecAttrAccount as String: id.uuidString, kSecReturnData as String: true]
        var result: CFTypeRef?; let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecItemNotFound { return [:] }
        guard status == errSecSuccess, let data = result as? Data else { throw MCPConnection.failure("Portachiavi MCP non accessibile (\(status)).") }
        return try JSONDecoder().decode([String: String].self, from: data)
    }
    static func write(_ value: [String: String], id: UUID) throws {
        let data = try JSONEncoder().encode(value)
        let key: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: "local.sucstudio.universal.mcp", kSecAttrAccount as String: id.uuidString]
        let status = SecItemUpdate(key as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if status == errSecItemNotFound { var add = key; add[kSecValueData as String] = data; guard SecItemAdd(add as CFDictionary, nil) == errSecSuccess else { throw MCPConnection.failure("Impossibile salvare le credenziali MCP.") }; return }
        guard status == errSecSuccess else { throw MCPConnection.failure("Impossibile aggiornare le credenziali MCP.") }
    }
    static func remove(_ id: UUID) {
        SecItemDelete([kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: "local.sucstudio.universal.mcp", kSecAttrAccount as String: id.uuidString] as CFDictionary)
    }
}
// Credentials must never follow a redirect to another endpoint.
final class MCPHTTPDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) { completionHandler(nil) }
}
@MainActor final class MCPConnection {
    nonisolated static func failure(_ text: String) -> NSError { NSError(domain: "SUCStudio.MCP", code: 1, userInfo: [NSLocalizedDescriptionKey: text]) }
    let configuration: MCPConfiguration
    private let credentials: [String: String]
    private var process: Process?
    private var stdin: FileHandle?, stdout: FileHandle?, stderr: FileHandle?
    private let writer = DispatchQueue(label: "studio.mcp.writer")
    private var pending: [String: CheckedContinuation<[String: Any], Error>] = [:]
    private var timeouts: [String: Task<Void, Never>] = [:]
    private var http: URLSession?
    private var sessionID: String?
    private var protocolVersion = "2025-11-25"
    private var stopped = false
    var onDisconnect: (() -> Void)?
    init(_ configuration: MCPConfiguration, credentials: [String: String]) { self.configuration = configuration; self.credentials = (configuration.transport=="http" ? configuration.headers ?? [:] : [:]).merging(credentials,uniquingKeysWith:{$1}) }
    func start() async throws -> [MCPTool] {
        if configuration.transport == "stdio" {
            guard configuration.executable.hasPrefix("/"), FileManager.default.isExecutableFile(atPath: configuration.executable) else { throw Self.failure("Scegli il percorso assoluto di un eseguibile MCP locale.") }
            let p = Process(), input = Pipe(), output = Pipe(), errors = Pipe(), lines = LineReader()
            p.executableURL = URL(fileURLWithPath: configuration.executable); p.arguments = configuration.arguments
            p.environment = ProcessInfo.processInfo.environment.merging(configuration.environment ?? [:],uniquingKeysWith:{$1}).merging(credentials, uniquingKeysWith: { _, new in new })
            p.standardInput = input; p.standardOutput = output; p.standardError = errors
            output.fileHandleForReading.readabilityHandler = { [weak self] handle in
                let bytes = handle.availableData
                if bytes.isEmpty { handle.readabilityHandler = nil; Task { @MainActor in self?.stop() }; return }
                lines.consume(bytes) { data in Task { @MainActor in self?.receive(data) } }
            }
            // Drain diagnostic output, but never persist it (may contain server secrets).
            errors.fileHandleForReading.readabilityHandler = { handle in if handle.availableData.isEmpty { handle.readabilityHandler = nil } }
            p.terminationHandler = { [weak self] _ in Task { @MainActor in self?.stop() } }
            try p.run(); process = p; stdin = input.fileHandleForWriting; stdout = output.fileHandleForReading; stderr = errors.fileHandleForReading
        } else if configuration.transport == "http" {
            guard let url = URL(string: configuration.endpoint), ["https", "http"].contains(url.scheme ?? ""), url.host != nil, url.user == nil, url.password == nil, url.fragment == nil else { throw Self.failure("Endpoint MCP HTTP/HTTPS non valido.") }
            let settings = URLSessionConfiguration.ephemeral; settings.timeoutIntervalForRequest = 30; settings.timeoutIntervalForResource = 3600
            http = URLSession(configuration: settings, delegate: MCPHTTPDelegate(), delegateQueue: nil)
        } else { throw Self.failure("Trasporto MCP non supportato.") }
        do {
            let result = try await request("initialize", ["protocolVersion": protocolVersion, "capabilities": [:], "clientInfo": ["name": StudioPlatform.name, "version": StudioPlatform.version]])
            guard let version = result["protocolVersion"] as? String, ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"].contains(version) else { throw Self.failure("Versione protocollo MCP non supportata.") }
            protocolVersion = version
            try await notify("notifications/initialized", [:])
            guard (result["capabilities"] as? [String: Any])?["tools"] != nil else { return [] }
            return try await listTools()
        } catch { stop(); throw error }
    }
    func listTools() async throws -> [MCPTool] {
        var tools: [MCPTool] = [], cursor: String?, seen = Set<String>()
        repeat {
            let result = try await request("tools/list", cursor.map { ["cursor": $0] } ?? [:])
            guard let values = result["tools"] as? [[String: Any]] else { throw Self.failure("tools/list non valido.") }
            for item in values {
                guard let name = item["name"] as? String, let schema = item["inputSchema"] as? [String: Any], !tools.contains(where: { $0.name == name }) else { throw Self.failure("Definizione strumento MCP non valida o duplicata.") }
                let data = try JSONSerialization.data(withJSONObject: schema, options: [.prettyPrinted, .sortedKeys])
                guard data.count <= 128 * 1024, name.utf8.count <= 256 else { throw Self.failure("Schema MCP troppo grande.") }
                tools.append(MCPTool(serverID: configuration.id, serverName: configuration.name, name: name, description: String((item["description"] as? String ?? "").prefix(8000)), schema: String(decoding: data, as: UTF8.self)))
            }
            guard tools.count <= 256 else { throw Self.failure("Massimo 256 strumenti per server.") }
            cursor = result["nextCursor"] as? String
            if let cursor { guard seen.insert(cursor).inserted, seen.count < 32 else { throw Self.failure("Paginazione MCP non valida.") } }
        } while cursor != nil
        return tools
    }
    func call(_ tool: MCPTool, arguments: [String: Any], timeout: Double = 30) async throws -> String {
        let result = try await request("tools/call", ["name": tool.name, "arguments": arguments],timeout:timeout)
        var text: [String] = [], unsupported = 0
        for item in result["content"] as? [[String: Any]] ?? [] {
            if item["type"] as? String == "text", let value = item["text"] as? String { text.append(value) }
            else if let resource = item["resource"] as? [String: Any], let value = resource["text"] as? String { text.append(value) }
            else { unsupported += 1 }
        }
        if let structured = result["structuredContent"], JSONSerialization.isValidJSONObject(structured) {
            text.append(String(decoding: try JSONSerialization.data(withJSONObject: structured, options: [.prettyPrinted, .sortedKeys]), as: UTF8.self))
        }
        if result["isError"] as? Bool == true { throw Self.failure("Errore dello strumento: " + text.joined(separator: "\n")) }
        guard !text.isEmpty else { throw Self.failure("Lo strumento non ha restituito testo utilizzabile dal modello.") }
        if unsupported > 0 { text.append("[\(unsupported) contenuti non testuali non inclusi.]") }
        let value = text.joined(separator: "\n\n")
        guard value.utf8.count <= AttachmentReader.maxTextBytes else { throw Self.failure("Risultato MCP oltre 256 KiB. Restringi la richiesta; nessun taglio automatico.") }
        return value
    }
    private func checked(_ value: [String: Any]) throws -> [String: Any] {
        if let error = value["error"] as? [String: Any] { throw Self.failure("MCP: " + (error["message"] as? String ?? "Errore JSON-RPC")) }
        guard let result = value["result"] as? [String: Any] else { throw Self.failure("Risposta JSON-RPC non valida.") }; return result
    }
    private func write(_ object: [String: Any]) throws {
        guard !stopped, let input = stdin else { throw Self.failure("Server MCP scollegato.") }
        var data = try JSONSerialization.data(withJSONObject: object); guard data.count <= 1024 * 1024 else { throw Self.failure("Richiesta MCP troppo grande.") }; data.append(10)
        writer.async { [weak self] in do { try input.write(contentsOf: data) } catch { Task { @MainActor in self?.stop() } } }
    }
    private func receive(_ data: Data) {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { stop(); return }
        if let method = object["method"] as? String {
            if let id = object["id"] {
                try? write(method == "ping" ? ["jsonrpc":"2.0", "id":id, "result":[:]] : ["jsonrpc":"2.0", "id":id, "error":["code":-32601,"message":"Client capability not supported"]])
            }; return
        }
        guard let id = object["id"] as? String, let continuation = pending.removeValue(forKey: id) else { return }
        timeouts.removeValue(forKey: id)?.cancel()
        do { continuation.resume(returning: try checked(object)) } catch { continuation.resume(throwing: error) }
    }
    private func request(_ method: String, _ params: [String: Any], timeout: Double = 30) async throws -> [String: Any] {
        guard !stopped else { throw Self.failure("Server MCP scollegato.") }
        let id = UUID().uuidString
        let wire: [String: Any] = ["jsonrpc":"2.0", "id":id, "method":method, "params":params]
        if configuration.transport == "http" { return try await httpRequest(wire, expectedID: id,timeout:timeout) }
        return try await withTaskCancellationHandler(operation:{
            try Task.checkCancellation()
            return try await withCheckedThrowingContinuation { continuation in
                pending[id] = continuation
                timeouts[id] = Task { [weak self] in
                    try? await Task.sleep(nanoseconds: UInt64(min(3600,max(0.1,timeout))*1_000_000_000))
                    guard !Task.isCancelled, let self, let waiting = self.pending.removeValue(forKey: id) else { return }
                    self.timeouts.removeValue(forKey: id)
                    try? self.write(["jsonrpc":"2.0", "method":"notifications/cancelled", "params":["requestId":id,"reason":"Timeout"]])
                    waiting.resume(throwing: Self.failure("Timeout MCP. L’esito di eventuali azioni può essere incerto; la chiamata non è stata ripetuta."))
                }
                do { try write(wire) } catch { pending.removeValue(forKey: id); timeouts.removeValue(forKey: id)?.cancel(); continuation.resume(throwing: error) }
            }
        },onCancel:{Task{@MainActor [weak self] in
            guard let self,let waiting=self.pending.removeValue(forKey:id) else{return}
            self.timeouts.removeValue(forKey:id)?.cancel()
            try? self.write(["jsonrpc":"2.0","method":"notifications/cancelled","params":["requestId":id,"reason":"Cancelled"]])
            waiting.resume(throwing:CancellationError())
        }})
    }

    private func notify(_ method: String, _ params: [String: Any]) async throws {
        let value: [String: Any] = ["jsonrpc":"2.0", "method":method, "params":params]
        if configuration.transport == "http" { _ = try await httpRequest(value, expectedID: nil) } else { try write(value) }
    }
    private func httpRequest(_ object: [String: Any], expectedID: String?, timeout: Double = 30) async throws -> [String: Any] {
        guard !stopped, let http, let endpoint = URL(string: configuration.endpoint) else { throw Self.failure("Server MCP scollegato.") }
        var request = URLRequest(url: endpoint);request.timeoutInterval=min(3600,max(0.1,timeout)); request.httpMethod = "POST"
        request.httpBody = try JSONSerialization.data(withJSONObject: object)
        guard (request.httpBody?.count ?? 0) <= 1024 * 1024 else { throw Self.failure("Richiesta MCP troppo grande.") }
        for (key,value) in credentials { guard !key.contains("\n"), !key.contains("\r"), !value.contains("\n"), !value.contains("\r") else { throw Self.failure("Header HTTP non valido.") }; request.setValue(value, forHTTPHeaderField: key) }
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json, text/event-stream", forHTTPHeaderField: "Accept")
        request.setValue(protocolVersion, forHTTPHeaderField: "MCP-Protocol-Version")
        if let sessionID { request.setValue(sessionID, forHTTPHeaderField: "MCP-Session-Id") }
        let (bytes,response) = try await http.bytes(for: request)
        defer { bytes.task.cancel() }
        guard let response = response as? HTTPURLResponse, (200...299).contains(response.statusCode) else { throw Self.failure("MCP HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0). Controlla endpoint/credenziali; riconnetti se la sessione è scaduta.") }
        if object["method"] as? String == "initialize", let value = response.value(forHTTPHeaderField: "MCP-Session-Id") {
            guard value.utf8.allSatisfy({ (0x21...0x7e).contains($0) }), value.utf8.count <= 1024 else { throw Self.failure("Sessione MCP non valida.") }; sessionID = value
        }
        guard let expectedID else { return [:] }
        if response.value(forHTTPHeaderField: "Content-Type")?.lowercased().contains("text/event-stream") == true {
            var buffer = Data(), eventLines: [String] = []; var count = 0
            for try await byte in bytes {
                count += 1; guard count <= 4 * 1024 * 1024 else { throw Self.failure("Risposta MCP troppo grande.") }
                if byte == 10 {
                    let line = String(decoding: buffer, as: UTF8.self).trimmingCharacters(in: .newlines); buffer.removeAll(keepingCapacity: true)
                    if line.isEmpty {
                        let data = Data(eventLines.joined(separator: "\n").utf8); eventLines.removeAll()
                        if !data.isEmpty, let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                            if value["id"] as? String == expectedID, value["method"] == nil { return try checked(value) }
                            if let method = value["method"] as? String, let id = value["id"] {
                                let reply: [String: Any] = method == "ping" ? ["jsonrpc":"2.0","id":id,"result":[:]] : ["jsonrpc":"2.0","id":id,"error":["code":-32601,"message":"Client capability not supported"]]
                                _ = try await httpRequest(reply, expectedID: nil)
                            }
                        }
                    } else if line.hasPrefix("data:") { var value = String(line.dropFirst(5)); if value.first == " " { value.removeFirst() }; eventLines.append(value) }
                } else { buffer.append(byte) }
            }
            throw Self.failure("Stream MCP interrotto prima della risposta; nessun tentativo automatico.")
        }
        var data = Data()
        for try await byte in bytes { data.append(byte); guard data.count <= 4 * 1024 * 1024 else { throw Self.failure("Risposta MCP troppo grande.") } }
        guard let value = try JSONSerialization.jsonObject(with: data) as? [String: Any], value["id"] as? String == expectedID else { throw Self.failure("ID risposta MCP non corrispondente.") }
        return try checked(value)
    }
    func stop() {
        guard !stopped else { return }; stopped = true
        http?.invalidateAndCancel(); http = nil
        stdout?.readabilityHandler = nil; stderr?.readabilityHandler = nil
        try? stdin?.close(); stdin = nil
        let waiters = pending; pending.removeAll(); timeouts.values.forEach { $0.cancel() }; timeouts.removeAll()
        for continuation in waiters.values { continuation.resume(throwing: Self.failure("Connessione MCP chiusa.")) }
        if let process, process.isRunning { process.terminate(); DispatchQueue.global().asyncAfter(deadline: .now()+2) { if process.isRunning { kill(process.processIdentifier, SIGKILL) } } }
        process = nil; onDisconnect?()
    }
}
