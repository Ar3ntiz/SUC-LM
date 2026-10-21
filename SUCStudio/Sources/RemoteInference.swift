import Foundation

// Keep the byte loop off MainActor. Crossing to the UI once per byte makes a
// buffered response drain for seconds after the server has finished sending.
private struct RemoteResponseReader<Iterator: AsyncIteratorProtocol> where Iterator.Element == UInt8 {
    var iterator:Iterator
    private var total=0
    init(iterator:Iterator){self.iterator=iterator}
    mutating func nextLine() async throws -> Data? {
        var line=Data()
        while let byte=try await iterator.next() {
            try Task.checkCancellation();total+=1
            guard total<=16*1024*1024 else{throw MCPConnection.failure("Stream remoto oltre il limite di 16 MiB.")}
            if byte==10{return line}
            line.append(byte)
            guard line.count<=4*1024*1024 else{throw MCPConnection.failure("Evento remoto troppo grande.")}
        }
        return line.isEmpty ? nil:line
    }
    mutating func body() async throws -> Data {
        var data=Data()
        while let byte=try await iterator.next() {
            try Task.checkCancellation();data.append(byte)
            guard data.count<=4*1024*1024 else{throw MCPConnection.failure("Risposta remota troppo grande.")}
        }
        return data
    }
}

/// One client per generation; no retries of POST requests and no credential redirects.
@MainActor final class RemoteInference {
    static let generationInactivityLimit:TimeInterval=24*60*60
    static let generationResourceLimit:TimeInterval=7*24*60*60
    static let healthIntervalNanoseconds:UInt64=15_000_000_000
    private(set) var modelContextWindows: [String:Int] = [:]
    let configuration: RemoteConfiguration
    private let key: String
    private let http: URLSession
    private var stopped = false
    init(_ configuration: RemoteConfiguration, key: String) {
        self.configuration = configuration; self.key = key
        let settings = URLSessionConfiguration.ephemeral
        // Generation can legitimately remain silent through a long prefill. Keep only a day-scale network safety
        // bound; user cancellation and the independent health probe are the normal control path.
        settings.timeoutIntervalForRequest = Self.generationInactivityLimit; settings.timeoutIntervalForResource = Self.generationResourceLimit
        http = URLSession(configuration: settings, delegate: MCPHTTPDelegate(), delegateQueue: nil)
    }
    // Do not invalidate while a cancelled generation may still be entering bytes(for:).
    // Foundation otherwise raises an Objective-C exception instead of a Swift error.
    func stop() {
        guard !stopped else{return};stopped=true
        http.getAllTasks {tasks in tasks.forEach{$0.cancel()}}
    }
    deinit {http.invalidateAndCancel()}
    static func endpoint(_ base: String, path: String) throws -> URL {
        guard var url = URL(string: base.trimmingCharacters(in: .whitespacesAndNewlines)), ["http", "https"].contains(url.scheme ?? ""), url.host != nil, url.user == nil, url.password == nil, url.query == nil, url.fragment == nil else { throw MCPConnection.failure("URL del server non valido. Indica la base API, per esempio http://192.0.2.2:8080/v1.") }
        if url.path.isEmpty || url.path == "/" { url.appendPathComponent("v1") }
        return url.appendingPathComponent(path)
    }
    private func request(_ path: String) throws -> URLRequest {
        try Task.checkCancellation();guard !stopped else{throw CancellationError()}
        var request = URLRequest(url: try Self.endpoint(configuration.endpoint, path: path))
        if !key.isEmpty { guard !key.contains("\r"), !key.contains("\n") else { throw MCPConnection.failure("API key non valida.") }; request.setValue("Bearer " + key, forHTTPHeaderField: "Authorization") }
        return request
    }
    func models() async throws -> [String] {
        var request = try request("models"); request.timeoutInterval = 15
        let (bytes, response) = try await http.bytes(for: request); defer { bytes.task.cancel() }
        guard let response = response as? HTTPURLResponse, response.statusCode == 200 else { throw MCPConnection.failure("Elenco modelli: HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0).") }
        var data = Data()
        for try await byte in bytes { data.append(byte); guard data.count <= 1024 * 1024 else { throw MCPConnection.failure("Elenco modelli troppo grande.") } }
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any], let values = object["data"] as? [[String:Any]] else { throw MCPConnection.failure("Risposta /models non valida.") }
        modelContextWindows=[:]
        for item in values {
            guard let id=item["id"] as? String else{continue}
            for key in ["max_model_len","context_length","context_window","max_context_length"] {
                if let n=RemoteTelemetry.count(item[key]),(128...10_000_000).contains(n) {modelContextWindows[id]=n;break}
            }
        }
        return Array(Set(values.compactMap { $0["id"] as? String })).sorted()
    }
    func healthStatus() async -> [String:Any] {
        do {
            var probe=try request("health");probe.timeoutInterval=8
            let (_,response)=try await http.data(for:probe)
            var code=(response as? HTTPURLResponse)?.statusCode ?? 0
            if code==404 || code==405,let url=probe.url,var root=URLComponents(url:url,resolvingAgainstBaseURL:false),root.path != "/health" {
                root.path="/health"
                if let fallback=root.url {probe.url=fallback;let (_,response)=try await http.data(for:probe);code=(response as? HTTPURLResponse)?.statusCode ?? 0}
            }
            if (200...299).contains(code){return ["status":"ok","http_status":code]}
            if code==404 || code==405{return ["status":"unsupported","http_status":code]}
            return ["status":"error","http_status":code]
        } catch is CancellationError {return ["status":"cancelled"]}
        catch {return ["status":"unreachable","detail":String(error.localizedDescription.prefix(240))]}
    }
    func generate(_ input: [String: Any], event: @escaping ([String: Any]) -> Void) async throws {
        try Task.checkCancellation();guard !stopped else{throw CancellationError()}
        var body = input
        body["model"] = configuration.model; body["stream"] = true; body["stream_options"] = ["include_usage":true]
        if configuration.telemetryProfile=="llama",configuration.liveTelemetry != false {body["timings_per_token"]=true;body["return_progress"]=true}
        if configuration.telemetryProfile=="ds4",configuration.sendExtensions,let thinking=(body["chat_template_kwargs"] as? [String:Any])?["enable_thinking"] as? Bool {body["think"]=thinking}
        if !configuration.sendExtensions { body.removeValue(forKey:"top_k"); body.removeValue(forKey:"min_p"); body.removeValue(forKey:"chat_template_kwargs") }
        if (body["seed"] as? Int ?? -1) < 0 { body.removeValue(forKey:"seed") }
        var request = try request("chat/completions"); request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField:"Content-Type"); request.setValue("text/event-stream", forHTTPHeaderField:"Accept")
        request.httpBody = try JSONSerialization.data(withJSONObject:body)
        guard request.httpBody!.count <= 32 * 1024 * 1024 else { throw MCPConnection.failure("Richiesta remota troppo grande.") }
        let healthTask=Task { [weak self] in
            while !Task.isCancelled {
                do {try await Task.sleep(nanoseconds:Self.healthIntervalNanoseconds)} catch {return}
                guard !Task.isCancelled,let self else{return}
                event(["event":"health","payload":await self.healthStatus()])
            }
        }
        defer{healthTask.cancel()}
        let (bytes,response) = try await http.bytes(for:request); defer { bytes.task.cancel() }
        guard let response = response as? HTTPURLResponse, (200...299).contains(response.statusCode) else { throw MCPConnection.failure("Server remoto: HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0). Controlla endpoint, modello e API key. Nessun reinvio automatico.") }
        var telemetry=RemoteTelemetry(configuration:configuration)
        var reasoningBuffer="",lastMetrics=Date.distantPast
        var dataLines: [String] = [], buffer = "", lastFlush = Date(), finished = false, emitted = false
        func flush() { if !reasoningBuffer.isEmpty{event(["event":"reasoning","payload":["text":reasoningBuffer]]);reasoningBuffer="";lastFlush=Date();emitted=true};if !buffer.isEmpty { event(["event":"delta","payload":["text":buffer]]); buffer = ""; lastFlush = Date(); emitted = true } }
        func consume(_ data: Data, full: Bool = false) throws -> Bool {
            if String(decoding:data,as:UTF8.self).trimmingCharacters(in:.whitespacesAndNewlines) == "[DONE]" { flush();event(["event":"metrics","payload":telemetry.payload(final:true)]);return true }
            guard let value = try JSONSerialization.jsonObject(with:data) as? [String:Any] else { throw MCPConnection.failure("Risposta remota non valida.") }
            if let failure = value["error"] as? [String:Any] { throw MCPConnection.failure(failure["message"] as? String ?? "Errore del server remoto.") }
            if let choice = (value["choices"] as? [[String:Any]])?.first {
                let delta = choice[full ? "message" : "delta"] as? [String:Any] ?? [:]
                if let thought=(delta["reasoning_content"] as? String) ?? (delta["reasoning"] as? String){telemetry.observeText(thought);reasoningBuffer+=thought}
                if let text = delta["content"] as? String { telemetry.observeText(text);buffer += text }
                if let reason = choice["finish_reason"] as? String, !reason.isEmpty { finished = true; event(["event":"finish_reason","payload":["reason":reason]]) }
                if !emitted || Date().timeIntervalSince(lastFlush) >= 1.0/30 { flush() }
            }
            if let progress=value["prompt_progress"] as? [String:Any] {
                event(["event":"prefill_progress","payload":progress])
            }
            telemetry.ingest(value)
            if configuration.liveTelemetry != false && (finished || Date().timeIntervalSince(lastMetrics)>0.2) {
                flush();event(["event":"metrics","payload":telemetry.payload()]);lastMetrics=Date()
            }
            return full
        }
        event(["event":"started","payload":[:]])
        var reader=RemoteResponseReader(iterator:bytes.makeAsyncIterator())
        if !((response.value(forHTTPHeaderField:"Content-Type") ?? "").lowercased().contains("text/event-stream")) {
            _ = try consume(await reader.body(), full:true);flush();event(["event":"metrics","payload":telemetry.payload(final:true)]);return
        }
        while let line=try await reader.nextLine() {
            let value=String(decoding:line,as:UTF8.self).trimmingCharacters(in:.newlines)
            if value.isEmpty {
                if !dataLines.isEmpty {let data=Data(dataLines.joined(separator:"\n").utf8);dataLines.removeAll();if try consume(data){return}}
            } else if value.hasPrefix("data:") {
                var text=String(value.dropFirst(5));if text.first==" "{text.removeFirst()};dataLines.append(text)
            }
        }
        if !dataLines.isEmpty {if try consume(Data(dataLines.joined(separator:"\n").utf8)){return}}

        flush()
        if finished { event(["event":"metrics","payload":telemetry.payload(final:true)]) }
        guard finished else { throw MCPConnection.failure("Connessione remota interrotta prima della fine della risposta.") }
    }
}
