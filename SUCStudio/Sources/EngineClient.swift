import Foundation
import Security

final class LineReader {
    private var data = Data()
    private let lock = NSLock()
    func consume(_ bytes: Data, receive: (Data) -> Void) {
        lock.lock(); defer { lock.unlock() }
        data.append(bytes)
        while let end = data.firstIndex(of: 10) {
            let line = Data(data[..<end]); data.removeSubrange(...end)
            if line.count <= 4 * 1024 * 1024 { receive(line) }
        }
        if data.count > 4 * 1024 * 1024 { data.removeAll() }
    }
}
@MainActor final class EngineClient {
    private var pendingEvents: [[String: Any]] = []
    private var flushScheduled = false
    var process: Process?
    private var input: FileHandle?
    private var output: FileHandle?
    private var errors: FileHandle?
    private let writer = DispatchQueue(label: "studio.engine.writer")
    var onEvent: (([String: Any]) -> Void)?
    var onExit: ((Int32) -> Void)?
    var onLog: ((String) -> Void)?
    func start() throws {
        guard StudioPlatform.localInferenceAvailable else { throw NSError(domain: "SUCStudio", code: 3, userInfo: [NSLocalizedDescriptionKey: StudioPlatform.localUnavailableMessage]) }
        guard process == nil else { return }
        let helper = Bundle.main.bundleURL.appendingPathComponent("Contents/Helpers/suc-studio-engine")
        guard FileManager.default.isExecutableFile(atPath: helper.path) else { throw NSError(domain: "SUCStudio", code: 2, userInfo: [NSLocalizedDescriptionKey: "Helper assente dal bundle: ricostruire l’app con scripts/build.sh."]) }
        let p = Process(); p.executableURL = helper
        let stdin = Pipe(), stdout = Pipe(), stderr = Pipe()
        p.standardInput = stdin; p.standardOutput = stdout; p.standardError = stderr
        let lines = LineReader(), logs = LineReader()
        stdout.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let bytes = handle.availableData; if bytes.isEmpty { handle.readabilityHandler = nil; return }
            lines.consume(bytes) { line in
                guard let event = try? JSONSerialization.jsonObject(with: line) as? [String: Any] else { return }
                Task { @MainActor in self?.enqueue(event) }
            }
        }
        stderr.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let bytes = handle.availableData; if bytes.isEmpty { handle.readabilityHandler = nil; return }
            logs.consume(bytes) { line in
                if let text = String(data: line, encoding: .utf8) { Task { @MainActor in self?.onLog?(text) } }
            }
        }
        p.terminationHandler = { [weak self] process in Task { @MainActor in
            guard let self, self.process === process else { return }
            self.output?.readabilityHandler = nil; self.errors?.readabilityHandler = nil
            self.input = nil; self.process = nil; self.onExit?(process.terminationStatus)
        }}
        try p.run(); process = p; input = stdin.fileHandleForWriting; output = stdout.fileHandleForReading; errors = stderr.fileHandleForReading
    }
    private func enqueue(_ event: [String: Any]) {
        if event["event"] as? String == "delta", let last = pendingEvents.last, last["event"] as? String == "delta", last["id"] as? String == event["id"] as? String {
            var merged = last; let old = last["payload"] as? [String: Any] ?? [:]; let new = event["payload"] as? [String: Any] ?? [:]
            merged["payload"] = ["text": (old["text"] as? String ?? "") + (new["text"] as? String ?? "")]; pendingEvents[pendingEvents.count - 1] = merged
        } else { pendingEvents.append(event) }
        guard !flushScheduled else { return }; flushScheduled = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0 / 30.0) { [weak self] in
            guard let self else { return }; let events = self.pendingEvents; self.pendingEvents.removeAll(keepingCapacity: true); self.flushScheduled = false
            for event in events { self.onEvent?(event) }
        }
    }
    func send(_ value: [String: Any]) {
        guard let input, var data = try? JSONSerialization.data(withJSONObject: value), data.count <= 4 * 1024 * 1024 else { return }
        data.append(10)
        writer.async { try? input.write(contentsOf: data) }
    }
    func shutdown() {
        guard let p = process else { return }
        send(["command": "shutdown"])
        writer.async { [input] in try? input?.close() }
        DispatchQueue.global().asyncAfter(deadline: .now() + 6) { if p.isRunning { p.terminate() } }
    }
}
enum KeyStore {
    static let service = "local.sucstudio.universal.apikey"
    static func read() throws -> String {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service, kSecAttrAccount as String: "server", kSecReturnData as String: true, kSecMatchLimit as String: kSecMatchLimitOne]
        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecSuccess, let data = result as? Data, let key = String(data: data, encoding: .utf8) { return key }
        guard status == errSecItemNotFound else { throw failure(status) }
        return try rotate()
    }
    static func rotate() throws -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        let status = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        guard status == errSecSuccess else { throw failure(status) }
        let key = bytes.map { String(format: "%02x", $0) }.joined()
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service, kSecAttrAccount as String: "server"]
        let update = SecItemUpdate(query as CFDictionary, [kSecValueData as String: Data(key.utf8)] as CFDictionary)
        if update == errSecItemNotFound {
            var add = query; add[kSecValueData as String] = Data(key.utf8)
            let result = SecItemAdd(add as CFDictionary, nil); guard result == errSecSuccess else { throw failure(result) }
        } else if update != errSecSuccess { throw failure(update) }
        return key
    }
    static func failure(_ status: OSStatus) -> NSError { NSError(domain: "Keychain", code: Int(status), userInfo: [NSLocalizedDescriptionKey: "Keychain: \(SecCopyErrorMessageString(status, nil) as String? ?? String(status))"]) }
}
