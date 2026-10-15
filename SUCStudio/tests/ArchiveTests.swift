import Foundation
@main struct ArchiveTests {
    static func main() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("sessions.json"), disk = ArchiveDisk(url: root.appendingPathComponent("sessions.json"))
        var a = Archive(); var session = ChatSession(); session.draft = "bozza 🇮🇹"; session.messages = [ChatMessage(role: "assistant", text: "parziale", state: "generating")]; a.sessions = [session]; a.selected = session.id
        try await disk.save(a)
        let restored = try ArchiveDisk.load(url)
        precondition(restored.sessions[0].draft == "bozza 🇮🇹")
        precondition(restored.sessions[0].messages[0].state == "interrupted")
        a.sessions[0].title = "Aggiornata"; try await disk.save(a)
        let backup = try ArchiveDisk.load(url.appendingPathExtension("backup")); precondition(backup.sessions[0].title == "Nuova chat")
        try Data("broken".utf8).write(to: url)
        do { _ = try ArchiveDisk.load(url); preconditionFailure("Corruption accepted") } catch {}
        try await disk.save(a)
        let preserved = try ArchiveDisk.load(url.appendingPathExtension("backup")); precondition(preserved.sessions[0].title == "Nuova chat")
        let export = SessionExport(session: session); let roundtrip = try JSONDecoder().decode(SessionExport.self, from: JSONEncoder().encode(export)); precondition(roundtrip.session.messages[0].text == "parziale")
        var m = Metrics(); precondition(m.decodeRate == nil); m.completionTokens = 1; m.decodeSeconds = 2; precondition(m.decodeRate == nil); m.completionTokens = 5; precondition(m.decodeRate == 2)
        // Simulate an actual v0.1 archive with no memoryMode key.
        var legacy = try JSONSerialization.jsonObject(with: JSONEncoder().encode(AppSettings())) as! [String: Any]
        legacy.removeValue(forKey: "memoryMode")
        let oldAuto = try JSONDecoder().decode(AppSettings.self, from: JSONSerialization.data(withJSONObject: legacy))
        precondition(oldAuto.effectiveMemoryMode == "auto")
        legacy["expertBudget"] = "8G"
        let oldManual = try JSONDecoder().decode(AppSettings.self, from: JSONSerialization.data(withJSONObject: legacy))
        precondition(oldManual.effectiveMemoryMode == "manual" && oldManual.expertBudget == "8G")
        var explicit = oldManual; explicit.memoryMode = "resident"
        let persisted = try JSONDecoder().decode(AppSettings.self, from: JSONEncoder().encode(explicit))
        precondition(persisted.effectiveMemoryMode == "resident")
        print("Memory settings migration: v0.1 automatic/manual and v0.2 persistence passed")
        print("ArchiveTests: persistence, recovery, backup, export and metrics passed")
    }
}
