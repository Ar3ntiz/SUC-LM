import Foundation

@main struct CustomizationTests {
    @MainActor static func main() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = StudioStore(storageURL: root.appendingPathComponent("archive.json"))
        store.archive.settings.globalSystemPrompt = "GLOBAL"
        store.archive.settings.skills = [StudioSkill(name: "On", description: "DESCRIPTION", instructions: "ACTIVE", enabled: true), StudioSkill(name: "Off", instructions: "EXCLUDED", enabled: false)]
        store.archive.settings.sectionSidebarCollapsed = false
        store.mutateSelected { $0.systemPrompt = "EXTRA"; $0.draft = "hello" }
        let prompt = PromptComposer.compose(settings: store.archive.settings, session: store.selected!)
        precondition(prompt.contains("GLOBAL") && prompt.contains("DESCRIPTION") && !prompt.contains("ACTIVE") && prompt.hasSuffix("EXTRA\n\n"+AgentProtocol.webOnlyInstructions) && !prompt.contains("EXCLUDED")) // web tools are offered by default (1.4.5_01)
        precondition(prompt.range(of: "GLOBAL")!.lowerBound < prompt.range(of: "DESCRIPTION")!.lowerBound)
        store.modelState = "ready"; store.loadedModel = "fixture"; store.sendMessage()
        precondition(store.selected!.messages.last!.generation?.systemPrompt == prompt)
        store.receive(["event":"finished", "id":store.requests.keys.first!, "payload":[:]])
        try await Task.sleep(nanoseconds: 600_000_000)
        let disk = try ArchiveDisk.load(root.appendingPathComponent("archive.json"))
        precondition(disk.settings.skills?.count == 2 && disk.settings.globalSystemPrompt == "GLOBAL" && disk.settings.sectionSidebarCollapsed == false)
        var old = try JSONSerialization.jsonObject(with: JSONEncoder().encode(disk.settings)) as! [String: Any]
        for key in ["globalSystemPrompt", "skills", "mcpServers", "sectionSidebarCollapsed"] { old.removeValue(forKey: key) }
        let migrated = try JSONDecoder().decode(AppSettings.self, from: JSONSerialization.data(withJSONObject: old))
        precondition(migrated.globalSystemPrompt == nil && migrated.skills == nil && migrated.mcpServers == nil)
        print("CustomizationTests: prompt composition, actual generation snapshot, persistence and migration passed")
        let fixture = CommandLine.arguments[1]
        var local = MCPConfiguration(name: "Fixture", executable: "/usr/bin/python3", arguments: [fixture], enabled: true)
        let client = MCPConnection(local, credentials: [:]); defer { client.stop() }
        let tools = try await client.start(); precondition(tools.map(\.name) == ["echo", "fail"])
        let echo = try await client.call(tools[0], arguments: ["text":"stdio works"])
        precondition(echo.contains("stdio works") && echo.contains("true"))
        do { _ = try await client.call(tools[1], arguments: [:]); preconditionFailure("Expected tool error") } catch { precondition(error.localizedDescription.contains("expected failure")) }
        store.archive.settings.mcpServers = [local]; store.mcpConnections[local.id] = client; store.mcpTools = tools
        let origin = store.selected!.id
        let run = Task { try await store.runMCP(tools[0], argumentsJSON: "{\"text\":\"origin result\",\"delay\":true}", sessionID: origin) }
        try await Task.sleep(nanoseconds: 50_000_000)
        store.newSession(); let other = store.selected!.id; precondition(other != origin)
        try await run.value
        precondition(store.archive.sessions.first { $0.id == origin }!.draftAttachments!.first!.text.contains("origin result"))
        precondition(store.selected!.draftAttachments?.isEmpty != false && store.mcpRuns.isEmpty)
        client.stop()
        do { _ = try await client.call(tools[0], arguments: [:]); preconditionFailure("Expected disconnect") } catch {}
        print("MCPTests: stdio initialize, pagination, call, tool errors, disconnect and session routing passed")
        let server = Process(), output = Pipe()
        server.executableURL = URL(fileURLWithPath: "/usr/bin/python3"); server.arguments = [fixture, "--http"]; server.standardOutput = output
        try server.run(); defer { server.terminate() }
        let data = output.fileHandleForReading.availableData
        guard let port = Int(String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)) else { throw MCPConnection.failure("Fixture HTTP non avviata: verifica i permessi di ascolto localhost dell’ambiente di test.") }
        local = MCPConfiguration(name: "HTTP Fixture", transport: "http", endpoint: "http://127.0.0.1:\(port)/mcp", enabled: true)
        let remote = MCPConnection(local, credentials: ["Authorization":"Bearer fixture"]); defer { remote.stop() }
        let remoteTools = try await remote.start(); precondition(remoteTools.count == 2)
        let result = try await remote.call(remoteTools[0], arguments: ["text":"SSE works"])
        precondition(result.contains("SSE works"))
        do { _ = try await remote.call(remoteTools[1], arguments: [:]); preconditionFailure("Expected SSE tool error") } catch { precondition(error.localizedDescription.contains("expected failure")) }
        let unauthenticated = MCPConnection(local, credentials: [:])
        do { _ = try await unauthenticated.start(); preconditionFailure("Expected 401") } catch { precondition(error.localizedDescription.contains("401")) }
        local.endpoint = "http://127.0.0.1:\(port)/redirect"
        let redirect = MCPConnection(local, credentials: ["Authorization":"Bearer fixture"])
        do { _ = try await redirect.start(); preconditionFailure("Expected redirect rejection") } catch { precondition(error.localizedDescription.contains("307")) }
        print("MCPTests: HTTP JSON, SSE, auth/session/protocol headers, tool errors and redirect rejection passed")
    }
}
