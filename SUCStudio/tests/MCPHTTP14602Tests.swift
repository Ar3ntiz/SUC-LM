import Foundation
@main struct MCPHTTP14602Tests {
    @MainActor static func main() async throws {
        let fixture=CommandLine.arguments[1]
        let server = Process(), output = Pipe()
        server.executableURL = URL(fileURLWithPath: "/usr/bin/python3"); server.arguments = [fixture, "--http"]; server.standardOutput = output
        try server.run(); defer { server.terminate() }
        let data = output.fileHandleForReading.availableData
        guard let port = Int(String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)) else { throw MCPConnection.failure("Fixture HTTP non avviata: verifica i permessi di ascolto localhost dell’ambiente di test.") }
        var local = MCPConfiguration(name: "HTTP Fixture", transport: "http", endpoint: "http://127.0.0.1:\(port)/mcp", enabled: true)
        local.headers=["Authorization":"Bearer fixture"]
        let remote = MCPConnection(local, credentials: [:]); defer { remote.stop() }
        let remoteTools = try await remote.start(); precondition(remoteTools.count == 2)
        let result = try await remote.call(remoteTools[0], arguments: ["text":"SSE works"])
        precondition(result.contains("SSE works"))
        do { _ = try await remote.call(remoteTools[1], arguments: [:]); preconditionFailure("Expected SSE tool error") } catch { precondition(error.localizedDescription.contains("expected failure")) }
        local.headers=nil
        let unauthenticated = MCPConnection(local, credentials: [:])
        do { _ = try await unauthenticated.start(); preconditionFailure("Expected 401") } catch { precondition(error.localizedDescription.contains("401")) }
        local.endpoint = "http://127.0.0.1:\(port)/redirect"
        let redirect = MCPConnection(local, credentials: ["Authorization":"Bearer fixture"])
        do { _ = try await redirect.start(); preconditionFailure("Expected redirect rejection") } catch { precondition(error.localizedDescription.contains("307")) }
        print("MCPTests: HTTP JSON, SSE, auth/session/protocol headers, tool errors and redirect rejection passed")
    }
}
