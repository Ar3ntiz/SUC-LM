import Foundation
import AppKit

extension StudioStore {
    func openHelp() {
        guard let page = Bundle.main.url(forResource: archive.settings.effectiveLanguage == "en" ? "USAGE-en":"USAGE", withExtension: "html") else { error = "Guida HTML assente: ricompila l’app."; return }
        let browser = NSWorkspace.shared.urlForApplication(toOpen: URL(string: "https://example.com")!) ?? NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.Safari")
        guard let browser else { error = "Nessun browser disponibile per la guida HTML."; return }
        NSWorkspace.shared.open([page], withApplicationAt: browser, configuration: NSWorkspace.OpenConfiguration()) { _, failure in
            if let failure { Task { @MainActor in self.error = failure.localizedDescription } }
        }
    }
    func saveSkill(_ skill: StudioSkill) {
        guard !skill.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, skill.instructions.utf8.count <= 64 * 1024, (skill.description ?? "").utf8.count <= 4 * 1024 else { error = "Titolo obbligatorio; descrizione fino a 4 KiB e istruzioni fino a 64 KiB."; return }
        var skills = archive.settings.skills ?? []
        if let index = skills.firstIndex(where: { $0.id == skill.id }) { skills[index] = skill } else { skills.append(skill) }
        archive.settings.skills = skills; scheduleSave()
    }
    func importSkill() {
        let panel=NSOpenPanel();panel.canChooseFiles=true;panel.canChooseDirectories=true;panel.allowsMultipleSelection=false
        panel.message=L("Scegli SKILL.md o una cartella skill con risorse")
        guard panel.runModal() == .OK,let url=panel.url else{return}
        do {
            let directory=(try url.resourceValues(forKeys:[.isDirectoryKey]).isDirectory)==true
            let root=directory ? url:url.deletingLastPathComponent()
            if directory || url.lastPathComponent=="SKILL.md" {
                let target=try ExtensionFiles.copy(root,kind:"Skills")
                do {saveSkill(try SkillDocument.directory(target))} catch {try? FileManager.default.removeItem(at:target);throw error}
            } else {saveSkill(SkillDocument.parse(try ExtensionFiles.read(url),fallbackName:url.deletingPathExtension().lastPathComponent))}
        } catch {self.error=error.localizedDescription}
    }
    func saveMCP(_ configuration: MCPConfiguration, credentials: [String: String]) throws {
        guard !configuration.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw MCPConnection.failure("Nome server obbligatorio.") }
        guard configuration.arguments.count <= 256 else { throw MCPConnection.failure("Troppi argomenti.") }
        try MCPSecrets.write(credentials, id: configuration.id)
        disconnectMCP(configuration.id)
        var servers = archive.settings.mcpServers ?? []
        if let index = servers.firstIndex(where: { $0.id == configuration.id }) { servers[index] = configuration } else { servers.append(configuration) }
        archive.settings.mcpServers = servers; scheduleSave()
        if configuration.enabled { Task { await connectMCP(configuration) } }
    }
    func toggleMCP(_ configuration: MCPConfiguration, enabled: Bool) {
        guard let index = archive.settings.mcpServers?.firstIndex(where: { $0.id == configuration.id }) else { return }
        archive.settings.mcpServers?[index].enabled = enabled; scheduleSave()
        disconnectMCP(configuration.id)
        if enabled { var value = configuration; value.enabled = true; Task { await connectMCP(value) } }
    }
    func disconnectMCP(_ id: UUID) {
        let old = mcpConnections.removeValue(forKey: id); old?.stop()
        mcpTools.removeAll { $0.serverID == id }; mcpStatus[id] = "Spento"
    }
    func connectMCP(_ configuration: MCPConfiguration) async {
        guard archive.settings.mcpServers?.contains(where: { $0.id == configuration.id && $0.enabled }) == true else { return }
        disconnectMCP(configuration.id); mcpStatus[configuration.id] = "Connessione…"
        var attempt: MCPConnection?
        do {
            let client = MCPConnection(configuration, credentials: try MCPSecrets.read(configuration.id)); attempt = client; mcpConnections[configuration.id] = client
            client.onDisconnect = { [weak self, weak client] in
                guard let self, let client, self.mcpConnections[configuration.id] === client else { return }
                self.mcpConnections.removeValue(forKey: configuration.id); self.mcpTools.removeAll { $0.serverID == configuration.id }; self.mcpStatus[configuration.id] = "Scollegato"
            }
            let tools = try await client.start()
            guard mcpConnections[configuration.id] === client else { return }
            mcpTools += tools; mcpStatus[configuration.id] = "Connesso · \(tools.count) strumenti"
        } catch {
            if let attempt, mcpConnections[configuration.id] === attempt { disconnectMCP(configuration.id) }
            if mcpConnections[configuration.id] == nil, archive.settings.mcpServers?.contains(where: { $0.id == configuration.id && $0.enabled }) == true { mcpStatus[configuration.id] = "Errore: \(error.localizedDescription)" }
        }
    }
    func removeMCP(_ id: UUID) { disconnectMCP(id); archive.settings.mcpServers?.removeAll { $0.id == id }; MCPSecrets.remove(id); scheduleSave() }
    func runMCP(_ tool: MCPTool, argumentsJSON: String, sessionID: UUID) async throws {
        guard !mcpRuns.contains(sessionID), let session = archive.sessions.first(where: { $0.id == sessionID }), (session.draftAttachments ?? []).count < AttachmentReader.maxFiles else { throw MCPConnection.failure("Attendi la chiamata corrente o libera uno spazio tra gli allegati.") }
        guard !attachmentLoads.contains(sessionID) else { throw MCPConnection.failure("Attendi l’importazione degli allegati.") }
        guard let client = mcpConnections[tool.serverID], mcpTools.contains(where: { $0.id == tool.id }), archive.settings.mcpServers?.contains(where: { $0.id == tool.serverID && $0.enabled }) == true else { throw MCPConnection.failure("MCP non attivo. Riconnettilo nei Settings.") }
        guard argumentsJSON.utf8.count <= 64 * 1024, let arguments = try JSONSerialization.jsonObject(with: Data(argumentsJSON.utf8)) as? [String: Any] else { throw MCPConnection.failure("Gli argomenti devono essere un oggetto JSON (massimo 64 KiB).") }
        mcpRuns.insert(sessionID); defer { mcpRuns.remove(sessionID); scheduleChatQueue(sessionID) }
        let logID=UUID().uuidString;operationJournal.update(id:logID,title:"MCP · "+tool.serverName+" / "+tool.name,state:"In corso",sessionID:sessionID)
        var logSuccess=false
        defer{operationJournal.update(id:logID,state:logSuccess ? "Completata":"Errore")}
        let result = try await client.call(tool, arguments: arguments)
        guard let index = archive.sessions.firstIndex(where: { $0.id == sessionID }) else { return }
        let attachment = ChatAttachment(name: "MCP · \(tool.serverName) / \(tool.name)", text: result, originalBytes: result.utf8.count)
        var files = archive.sessions[index].draftAttachments ?? []
        try AttachmentReader.validate(files + [attachment]); files.append(attachment)
        archive.sessions[index].draftAttachments = files; archive.sessions[index].updated = Date(); scheduleSave();logSuccess=true
    }
}
