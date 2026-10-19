import Foundation
import AppKit
import CryptoKit

@MainActor enum PluginValidation {
    static func decode(_ data: Data) throws -> PluginManifest {
        guard data.count<=1024*1024 else{throw MCPConnection.failure("Manifest oltre 1 MiB.")}
        let value=try JSONDecoder().decode(PluginManifest.self,from:data)
        guard value.formatVersion==1,!value.name.isEmpty,value.name.count<=128,!value.version.isEmpty,value.version.count<=40,
              value.identifier.range(of:"^[a-z0-9][a-z0-9.-]{0,127}$",options:.regularExpression) != nil,
              (value.skills?.count ?? 0)<=64,(value.mcpServers?.count ?? 0)<=16,(value.hooks?.count ?? 0)<=16
        else{throw MCPConnection.failure("Manifest plugin non valido o limiti superati.")}
        for skill in value.skills ?? [] {guard !skill.name.isEmpty,skill.instructions.utf8.count<=64*1024,(skill.description ?? "").utf8.count<=4*1024 else{throw MCPConnection.failure("Skill plugin non valida.")}}
        for server in value.mcpServers ?? [] {
            guard !server.name.isEmpty,["stdio","http"].contains(server.transport),(server.arguments?.count ?? 0)<=256 else{throw MCPConnection.failure("MCP plugin non valido.")}
            if server.transport=="stdio" {guard server.executable?.hasPrefix("/")==true else{throw MCPConnection.failure("Eseguibile MCP: percorso assoluto obbligatorio.")}}
            else{_ = try RemoteInference.endpoint(server.endpoint ?? "",path:"")}
        }
        for hook in value.hooks ?? [] {guard !hook.name.isEmpty,["before_tool","after_tool","request_start","request_end"].contains(hook.event),!hook.command.isEmpty,hook.command.utf8.count<=32768 else{throw MCPConnection.failure("Hook plugin non valido.")}}
        return value
    }
    static func verified(_ data: Data, sha256: String) throws -> PluginManifest {
        let digest=SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined()
        guard sha256.lowercased()==digest else{throw MCPConnection.failure("Checksum del plugin non corrispondente.")}
        return try decode(data)
    }
    static func download(_ address: String) async throws -> Data {
        guard let url=URL(string:address),url.scheme=="https",url.host != nil,url.user==nil,url.password==nil else{throw MCPConnection.failure("Marketplace e manifest remoti richiedono URL HTTPS senza credenziali.")}
        let settings=URLSessionConfiguration.ephemeral;settings.timeoutIntervalForRequest=20;settings.timeoutIntervalForResource=30
        let client=URLSession(configuration:settings,delegate:MCPHTTPDelegate(),delegateQueue:nil);defer{client.invalidateAndCancel()}
        let (bytes,response)=try await client.bytes(from:url)
        guard (response as? HTTPURLResponse)?.statusCode==200 else{throw MCPConnection.failure("Download plugin: HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0). Configura l’URL finale, senza redirect.")}
        var data=Data();for try await byte in bytes{data.append(byte);guard data.count<=1024*1024 else{throw MCPConnection.failure("Download oltre 1 MiB.")}}
        return data
    }
}
extension StudioStore {
    func importPlugin() {
        let panel=NSOpenPanel();panel.canChooseFiles=true;panel.canChooseDirectories=true;panel.allowsMultipleSelection=false;panel.message="Scegli plugin.json o la cartella che lo contiene"
        guard panel.runModal() == .OK,let selected=panel.url else{return}
        do {
            let directory=try selected.resourceValues(forKeys:[.isDirectoryKey]).isDirectory==true
            var root=directory ? selected:selected.deletingLastPathComponent()
            if [".codex-plugin",".claude-plugin"].contains(root.lastPathComponent) {root.deleteLastPathComponent()}
            try installPackage(root)
        } catch {self.error=error.localizedDescription}
    }
    func installPlugin(_ manifest: PluginManifest, source: String) throws {
        guard !(archive.settings.plugins ?? []).contains(where:{$0.manifest.identifier==manifest.identifier}) else{throw MCPConnection.failure("Plugin già installato. Rimuovilo esplicitamente prima di installare un’altra versione.")}
        guard (archive.settings.plugins?.count ?? 0)<32 else{throw MCPConnection.failure("Massimo 32 plugin.")}
        let skills=(manifest.skills ?? []).map{StudioSkill(name:$0.name,description:$0.description,instructions:$0.instructions)}
        let servers=(manifest.mcpServers ?? []).map{MCPConfiguration(name:$0.name,transport:$0.transport,endpoint:$0.endpoint ?? "",executable:$0.executable ?? "",arguments:$0.arguments ?? [])}
        let hooks=(manifest.hooks ?? []).map{StudioHook(name:$0.name,event:$0.event,command:$0.command)}
        archive.settings.plugins=(archive.settings.plugins ?? [])+[InstalledPlugin(manifest:manifest,source:source,skills:skills,servers:servers,hooks:hooks)];scheduleSave()
    }
    func setPlugin(_ plugin: InstalledPlugin, enabled: Bool, review: Bool = true) {
        guard let index=archive.settings.plugins?.firstIndex(where:{$0.id==plugin.id}) else{return}
        if enabled && review {
            let alert=NSAlert();alert.messageText="Attivare \(plugin.manifest.name)?"
            alert.informativeText="\(plugin.skills.count) skill, \(plugin.servers.count) MCP e \(plugin.hooks.count) hook. Gli MCP locali verranno avviati. Gli hook restano disattivati fino alla revisione esplicita della loro definizione nella sezione Hook. Consulta Dettagli prima di attivare."
            alert.addButton(withTitle:"Attiva");alert.addButton(withTitle:"Annulla");guard alert.runModal() == .alertFirstButtonReturn else{return}
        }
        var plugin=archive.settings.plugins![index]
        if plugin.enabled {
            plugin.skills=plugin.skills.map{original in archive.settings.skills?.first{$0.id==original.id} ?? original}
            plugin.servers=plugin.servers.map{original in archive.settings.mcpServers?.first{$0.id==original.id} ?? original}
            plugin.hooks=plugin.hooks.map{original in archive.settings.hooks?.first{$0.id==original.id} ?? original}
        }
        let skillIDs=Set(plugin.skills.map(\.id)),serverIDs=Set(plugin.servers.map(\.id)),hookIDs=Set(plugin.hooks.map(\.id))
        archive.settings.skills?.removeAll{skillIDs.contains($0.id)};archive.settings.mcpServers?.removeAll{serverIDs.contains($0.id)};archive.settings.hooks?.removeAll{hookIDs.contains($0.id)}
        for id in serverIDs{disconnectMCP(id)}
        if enabled && plugin.activationDone != true {
            plugin.skills=plugin.skills.map{var v=$0;v.enabled=true;return v}
            plugin.servers=plugin.servers.map{var v=$0;v.enabled=true;return v}
            plugin.activationDone=true
        }
        if enabled {
            archive.settings.skills=(archive.settings.skills ?? [])+plugin.skills
            archive.settings.mcpServers=(archive.settings.mcpServers ?? [])+plugin.servers
            archive.settings.hooks=(archive.settings.hooks ?? [])+plugin.hooks.map{var v=$0;v.enabled=v.enabled && HookDefinition.trusted(v);return v}
            for server in plugin.servers where server.enabled {Task{await connectMCP(server)}}
        }
        plugin.enabled=enabled;archive.settings.plugins?[index]=plugin;scheduleSave()
    }
    func removePlugin(_ plugin: InstalledPlugin) {setPlugin(plugin,enabled:false);for server in plugin.servers{MCPSecrets.remove(server.id)};archive.settings.plugins?.removeAll{$0.id==plugin.id};scheduleSave()}
}
