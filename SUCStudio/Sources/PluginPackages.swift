import Foundation
import AppKit

struct PluginPackage {
    var manifest: PluginManifest
    var skills: [StudioSkill]
    var servers: [MCPConfiguration]
    var hooks: [StudioHook]
    var format: String
    static func object(_ url: URL) throws -> [String:Any] {
        guard let value=try JSONSerialization.jsonObject(with:Data(ExtensionFiles.read(url,limit:1024*1024).utf8)) as? [String:Any] else{throw ExtensionFiles.failure("JSON plugin non valido.")};return value
    }
    @MainActor static func load(_ root: URL, dataRoot: URL) throws -> PluginPackage {
        let fm=FileManager.default
        let modern=root.appendingPathComponent("plugin.json")
        let legacy=root.appendingPathComponent(".codex-plugin/plugin.json")
        let claude=root.appendingPathComponent(".claude-plugin/plugin.json")
        let url=fm.fileExists(atPath:modern.path) ? modern : fm.fileExists(atPath:legacy.path) ? legacy:claude
        let relative=String(url.path.dropFirst(root.path.count+1))
        let doc=try object(ExtensionFiles.resolve(relative,root:root))
        if doc["formatVersion"] != nil {
            let manifest=try PluginValidation.decode(Data(ExtensionFiles.read(url,limit:1024*1024).utf8))
            return PluginPackage(manifest:manifest,skills:(manifest.skills ?? []).map{StudioSkill(name:$0.name,description:$0.description,instructions:$0.instructions)},servers:(manifest.mcpServers ?? []).map{MCPConfiguration(name:$0.name,transport:$0.transport,endpoint:$0.endpoint ?? "",executable:$0.executable ?? "",arguments:$0.arguments ?? [])},hooks:(manifest.hooks ?? []).map{StudioHook(name:$0.name,event:$0.event,command:$0.command)},format:"SUC v1")
        }
        guard let name=doc["name"] as? String,name.range(of:"^[a-z0-9][a-z0-9-]{0,127}$",options:.regularExpression) != nil else{throw ExtensionFiles.failure("Nome plugin obbligatorio in kebab-case.")}
        var metadata=doc
        let portable=url==modern
        if portable {
            // Portable extension overrides the complete Codex overlay, not individual fields.
            if let ext=(doc["extensions"] as? [String:Any])?["com.openai"] as? [String:Any] {metadata=ext}
            else if fm.fileExists(atPath:legacy.path) {metadata=try object(ExtensionFiles.resolve(".codex-plugin/plugin.json",root:root))}
        }
        var manifest=PluginManifest(formatVersion:1,identifier:name,name:name,version:doc["version"] as? String ?? "0.0.0",description:doc["description"] as? String,onboardingSkill:metadata["onboardingSkill"] as? String)
        var skills=[StudioSkill](),servers=[MCPConfiguration](),hooks=[StudioHook]()
        let skillPaths: [String] = portable ? ["skills"] : (doc["skills"] as? [String] ?? (doc["skills"] as? String).map{[$0]} ?? ["skills"])
        for path in skillPaths {
            let skillRoot=try ExtensionFiles.resolve(path,root:root)
            if fm.fileExists(atPath:skillRoot.appendingPathComponent("SKILL.md").path) {skills.append(try SkillDocument.directory(skillRoot))}
            else if fm.fileExists(atPath:skillRoot.path) {
                for folder in try fm.contentsOfDirectory(at:skillRoot,includingPropertiesForKeys:nil).sorted(by:{$0.path<$1.path}) {
                    if fm.fileExists(atPath:folder.appendingPathComponent("SKILL.md").path) {_ = try ExtensionFiles.resolve(String(folder.path.dropFirst(root.path.count+1)),root:root);skills.append(try SkillDocument.directory(folder))}
                }
            }
        }
        func expand(_ value: String) -> String {
            value.replacingOccurrences(of:"${PLUGIN_ROOT}",with:root.path).replacingOccurrences(of:"${CLAUDE_PLUGIN_ROOT}",with:root.path).replacingOccurrences(of:"${PLUGIN_DATA}",with:dataRoot.path).replacingOccurrences(of:"${CLAUDE_PLUGIN_DATA}",with:dataRoot.path)
        }
        let mcpPath=portable ? "mcp.json" : (doc["mcpServers"] as? String ?? ".mcp.json")
        let mcpURL=try ExtensionFiles.resolve(mcpPath,root:root)
        var mcp=[String:Any]()
        if fm.fileExists(atPath:mcpURL.path) {mcp=try object(mcpURL)["mcpServers"] as? [String:Any] ?? [:]}
        if !portable,let inline=doc["mcpServers"] as? [String:Any] {mcp=inline}
        for name in mcp.keys.sorted() {
            guard let value=mcp[name] as? [String:Any] else{throw ExtensionFiles.failure("MCP non valido: "+name)}
            var server=MCPConfiguration(name:name)
            let kind=value["type"] as? String ?? (value["command"]==nil ? "streamable-http":"stdio")
            if ["http","streamable-http"].contains(kind) {
                server.transport="http";server.endpoint=expand(value["url"] as? String ?? "");_ = try RemoteInference.endpoint(server.endpoint,path:"")
                server.headers=(value["headers"] as? [String:String])?.mapValues(expand)
            } else if kind=="stdio" {
                let command=expand(value["command"] as? String ?? "")
                guard !command.isEmpty else{throw ExtensionFiles.failure("MCP senza command.")}
                server.arguments=(value["args"] as? [String] ?? []).map(expand)
                if command.hasPrefix("/") {server.executable=command} else {server.executable="/usr/bin/env";server.arguments.insert(command,at:0)}
                server.environment=(value["env"] as? [String:String] ?? [:]).mapValues(expand)
                server.environment?["PLUGIN_ROOT"]=root.path;server.environment?["PLUGIN_DATA"]=dataRoot.path
                server.environment?["CLAUDE_PLUGIN_ROOT"]=root.path;server.environment?["CLAUDE_PLUGIN_DATA"]=dataRoot.path
            } else {throw ExtensionFiles.failure("Trasporto MCP non supportato: "+kind)}
            servers.append(server)
        }
        func addHooks(_ config: Any) throws {
            if let path=config as? String {hooks+=try HookDefinition.decode(Data(ExtensionFiles.read(ExtensionFiles.resolve(path,root:root),limit:1024*1024).utf8),root:root.path,dataRoot:dataRoot.path)}
            else if let array=config as? [Any] {for item in array{try addHooks(item)}}
            else if let object=config as? [String:Any] {hooks+=try HookDefinition.decode(JSONSerialization.data(withJSONObject:object),root:root.path,dataRoot:dataRoot.path)}
            else {throw ExtensionFiles.failure("Configurazione hooks non valida.")}
        }
        if let config=metadata["hooks"] {try addHooks(config)}
        else if fm.fileExists(atPath:root.appendingPathComponent("hooks/hooks.json").path) {try addHooks("hooks/hooks.json")}
        guard skills.count<=64,servers.count<=16,hooks.count<=128 else{throw ExtensionFiles.failure("Troppi componenti nel plugin.")}
        if metadata["apps"] != nil || fm.fileExists(atPath:root.appendingPathComponent(".app.json").path) {throw ExtensionFiles.failure("Il pacchetto richiede app OpenAI: non disponibili in SUC Studio. Rimuovi quel componente per creare un pacchetto portabile.")}
        if let onboarding=manifest.onboardingSkill {
            let path=try ExtensionFiles.resolve(onboarding,root:root)
            guard skills.contains(where:{$0.sourcePath==path.path}) else{throw ExtensionFiles.failure("onboardingSkill deve puntare a una skill inclusa.")}
        }
        manifest.skills=skills.map{PluginSkill(name:$0.name,description:$0.description,instructions:$0.instructions)}
        return PluginPackage(manifest:manifest,skills:skills,servers:servers,hooks:hooks,format:portable ? "Portable 1.0":"Codex / Claude")
    }
}
struct PackageMarketplaceEntry: Identifiable {
    var name: String
    var source: [String:String]
    var unavailable: Bool
    var root: URL
    var id: String {name}
}
enum PackageMarketplace {
    static func decode(_ data: Data, root: URL) throws -> [PackageMarketplaceEntry] {
        guard data.count<=1024*1024,let object=try JSONSerialization.jsonObject(with:data) as? [String:Any],let plugins=object["plugins"] as? [[String:Any]],plugins.count<=200 else{throw ExtensionFiles.failure("Indice marketplace non valido.")}
        return try plugins.map { item in
            guard let name=item["name"] as? String else{throw ExtensionFiles.failure("Plugin marketplace senza nome.")}
            let source=(item["source"] as? [String:String]) ?? (item["source"] as? String).map{["source":"local","path":$0]} ?? [:]
            guard ["local","url","git-subdir"].contains(source["source"] ?? "") else{throw ExtensionFiles.failure("Sorgente marketplace non supportata (npm non incluso).")}
            if source["source"]=="local" {_ = try ExtensionFiles.resolve(source["path"] ?? "",root:root)}
            return PackageMarketplaceEntry(name:name,source:source,unavailable:(item["policy"] as? [String:String])?["installation"]=="NOT_AVAILABLE",root:root)
        }
    }
    static func checkout(_ address: String, ref: String? = nil) async throws -> URL {
        guard let url=URL(string:address),url.scheme=="https",url.host != nil,url.user==nil,url.password==nil else{throw ExtensionFiles.failure("Repository HTTPS senza credenziali obbligatorio.")}
        let target=FileManager.default.temporaryDirectory.appendingPathComponent("suc-plugin-"+UUID().uuidString)
        func quote(_ value: String) -> String {"'"+value.replacingOccurrences(of:"'",with:"'\\''")+"'"}
        var command="/usr/bin/git -c core.hooksPath=/dev/null -c protocol.file.allow=never clone --depth 1 "
        if let ref {guard !ref.hasPrefix("-"),ref.utf8.count<256 else{throw ExtensionFiles.failure("Ref non valido.")};command+="--branch "+quote(ref)+" "}
        command+="-- "+quote(address)+" "+quote(target.path)
        do {
            let output=try await HookCommandRunner().run(command,input:Data(),root:FileManager.default.temporaryDirectory,environment:["GIT_TERMINAL_PROMPT":"0","GIT_CONFIG_NOSYSTEM":"1","GIT_CONFIG_GLOBAL":"/dev/null"],timeout:120,helper:StudioStore.commandHelper)
            guard output.2==0 else{throw ExtensionFiles.failure("Git: "+output.1)};return target
        } catch {try? FileManager.default.removeItem(at:target);throw error}
    }
}

extension StudioStore {
    func installPackage(_ source: URL) throws {
        let root=try ExtensionFiles.copy(source,kind:"Plugins")
        let dataRoot=try ExtensionFiles.storage("PluginData").appendingPathComponent(root.lastPathComponent)
        do {
            let package=try PluginPackage.load(root,dataRoot:dataRoot)
            guard !(archive.settings.plugins ?? []).contains(where:{$0.manifest.identifier==package.manifest.identifier}),(archive.settings.plugins?.count ?? 0)<32 else{throw ExtensionFiles.failure("Plugin già installato o limite 32 raggiunto.")}
            try FileManager.default.createDirectory(at:dataRoot,withIntermediateDirectories:true)
            let plugin=InstalledPlugin(manifest:package.manifest,source:source.path,skills:package.skills,servers:package.servers,hooks:package.hooks,bundleRoot:root.path,dataRoot:dataRoot.path,packageFormat:package.format)
            archive.settings.plugins=(archive.settings.plugins ?? [])+[plugin];scheduleSave()
        } catch {try? FileManager.default.removeItem(at:root);try? FileManager.default.removeItem(at:dataRoot);throw error}
    }
    func installMarketplacePackage(_ entry: PackageMarketplaceEntry) async throws {
        guard !entry.unavailable else{throw ExtensionFiles.failure("Plugin non disponibile secondo la policy del marketplace.")}
        if entry.source["source"]=="local" {try installPackage(ExtensionFiles.resolve(entry.source["path"] ?? "",root:entry.root))}
        else {
            let root=try await PackageMarketplace.checkout(entry.source["url"] ?? "",ref:entry.source["ref"])
            defer{try? FileManager.default.removeItem(at:root)}
            try installPackage(entry.source["source"]=="git-subdir" ? ExtensionFiles.resolve(entry.source["path"] ?? "",root:root):root)
        }
    }
}
