import Foundation

enum PluginCatalog {
    static func find(_ key:String,plugins:[InstalledPlugin])->InstalledPlugin? {
        plugins.first{$0.enabled && [$0.id.uuidString.lowercased(),$0.manifest.identifier.lowercased(),$0.manifest.name.lowercased()].contains(key.lowercased())}
    }
    static func load(_ key:String,plugins:[InstalledPlugin],skills:[StudioSkill],explicit:Bool=false)->String? {
        guard let plugin=find(key,plugins:plugins) else{return nil}
        let ids=Set(plugin.skills.map(\.id)),active=skills.filter{ids.contains($0.id) && $0.enabled && (explicit || $0.allowImplicitInvocation != false)}
        var parts=["Plugin: \(plugin.manifest.name) [\(plugin.manifest.identifier)]",plugin.manifest.description ?? ""]
        for skill in active {parts.append("Skill \(skill.name):\n\(skill.instructions)")}
        parts.append("MCP del plugin: "+plugin.servers.filter{$0.enabled}.map(\.name).joined(separator:", "))
        parts.append("Usa mcp_list e mcp_schema per gli strumenti connessi. Restano necessarie le normali approvazioni. Gli hook seguono gli eventi già configurati, senza attivarne di nuovi.")
        return parts.filter{!$0.isEmpty}.joined(separator:"\n\n")
    }
}
