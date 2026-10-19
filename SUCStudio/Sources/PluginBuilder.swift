import Foundation

/// Independent drafts survive component/editor changes; only selected components are exported.
struct PluginBuilderDraft {
    var name="my-plugin"
    var description="Skill, hook e strumenti riutilizzabili per il mio lavoro."
    var includeSkill=true
    var includeHook=false
    var includeMCP=false
    var skill=Self.skillSample
    var hook=Self.hookSample
    var mcp=Self.mcpSample
    var selectedComponents: [String] {
        (includeSkill ? ["skill"]:[]) + (includeHook ? ["hook"]:[]) + (includeMCP ? ["mcp"]:[])
    }
    static let skillSample="---\nname: review-code\ndescription: Rivedi il codice quando viene richiesta una revisione.\n---\nLeggi i file pertinenti. Cerca bug concreti. Cita file e righe; distingui i problemi dalle ipotesi."
    static let hookSample=#"{"hooks":{"SessionStart":[{"hooks":[{"type":"command","command":"printf 'Prima di agire consulta README.md.\n'","timeout":10}]}]}}"#
    static let mcpSample=#"{"$schema":"https://agent-plugins.org/schemas/1.0.0/mcp.schema.json","mcpServers":{"example":{"type":"streamable-http","url":"http://127.0.0.1:8081/mcp"}}}"#
    mutating func reset(_ component: String) {
        switch component {case "hook":hook=Self.hookSample;case "mcp":mcp=Self.mcpSample;default:skill=Self.skillSample}
    }
    func documents() throws -> [String:String] {
        guard name.range(of:"^[a-z0-9][a-z0-9-]{0,127}$",options:.regularExpression) != nil else{throw ExtensionFiles.failure("Usa un nome plugin kebab-case, per esempio my-plugin.")}
        guard !selectedComponents.isEmpty else{throw ExtensionFiles.failure("Seleziona almeno un componente: Skill, Hook o MCP.")}
        var files=[String:String]()
        if includeSkill {
            guard skill.utf8.count<=64*1024 else{throw ExtensionFiles.failure("Skill oltre 64 KiB.")}
            let parsed=SkillDocument.parse(skill,fallbackName:"")
            guard !parsed.name.isEmpty,!(parsed.description ?? "").trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,!parsed.instructions.isEmpty else{throw ExtensionFiles.failure("SKILL.md: servono name, description e corpo.")}
            guard parsed.name.range(of:"^[a-z0-9][a-z0-9-]{0,63}$",options:.regularExpression) != nil else{throw ExtensionFiles.failure("Nome skill: usa lettere minuscole, numeri e trattini (massimo 64 caratteri).")}
            files["skills/\(parsed.name)/SKILL.md"]=skill
        }
        if includeHook {_ = try HookDefinition.decode(Data(hook.utf8));files["hooks/hooks.json"]=hook}
        if includeMCP {
            guard mcp.utf8.count<=1024*1024,(try JSONSerialization.jsonObject(with:Data(mcp.utf8)) as? [String:Any])?["mcpServers"] is [String:Any] else{throw ExtensionFiles.failure("mcp.json: serve un oggetto mcpServers (massimo 1 MiB).")}
            files["mcp.json"]=mcp
        }
        let manifest:[String:Any]=["$schema":"https://agent-plugins.org/schemas/1.0.0/plugin.schema.json","name":name,"version":"1.0.0","description":description]
        files["plugin.json"]=String(decoding:try JSONSerialization.data(withJSONObject:manifest,options:[.prettyPrinted,.sortedKeys]),as:UTF8.self)
        return files
    }
    @MainActor func export(to parent: URL) throws -> URL {
        let files=try documents(),fm=FileManager.default
        let target=parent.appendingPathComponent(name,isDirectory:true)
        guard !fm.fileExists(atPath:target.path) else{throw ExtensionFiles.failure("Cartella già esistente: scegli un altro nome.")}
        // Validate the complete bundle before publishing it. Existing folders are never removed on failure.
        let staging=parent.appendingPathComponent(".suc-plugin-"+UUID().uuidString,isDirectory:true)
        try fm.createDirectory(at:staging,withIntermediateDirectories:false)
        defer{try? fm.removeItem(at:staging)}
        for (path,code) in files {
            let file=staging.appendingPathComponent(path)
            try fm.createDirectory(at:file.deletingLastPathComponent(),withIntermediateDirectories:true)
            try code.write(to:file,atomically:true,encoding:.utf8)
        }
        _ = try PluginPackage.load(staging,dataRoot:staging.appendingPathComponent("data"))
        try fm.moveItem(at:staging,to:target)
        return target
    }
}
