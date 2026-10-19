import Foundation
import AppKit

/// Copies reviewed resources into app-owned storage. No traversal or external symlink targets.
enum ExtensionFiles {
    static func failure(_ message: String) -> NSError { NSError(domain:"SUCExtensions",code:1,userInfo:[NSLocalizedDescriptionKey:message]) }
    static func storage(_ kind: String) throws -> URL {
        let base=try FileManager.default.url(for:.applicationSupportDirectory,in:.userDomainMask,appropriateFor:nil,create:true)
        let url=base.appendingPathComponent(StudioPlatform.dataDirectory).appendingPathComponent(kind)
        try FileManager.default.createDirectory(at:url,withIntermediateDirectories:true);return url
    }
    static func resolve(_ path: String, root: URL) throws -> URL {
        guard !path.hasPrefix("/"),!path.contains("\0") else{throw failure("Percorso risorsa relativo obbligatorio.")}
        let base=root.resolvingSymlinksInPath().standardizedFileURL
        let url=base.appendingPathComponent(path).resolvingSymlinksInPath().standardizedFileURL
        guard url.path==base.path || url.path.hasPrefix(base.path+"/") else{throw failure("La risorsa esce dal pacchetto.")}
        return url
    }
    static func files(_ root: URL) throws -> [String] {
        guard let enumerator=FileManager.default.enumerator(at:root,includingPropertiesForKeys:[.isRegularFileKey,.fileSizeKey],options:[]) else{throw failure("Cartella non leggibile.")}
        var result:[String]=[];var total=0
        for case let url as URL in enumerator {
            if [".git",".build","node_modules"].contains(url.lastPathComponent) {enumerator.skipDescendants();continue}
            let relative=String(url.path.dropFirst(root.path.count+1));let safe=try resolve(relative,root:root)
            let info=try safe.resourceValues(forKeys:[.isRegularFileKey,.fileSizeKey])
            if info.isRegularFile==true {total+=info.fileSize ?? 0;result.append(relative)}
            guard result.count<=2048,total<=64*1024*1024 else{throw failure("Pacchetto oltre 2048 file o 64 MiB.")}
        }
        return result.sorted()
    }
    static func copy(_ root: URL, kind: String) throws -> URL {
        let entries=try files(root)
        let target=try storage(kind).appendingPathComponent(UUID().uuidString)
        do {
            try FileManager.default.createDirectory(at:target,withIntermediateDirectories:true)
            for entry in entries {
                let output=target.appendingPathComponent(entry)
                try FileManager.default.createDirectory(at:output.deletingLastPathComponent(),withIntermediateDirectories:true)
                try FileManager.default.copyItem(at:resolve(entry,root:root),to:output)
            }
            return target
        } catch {try? FileManager.default.removeItem(at:target);throw error}
    }
    static func read(_ url: URL, limit: Int = 64*1024) throws -> String {
        guard (try url.resourceValues(forKeys:[.fileSizeKey]).fileSize ?? Int.max)<=limit else{throw failure("File troppo grande (limite \(limit) byte).")}
        guard let text=String(data:try Data(contentsOf:url),encoding:.utf8),!text.contains("\0") else{throw failure("Risorsa non testuale UTF-8.")};return text
    }
}

extension SkillDocument {
    static func directory(_ root: URL) throws -> StudioSkill {
        let document=try ExtensionFiles.resolve("SKILL.md",root:root)
        var skill=parse(try ExtensionFiles.read(document),fallbackName:root.lastPathComponent)
        guard !skill.name.isEmpty,!(skill.description ?? "").isEmpty,!skill.instructions.isEmpty else{throw ExtensionFiles.failure("SKILL.md richiede name, description e istruzioni.")}
        skill.resourceRoot=root.path;skill.sourcePath=document.path
        let metadata=root.appendingPathComponent("agents/openai.yaml")
        if FileManager.default.fileExists(atPath:metadata.path) {
            // The scalar metadata subset is deliberately independent of third-party YAML runtimes.
            let text=try ExtensionFiles.read(try ExtensionFiles.resolve("agents/openai.yaml",root:root))
            for line in text.components(separatedBy:.newlines) {
                let trimmed=line.trimmingCharacters(in:.whitespaces)
                guard let colon=trimmed.firstIndex(of:":") else{continue}
                let key=String(trimmed[..<colon]);var value=String(trimmed[trimmed.index(after:colon)...]).trimmingCharacters(in:.whitespaces)
                if value.count>=2, ["\"","'"].contains(String(value.prefix(1))),value.first==value.last {value=String(value.dropFirst().dropLast())}
                switch key {case "display_name":skill.displayName=value;case "default_prompt":skill.defaultPrompt=value;case "allow_implicit_invocation":skill.allowImplicitInvocation=value != "false";default:break}
            }
        }
        return skill
    }
    static func resource(_ action: WorkspaceAction, skills: [StudioSkill]) throws -> String {
        let id=action.arguments["id"] ?? action.arguments["name"] ?? ""
        guard let skill=skills.first(where:{$0.enabled && ($0.id.uuidString.caseInsensitiveCompare(id) == .orderedSame || $0.name==id)}),let path=skill.resourceRoot else{throw ExtensionFiles.failure("Skill con risorse non disponibile.")}
        let root=URL(fileURLWithPath:path)
        if action.name=="skill_resources" {return try ExtensionFiles.files(root).joined(separator:"\n")}
        return try ExtensionFiles.read(ExtensionFiles.resolve(action.arguments["path"] ?? "",root:root))
    }
    static func explicitInstructions(_ prompt: String, skills: [StudioSkill]) -> String {
        skills.filter { skill in
            guard skill.enabled else{return false}
            let escaped=NSRegularExpression.escapedPattern(for:skill.name)
            return prompt.range(of:"(?:^|\\s)[@$]"+escaped+"(?=$|\\s|[.,:;!?])",options:.regularExpression) != nil
        }.map{"Skill richiesta esplicitamente: \($0.name)\n\($0.instructions)\nRisorse: \($0.resourceRoot ?? "nessuna")"}.joined(separator:"\n\n")
    }
}

extension StudioStore {
    func discoverSkills() {
        var roots=[FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".agents/skills"),URL(fileURLWithPath:"/etc/codex/skills")]
        if let path=selected?.workspacePath {
            var url=URL(fileURLWithPath:path).standardizedFileURL
            while true {
                roots.append(url.appendingPathComponent(".agents/skills"))
                if FileManager.default.fileExists(atPath:url.appendingPathComponent(".git").path) || url.path=="/" {break}
                url.deleteLastPathComponent()
            }
        }
        var count=0
        for root in roots {
            for folder in (try? FileManager.default.contentsOfDirectory(at:root,includingPropertiesForKeys:nil)) ?? [] {
                if let value=try? SkillDocument.directory(folder), !(archive.settings.skills ?? []).contains(where:{$0.sourcePath==value.sourcePath}) {saveSkill(value);count+=1}
            }
        }
        appendLog("Discovery skill: \(count) nuove; da revisionare e attivare nei Settings.")
    }
}
