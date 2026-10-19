import Foundation

enum SkillDocument {
    /// Small front-matter reader for name/title/description, including folded/literal descriptions.
    static func parse(_ source: String, fallbackName: String) -> StudioSkill {
        let normalized=source.replacingOccurrences(of:"\r\n",with:"\n")
        let lines=normalized.components(separatedBy:"\n")
        var metadata:[String:String]=[:],body=normalized
        if lines.first?.trimmingCharacters(in:.whitespacesAndNewlines)=="---",
           let end=lines.dropFirst().firstIndex(where:{$0.trimmingCharacters(in:.whitespacesAndNewlines)=="---"}) {
            var index=1
            while index<end {
                let line=lines[index];index+=1
                guard !line.hasPrefix(" "),!line.hasPrefix("\t"),let colon=line.firstIndex(of:":") else{continue}
                let key=line[..<colon].trimmingCharacters(in:.whitespaces)
                var value=line[line.index(after:colon)...].trimmingCharacters(in:.whitespaces)
                if [">", "|", ">-", "|-", ">+", "|+"].contains(value) {
                    let literal=value.hasPrefix("|");var chunks:[String]=[]
                    while index<end && (lines[index].hasPrefix(" ") || lines[index].hasPrefix("\t") || lines[index].isEmpty) {
                        chunks.append(lines[index].trimmingCharacters(in:.whitespaces));index+=1
                    }
                    value=chunks.joined(separator:literal ? "\n":" ").trimmingCharacters(in:.whitespacesAndNewlines)
                } else if value.count>=2 && ((value.first=="\"" && value.last=="\"") || (value.first=="'" && value.last=="'")) {
                    if value.first=="\"",let data=value.data(using:.utf8),let decoded=try? JSONDecoder().decode(String.self,from:data) {value=decoded}
                    else {value=String(value.dropFirst().dropLast()).replacingOccurrences(of:"''",with:"'")}
                }
                metadata[key]=value
            }
            body=lines[(end+1)...].joined(separator:"\n").trimmingCharacters(in:.whitespacesAndNewlines)
        }
        return StudioSkill(name:metadata["name"] ?? metadata["title"] ?? fallbackName,description:metadata["description"],instructions:body)
    }
    static func load(_ action: WorkspaceAction, skills: [StudioSkill]) throws -> String {
        let id=action.arguments["id"] ?? action.arguments["name"] ?? ""
        guard let skill=skills.first(where:{($0.id.uuidString.caseInsensitiveCompare(id) == .orderedSame || $0.name==id) && $0.enabled}),!skill.instructions.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty else {
            throw MCPConnection.failure("Skill non disponibile o disattivata. Usa un ID del catalogo corrente.")
        }
        return "Skill caricata: \(skill.name) [\(skill.id.uuidString)]\nDescrizione: \(skill.effectiveDescription)\nRisorse: \(skill.resourceRoot ?? "nessuna")\nUsa skill_resources e skill_resource per leggere solo i file necessari.\n\nIstruzioni della skill:\n\(skill.instructions)"
    }
}
