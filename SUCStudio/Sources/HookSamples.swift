import Foundation
struct HookSample { var name:String;var event:String;var command:String }
enum HookSamples {
    static let all:[HookSample] = [
        HookSample(name:"Registra evento",event:"after_tool",command:"printf 'Evento: %s\\nSessione: %s\\nDirectory: %s\\n' \"$SUC_HOOK_EVENT\" \"$SUC_SESSION_ID\" \"$SUC_WORKSPACE\""),
        HookSample(name:"Verifica spazi nel diff Git",event:"before_tool",command:"git diff --check"),
        HookSample(name:"Mostra file modificati",event:"request_end",command:"git status --short"),
        HookSample(name:"Richiedi una directory Git",event:"request_start",command:"git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { printf 'Seleziona una directory Git.\\n' >&2; exit 1; }")
    ]
}
