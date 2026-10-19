import Foundation
struct HookSample {var name:String;var event:String;var command:String}
enum HookSamples {
    static let all:[HookSample]=[
        HookSample(name:"Contesto a inizio sessione",event:"SessionStart",command:#"printf '%s\n' '{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"Prima di modificare file, consulta README.md nella directory di lavoro."}}'"#),
        HookSample(name:"Valida prompt (Python)",event:"UserPromptSubmit",command:"""
        /usr/bin/python3 -c 'import json,sys; event=json.load(sys.stdin); print(json.dumps({"hookSpecificOutput":{"hookEventName":"UserPromptSubmit","additionalContext":"Rispondi in italiano e cita i file consultati."}}))'
        """),
        HookSample(name:"Blocca scritture (usa matcher ^apply_patch$)",event:"PreToolUse",command:#"printf '%s\n' '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Questo progetto è in sola lettura."}}'"#),
        HookSample(name:"Verifica diff dopo un tool",event:"PostToolUse",command:#"git diff --check || { printf 'Correggi gli spazi nel diff.\n' >&2; exit 2; }"#),
        HookSample(name:"Controllo finale senza ciclo",event:"Stop",command:"""
        /usr/bin/python3 -c 'import json,sys; e=json.load(sys.stdin); print(json.dumps({} if e.get("stop_hook_active") else {"decision":"block","reason":"Verifica una volta che la risposta soddisfi la richiesta, poi concludi."}))'
        """),
        HookSample(name:"Log evento in plugin data",event:"PostToolUse",command:#"test -n "$PLUGIN_DATA" || exit 0; mkdir -p "$PLUGIN_DATA"; cat >> "$PLUGIN_DATA/events.jsonl"; printf '\n' >> "$PLUGIN_DATA/events.jsonl""#)
    ]
}
