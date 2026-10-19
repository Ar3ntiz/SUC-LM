import Foundation
import AppKit
import CryptoKit
import Darwin

struct HookOutcome {
    var context=[String]()
    var deny: String?
    var permission: Bool?
    var updatedInput: [String:String]?
    var replacement: String?
    var continuation: String?
}
enum HookDefinition {
    static let events=["SessionStart","SessionEnd","UserPromptSubmit","PreToolUse","PermissionRequest","PostToolUse","Stop","Interrupt","SubagentStart","SubagentStop","PreCompact","PostCompact"]
    static let aliases=["request_start":"UserPromptSubmit","request_end":"Stop","before_tool":"PreToolUse","after_tool":"PostToolUse"]
    static func event(_ value: String) -> String { aliases[value] ?? value }
    static func digest(_ hook: StudioHook) -> String {
        var copy=hook;copy.trustedDigest=nil;copy.enabled=false
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        return SHA256.hash(data:(try? encoder.encode(copy)) ?? Data()).map{String(format:"%02x",$0)}.joined()
    }
    static func trusted(_ hook: StudioHook) -> Bool { hook.trustedDigest==digest(hook) }
    static func matches(_ hook: StudioHook, event: String, tool: String) -> Bool {
        guard Self.event(hook.event)==event else{return false}
        guard !["UserPromptSubmit","Stop","Interrupt"].contains(event),let matcher=hook.matcher,!matcher.isEmpty,matcher != "*" else{return true}
        return tool.range(of:matcher,options:.regularExpression) != nil
    }
    static func decode(_ data: Data, root: String? = nil, dataRoot: String? = nil) throws -> [StudioHook] {
        guard data.count<=1024*1024,let object=try JSONSerialization.jsonObject(with:data) as? [String:Any],let events=object["hooks"] as? [String:Any] else{throw ExtensionFiles.failure("hooks.json richiede un oggetto hooks.")}
        var result=[StudioHook]()
        for event in events.keys.sorted() {
            guard Self.events.contains(event),let groups=events[event] as? [[String:Any]] else{throw ExtensionFiles.failure("Evento hook non supportato: \(event)")}
            for group in groups {
                let matcher=group["matcher"] as? String
                if let matcher,!matcher.isEmpty,matcher != "*" {_ = try NSRegularExpression(pattern:matcher)}
                guard let handlers=group["hooks"] as? [[String:Any]] else{throw ExtensionFiles.failure("Gruppo hook senza handlers.")}
                for handler in handlers {
                    let type=handler["type"] as? String ?? "command"
                    guard ["command","mcp_tool"].contains(type) else{throw ExtensionFiles.failure("Handler \(type) non supportato; usa command o mcp_tool.")}
                    var hook=StudioHook(name:handler["statusMessage"] as? String ?? event,event:event,command:handler["command"] as? String ?? "")
                    hook.matcher=matcher;hook.timeout=handler["timeout"] as? Double;hook.background=handler["async"] as? Bool;hook.handlerType=type
                    hook.server=handler["server"] as? String;hook.tool=handler["tool"] as? String
                    if let input=handler["input"] {hook.inputJSON=String(decoding:try JSONSerialization.data(withJSONObject:input,options:[.sortedKeys,.fragmentsAllowed]),as:UTF8.self)}
                    hook.additionalContextLimit=handler["additionalContextLimit"] as? Int;hook.bundleRoot=root;hook.dataRoot=dataRoot
                    guard (hook.timeout ?? 600)>0,(hook.timeout ?? 600)<=3600,hook.command.utf8.count<=32768 else{throw ExtensionFiles.failure("Timeout o comando hook non valido.")}
                    guard type=="command" ? !hook.command.isEmpty : !(hook.server ?? "").isEmpty && !(hook.tool ?? "").isEmpty else{throw ExtensionFiles.failure("Handler incompleto.")}
                    result.append(hook)
                    guard result.count<=128 else{throw ExtensionFiles.failure("Massimo 128 hook per pacchetto.")}
                }
            }
        }
        return result
    }
    static func toolName(_ action: WorkspaceAction?) -> String {
        guard let action else{return ""}
        switch action.name {case "run_command","run_interactive":return "Bash";case "write_file":return "apply_patch";case "mcp_call":return "mcp__\(action.arguments["server"] ?? "")__\(action.arguments["tool"] ?? "")";default:return action.name}
    }
    static func template(_ value: Any, input: [String:Any]) -> Any {
        if let string=value as? String {
            func resolve(_ path: String) -> Any? {path.split(separator:".").reduce(input as Any?){current,key in (current as? [String:Any])?[String(key)]}}
            if string.hasPrefix("${"),string.hasSuffix("}"),string.filter({$0=="$"}).count==1,let resolved=resolve(String(string.dropFirst(2).dropLast())) {return resolved}
            var output=string
            let pattern=try! NSRegularExpression(pattern:#"\$\{([^}]+)\}"#)
            for match in pattern.matches(in:string,range:NSRange(string.startIndex...,in:string)).reversed() {
                if let range=Range(match.range,in:output),let key=Range(match.range(at:1),in:string),let resolved=resolve(String(string[key])) {output.replaceSubrange(range,with:String(describing:resolved))}
            }
            return output
        }
        if let object=value as? [String:Any] {return object.mapValues{template($0,input:input)}}
        if let list=value as? [Any] {return list.map{template($0,input:input)}}
        return value
    }
    static func outcome(stdout: String, stderr: String, status: Int32, event: String) throws -> HookOutcome {
        var result=HookOutcome()
        if status==2 {
            let reason=stderr.isEmpty ? "Operazione bloccata dall’hook." : stderr
            if ["PreToolUse","UserPromptSubmit","PermissionRequest"].contains(event) {result.deny=reason;result.permission=false}
            else if event=="PostToolUse" {result.replacement=reason}
            else if ["Stop","SubagentStop"].contains(event) {result.continuation=reason}
            return result
        }
        guard status==0 else{throw ExtensionFiles.failure("Hook exit \(status): \(stderr)")}
        guard let object=(try? JSONSerialization.jsonObject(with:Data(stdout.utf8))) as? [String:Any] else {
            if ["SessionStart","UserPromptSubmit"].contains(event),!stdout.isEmpty {result.context.append(stdout)}
            return result
        }
        let specific=object["hookSpecificOutput"] as? [String:Any] ?? [:]
        if let declared=specific["hookEventName"] as? String,declared != event {throw ExtensionFiles.failure("hookEventName non corrispondente.")}
        if let context=specific["additionalContext"] as? String {result.context.append(context)}
        if let message=object["systemMessage"] as? String {result.context.append(message)}
        let reason=object["reason"] as? String ?? object["stopReason"] as? String ?? "Decisione dell’hook."
        if event=="PermissionRequest",let decision=specific["decision"] as? [String:Any] {
            guard decision["updatedInput"]==nil,decision["updatedPermissions"]==nil,decision["interrupt"]==nil else{throw ExtensionFiles.failure("Modifiche alle autorizzazioni non supportate.")}
            let behavior=decision["behavior"] as? String
            if behavior=="deny" {result.permission=false;result.deny=decision["message"] as? String ?? reason}
            else if behavior=="allow" {result.permission=true}
            else {throw ExtensionFiles.failure("Decisione PermissionRequest non valida.")}
        }
        if event=="PreToolUse" {
            if object["continue"] as? Bool==false {throw ExtensionFiles.failure("PreToolUse continue:false non supportato.")}
            switch specific["permissionDecision"] as? String {
            case "deny":result.deny=specific["permissionDecisionReason"] as? String ?? reason
            case "allow":
                if let updated=specific["updatedInput"] {guard let input=updated as? [String:String] else{throw ExtensionFiles.failure("updatedInput deve contenere argomenti stringa.")};result.updatedInput=input}
            case nil:break
            default:throw ExtensionFiles.failure("Decisione PreToolUse non supportata.")
            }
        }
        if object["decision"] as? String=="block" || object["continue"] as? Bool==false {
            if ["PreToolUse","UserPromptSubmit"].contains(event) {result.deny=reason}
            else if event=="PostToolUse" {result.replacement=reason}
            else if ["Stop","SubagentStop"].contains(event),object["continue"] as? Bool != false {result.continuation=reason}
        }
        return result
    }
}

/// Separate from ordinary workspace commands: JSON stdin, separate bounded stdout/stderr, configurable lifetime.
final class HookCommandRunner: @unchecked Sendable {
    private let lock=NSLock()
    private var process: Process?
    private var cancelled=false
    func cancel() {lock.lock();cancelled=true;let current=process;lock.unlock();if let current,current.isRunning {kill(-current.processIdentifier,SIGKILL);current.terminate()}}
    func run(_ command: String, input: Data, root: URL, environment: [String:String], timeout: Double, helper: URL?) async throws -> (String,String,Int32) {
        try await withTaskCancellationHandler(operation:{try await Task.detached { [self] in
            try runBlocking(command,input:input,root:root,environment:environment,timeout:timeout,helper:helper)
        }.value},onCancel:{self.cancel()})
    }
    private func runBlocking(_ command: String, input: Data, root: URL, environment: [String:String], timeout: Double, helper: URL?) throws -> (String,String,Int32) {
        let p=Process(),stdin=Pipe(),stdout=Pipe(),stderr=Pipe()
        if let helper {p.executableURL=helper;p.arguments=[command]}
        else {p.executableURL=URL(fileURLWithPath:"/bin/zsh");p.arguments=["-c",command]}
        p.currentDirectoryURL=root;p.environment=ProcessInfo.processInfo.environment.merging(environment,uniquingKeysWith:{$1})
        p.standardInput=stdin;p.standardOutput=stdout;p.standardError=stderr
        let out=HookBytes(),err=HookBytes()
        stdout.fileHandleForReading.readabilityHandler={out.append($0.availableData)}
        stderr.fileHandleForReading.readabilityHandler={err.append($0.availableData)}
        lock.lock();if cancelled {lock.unlock();throw CancellationError()}
        do {try p.run();process=p;lock.unlock()} catch {lock.unlock();throw error}
        let timer=DispatchWorkItem{self.cancel()};DispatchQueue.global().asyncAfter(deadline:.now()+timeout,execute:timer)
        defer {timer.cancel();stdout.fileHandleForReading.readabilityHandler=nil;stderr.fileHandleForReading.readabilityHandler=nil}
        do {try stdin.fileHandleForWriting.write(contentsOf:input);try stdin.fileHandleForWriting.close()}
        catch {cancel();throw error}
        p.waitUntilExit();timer.cancel()
        if helper != nil {kill(-p.processIdentifier,SIGKILL)}
        stdout.fileHandleForReading.readabilityHandler=nil;stderr.fileHandleForReading.readabilityHandler=nil
        out.append(stdout.fileHandleForReading.readDataToEndOfFile());err.append(stderr.fileHandleForReading.readDataToEndOfFile())
        lock.lock();let stopped=cancelled;process=nil;lock.unlock()
        if stopped {throw ExtensionFiles.failure("Hook interrotto o timeout scaduto.")}
        return (out.string,err.string,p.terminationStatus)
    }
}
private final class HookBytes: @unchecked Sendable {
    private let lock=NSLock();private var data=Data()
    func append(_ bytes: Data){lock.lock();defer{lock.unlock()};data.append(bytes.prefix(max(0,256*1024-data.count)))}
    var string: String {lock.lock();defer{lock.unlock()};return String(decoding:data,as:UTF8.self)}
}

extension StudioStore {
    static var commandHelper: URL? {
        let url=Bundle.main.bundleURL.appendingPathComponent("Contents/Helpers/suc-studio-command")
        return FileManager.default.isExecutableFile(atPath:url.path) ? url:nil
    }
    func invokeHooks(_ event: String, session: ChatSession, action: WorkspaceAction? = nil, extra: [String:Any] = [:], helperURL: URL? = nil) async -> HookOutcome {
        let event=HookDefinition.event(event),tool=HookDefinition.toolName(action)
        let matchValue=action != nil ? tool : (extra["source"] as? String ?? extra["agent_type"] as? String ?? extra["trigger"] as? String ?? extra["reason"] as? String ?? "")
        let hooks=(archive.settings.hooks ?? []).filter{$0.enabled && HookDefinition.matches($0,event:event,tool:matchValue)}
        var combined=HookOutcome()
        var input:[String:Any]=["hook_event_name":event,"session_id":session.id.uuidString,"transcript_path":NSNull(),"cwd":session.workspacePath ?? FileManager.default.homeDirectoryForCurrentUser.path,"model":chatModel,"turn_id":hookTurnIDs[session.id] ?? session.id.uuidString,"permission_mode":session.approvalMode ?? "manual"]
        if let action {input["tool_name"]=tool;input["suc_tool_name"]=action.name;input["tool_input"]=action.arguments;input["tool_use_id"]=hookToolIDs[session.id] ?? UUID().uuidString}
        input.merge(extra,uniquingKeysWith:{$1})
        for hook in hooks {
            if Task.isCancelled {break}
            guard HookDefinition.trusted(hook) else{appendLog("Hook non autorizzato o modificato, saltato: "+hook.name);continue}
            if hook.background==true,event != "SessionEnd" {
                guard (backgroundHooks[session.id]?.count ?? 0)<8 else{appendLog("Limite 8 hook async per sessione.");continue}
                let taskID=UUID()
                backgroundHooks[session.id,default:[:]][taskID]=Task {let outcome=await executeHook(hook,event:event,session:session,input:input,helperURL:helperURL);if !Task.isCancelled {hookContexts[session.id,default:[]]+=outcome.context};backgroundHooks[session.id]?.removeValue(forKey:taskID)}
                continue
            }
            let result=await executeHook(hook,event:event,session:session,input:input,helperURL:helperURL)
            combined.context+=result.context
            if let deny=result.deny {combined.deny=deny}
            if result.permission==false {combined.permission=false} else if combined.permission==nil {combined.permission=result.permission}
            if let updated=result.updatedInput {combined.updatedInput=updated}
            if let replacement=result.replacement {combined.replacement=replacement}
            if let continuation=result.continuation {combined.continuation=continuation}
        }
        return combined
    }
    private func executeHook(_ hook: StudioHook, event: String, session: ChatSession, input: [String:Any], helperURL: URL?) async -> HookOutcome {
        let id=UUID().uuidString;operationJournal.update(id:id,title:"Hook · "+hook.name,state:"In corso",sessionID:session.id)
        do {
            let result:HookOutcome
            if hook.handlerType=="mcp_tool" {
                guard event != "SessionEnd",let config=archive.settings.mcpServers?.first(where:{$0.enabled && $0.name==hook.server}),let client=mcpConnections[config.id],let tool=hook.tool else{throw ExtensionFiles.failure("MCP hook non collegato o evento non supportato.")}
                let template=try JSONSerialization.jsonObject(with:Data((hook.inputJSON ?? "{}").utf8))
                let arguments=HookDefinition.template(template,input:input) as? [String:Any] ?? [:]
                guard let selected=mcpTools.first(where:{$0.serverID==config.id && $0.name==tool}) else{throw ExtensionFiles.failure("Strumento MCP hook non disponibile.")}
                let text=try await client.call(selected,arguments:arguments,timeout:hook.timeout ?? 600)
                result=try HookDefinition.outcome(stdout:text,stderr:"",status:0,event:event)
            } else {
                let root=URL(fileURLWithPath:input["cwd"] as? String ?? NSHomeDirectory())
                var env=["SUC_HOOK_EVENT":hook.event,"SUC_SESSION_ID":session.id.uuidString,"SUC_WORKSPACE":root.path,"SUC_TOOL_NAME":input["suc_tool_name"] as? String ?? input["tool_name"] as? String ?? "","SUC_TOOL_ARGUMENTS":String(decoding:try JSONSerialization.data(withJSONObject:input["tool_input"] ?? [:]),as:UTF8.self)]
                if let root=hook.bundleRoot {env["PLUGIN_ROOT"]=root;env["CLAUDE_PLUGIN_ROOT"]=root}
                if let root=hook.dataRoot {env["PLUGIN_DATA"]=root;env["CLAUDE_PLUGIN_DATA"]=root}
                let helper=helperURL ?? Self.commandHelper
                let timeout=["SessionEnd","Interrupt"].contains(event) ? min(3,hook.timeout ?? 1) : min(3600,hook.timeout ?? 600)
                let output=try await HookCommandRunner().run(hook.command,input:JSONSerialization.data(withJSONObject:input),root:root,environment:env,timeout:timeout,helper:helper)
                result=try HookDefinition.outcome(stdout:output.0,stderr:output.1,status:output.2,event:event)
            }
            var bounded=result
            let limit=max(0,hook.additionalContextLimit ?? 2500)*4
            bounded.context=try result.context.map { text in
                guard limit>0,text.count>limit else{return text}
                let url=try ExtensionFiles.storage("HookOutput").appendingPathComponent(UUID().uuidString+".txt")
                try text.write(to:url,atomically:true,encoding:.utf8)
                return String(text.prefix(limit/2))+"\n[Output completo: \(url.path)]\n"+String(text.suffix(limit/2))
            }
            operationJournal.update(id:id,state:"Completata");return bounded
        } catch {operationJournal.update(id:id,state:"Errore");appendLog("Hook \(hook.name): \(error.localizedDescription)");return HookOutcome()}
    }
    func reviewHook(_ hook: StudioHook) {
        var copy=hook;copy.trustedDigest=HookDefinition.digest(copy);copy.enabled=true
        var items=archive.settings.hooks ?? [];if let i=items.firstIndex(where:{$0.id==copy.id}){items[i]=copy}else{items.append(copy)}
        archive.settings.hooks=items;scheduleSave()
    }
    func importHooks() {
        let panel=NSOpenPanel();panel.message="Scegli hooks.json (importato disattivato, da revisionare)"
        guard panel.runModal() == .OK,let url=panel.url else{return}
        do {let hooks=try HookDefinition.decode(Data(try ExtensionFiles.read(url,limit:1024*1024).utf8));archive.settings.hooks=(archive.settings.hooks ?? [])+hooks;scheduleSave()}catch{self.error=error.localizedDescription}
    }
}
