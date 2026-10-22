import AppKit
import Foundation
@main struct SteeringTests {
    @MainActor static func wait(_ label:String,_ check:()->Bool) async throws {
        for _ in 0..<600 {if check(){return};try await Task.sleep(nanoseconds:20_000_000)}
        fatalError("Timeout: "+label)
    }
    @MainActor static func main() async throws {
        _=NSApplication.shared;NSApp.setActivationPolicy(.accessory)
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
        let server=Process(),pipe=Pipe();server.executableURL=URL(fileURLWithPath:"/usr/bin/python3");server.arguments=[CommandLine.arguments[1]];server.standardOutput=pipe
        try server.run();defer{server.terminate()}
        let port=String(decoding:pipe.fileHandleForReading.availableData,as:UTF8.self).trimmingCharacters(in:.whitespacesAndNewlines)
        let helper=URL(fileURLWithPath:CommandLine.arguments[2])
        for model in ["before","during","approval"] {
            let work=root.appendingPathComponent(model);try FileManager.default.createDirectory(at:work,withIntermediateDirectories:true)
            let store=StudioStore(storageURL:root.appendingPathComponent(model+".json"),commandHelperURL:helper);store.schedulingEnabled=false
            store.archive.settings.chatBackend="remote";store.archive.settings.remote=RemoteConfiguration(endpoint:"http://127.0.0.1:\(port)/v1",model:model)
            store.mutateSelected{$0.workspacePath=work.path;$0.agentEnabled=true;$0.allowCommands=true;$0.approvalMode=model=="approval" ? "manual":"automatic";$0.draft="START"}
            var hook=StudioHook(name:"Prompt fixture",event:"UserPromptSubmit",command:#"/usr/bin/python3 -c 'import json,sys; p=json.load(sys.stdin); open("hook-prompts.jsonl","a").write(json.dumps(p)+"\n")'"#,enabled:true)
            hook.trustedDigest=HookDefinition.digest(hook);store.archive.settings.hooks=[hook]
            let sid=store.selected!.id;store.sendMessage()
            if model=="before" {try await wait("decode started"){store.selected?.messages.last?.text.contains("Preparing.")==true}}
            else if model=="during" {try await wait("command running"){store.commandJobs.running(sid)}}
            else {try await wait("approval displayed"){!store.approvals.isEmpty}}
            store.activeSkillInstructions[sid]="ORIGINAL_SKILL_SENTINEL"
            let actionsBefore=store.agentIterations[sid] ?? 0
            store.mutateSelected{$0.draft="CORRECTION: use the new instruction";$0.draftAttachments=[ChatAttachment(name:"update.txt",text:"ATTACHMENT_SENTINEL",originalBytes:19)]};store.sendMessage()
            let queuedID=store.selected!.queuedMessages!.first!.id
            precondition(store.selected!.queuedMessages?.count==1)
            store.mutateSelected{$0.draft="DRAFT_NOT_SENT"}
            if model=="approval" {try await wait("revised action approval"){store.approvals.first?.action.arguments["path"]=="correct.txt"};precondition(store.approvals[0].action.arguments["path"]=="correct.txt");store.resolveApproval(store.approvals[0].id,allowed:true)}
            let expected=model=="during" ? "CORRECTED_AFTER_TOOL":"CORRECTED_FINAL"
            try await wait("corrected result"){store.selected!.messages.contains{$0.text==expected && $0.state=="complete"}}
            precondition(!FileManager.default.fileExists(atPath:work.appendingPathComponent("wrong.txt").path))
            if model != "during" {let saved=try String(contentsOf:work.appendingPathComponent("correct.txt"),encoding:.utf8);precondition(saved=="CORRECTED_CONTENT")}
            let received=store.selected!.messages.filter{$0.id==queuedID};precondition(received.count==1 && received[0].attachments?.first?.text=="ATTACHMENT_SENTINEL")
            precondition(store.selected!.queuedMessages?.isEmpty==true && store.selected!.draft=="DRAFT_NOT_SENT")
            precondition((store.agentIterations[sid] ?? 0)>=actionsBefore && store.agentIterations[sid]==(model=="approval" ? 2:1))
            if model=="during" {precondition(store.commandJobs.jobs.count==1 && store.commandJobs.jobs[0].snapshot().output=="TOOL_SENTINEL")}
            let promptLog=try String(contentsOf:work.appendingPathComponent("hook-prompts.jsonl"),encoding:.utf8)
            let prompts=try promptLog.split(separator:"\n").map{try JSONSerialization.jsonObject(with:Data($0.utf8)) as! [String:Any]}
            precondition(prompts.count==2 && prompts[0]["prompt"] as? String=="START" && (prompts[1]["prompt"] as? String)?.hasPrefix("CORRECTION")==true)
            precondition(prompts[0]["turn_id"] as? String==prompts[1]["turn_id"] as? String)
            precondition(store.selected!.messages.last(where:{!$0.isUIReport})?.generation?.systemPrompt.contains("ORIGINAL_SKILL_SENTINEL")==true)
            let archived=try JSONDecoder().decode(Archive.self,from:JSONEncoder().encode(store.archive));precondition(archived.sessions[0].messages.filter{$0.id==queuedID}.count==1)
            print("PASS \(model): correction at safe boundary, original operation not repeated, attachment/draft/skill and prompt hooks preserved, one delivery")
        }
        // Context failure must retain the completed tool result and the unconsumed update.
        let limited=StudioStore(storageURL:root.appendingPathComponent("limited.json"));limited.schedulingEnabled=false
        limited.archive.settings.chatBackend="remote";limited.archive.settings.remote=RemoteConfiguration(endpoint:"http://127.0.0.1:\(port)/v1",model:"before")
        limited.mutateSelected{$0.contextWindow=1024;$0.autoCompact=false;$0.messages=[ChatMessage(role:"user",text:"START")];$0.queuedMessages=[ChatMessage(role:"user",text:String(repeating:"long-update ",count:1000))]}
        let tool="COMPLETED_TOOL_SENTINEL";limited.sendMessage(sessionID:limited.selected!.id,toolResult:tool)
        precondition(limited.selected!.queuedMessages?.count==1 && limited.selected!.queuePaused==true)
        precondition(limited.selected!.messages.filter{$0.kind=="tool_result" && $0.text==tool}.count==1)
        print("PASS context failure: completed tool result and pending update retained; no rerun")
        print("Steering tests PASS")
    }
}
