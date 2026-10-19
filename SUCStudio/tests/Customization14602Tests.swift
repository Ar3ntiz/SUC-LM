import Foundation

@main struct Customization14602Tests {
    @MainActor static func main() async throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent("suc-ext-test-"+UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer{try? FileManager.default.removeItem(at:root)}
        func write(_ path: String,_ text: String) throws {
            let url=root.appendingPathComponent(path);try FileManager.default.createDirectory(at:url.deletingLastPathComponent(),withIntermediateDirectories:true);try text.write(to:url,atomically:true,encoding:.utf8)
        }
        try write("plugin.json",#"{"name":"test-plugin","version":"1.0.0","description":"Test package","extensions":{"com.openai":{"onboardingSkill":"./skills/review/SKILL.md"}}}"#)
        try write("skills/review/SKILL.md","---\nname: review\ndescription: >\n  Review source code\n  when requested.\n---\nPRIVATE_BODY")
        try write("skills/review/references/checklist.md","Check concrete failures.")
        try write("skills/review/agents/openai.yaml","interface:\n  display_name: Review code\n  default_prompt: Review this project.\npolicy:\n  allow_implicit_invocation: false\n")
        try write("mcp.json",#"{"mcpServers":{"local":{"type":"stdio","command":"python3","args":["${PLUGIN_ROOT}/server.py"],"env":{"CACHE":"${PLUGIN_DATA}"}},"remote":{"type":"streamable-http","url":"http://127.0.0.1:8081/mcp"}}}"#)
        let hookJSON=#"{"hooks":{"PreToolUse":[{"matcher":"^apply_patch$","hooks":[{"type":"command","command":"printf '%s' '{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"permissionDecision\":\"allow\",\"updatedInput\":{\"path\":\"safe.txt\",\"content\":\"rewritten\"}}}'","timeout":10}]}]}}"#
        try write("hooks/hooks.json",hookJSON)
        let package=try PluginPackage.load(root,dataRoot:root.appendingPathComponent("data"))
        precondition(package.format=="Portable 1.0" && package.skills.count==1 && package.servers.count==2 && package.hooks.count==1)
        var skill=package.skills[0];skill.enabled=true
        precondition(skill.name=="review" && skill.description=="Review source code when requested." && skill.displayName=="Review code" && skill.allowImplicitInvocation==false)
        var settings=AppSettings();settings.skills=[skill]
        let prompt=PromptComposer.compose(settings:settings,session:ChatSession())
        precondition(!prompt.contains("PRIVATE_BODY") && prompt.contains("solo su richiesta"))
        precondition(SkillDocument.explicitInstructions("Usa @review.",skills:[skill]).contains("PRIVATE_BODY"))
        precondition(SkillDocument.explicitInstructions("review ordinary prose",skills:[skill]).isEmpty)
        let listed=try SkillDocument.resource(WorkspaceAction(name:"skill_resources",arguments:["id":skill.id.uuidString]),skills:[skill]);precondition(listed.contains("references/checklist.md"))
        let resource=try SkillDocument.resource(WorkspaceAction(name:"skill_resource",arguments:["id":skill.id.uuidString,"path":"references/checklist.md"]),skills:[skill]);precondition(resource=="Check concrete failures.")
        do{_ = try ExtensionFiles.resolve("../outside",root:root);preconditionFailure("Traversal allowed")}catch{}
        try FileManager.default.createSymbolicLink(at:root.appendingPathComponent("outside"),withDestinationURL:URL(fileURLWithPath:"/etc/passwd"))
        do{_ = try ExtensionFiles.files(root);preconditionFailure("External symlink allowed")}catch{}
        try FileManager.default.removeItem(at:root.appendingPathComponent("outside"))
        precondition(package.servers[0].executable=="/usr/bin/env" && package.servers[0].arguments[1]==root.path+"/server.py")
        precondition(package.servers[0].environment?["CACHE"]==root.path+"/data")
        let hook=package.hooks[0]
        precondition(!HookDefinition.trusted(hook))
        var trusted=hook;trusted.trustedDigest=HookDefinition.digest(trusted);trusted.enabled=true
        precondition(HookDefinition.trusted(trusted));trusted.command+=" ";precondition(!HookDefinition.trusted(trusted))
        precondition(HookDefinition.matches(hook,event:"PreToolUse",tool:"apply_patch"))
        precondition(!HookDefinition.matches(hook,event:"PreToolUse",tool:"Bash"))
        let templated=HookDefinition.template(["value":"${tool_input.count}","prompt":"Text ${prompt}"],input:["tool_input":["count":3],"prompt":"hello"]) as! [String:Any]
        precondition(templated["value"] as? Int==3 && templated["prompt"] as? String=="Text hello")
        let denied=try HookDefinition.outcome(stdout:"",stderr:"denied",status:2,event:"PreToolUse");precondition(denied.deny=="denied")
        let permission=try HookDefinition.outcome(stdout:#"{"hookSpecificOutput":{"hookEventName":"PermissionRequest","decision":{"behavior":"allow"}}}"#,stderr:"",status:0,event:"PermissionRequest");precondition(permission.permission==true)
        let feedback=try HookDefinition.outcome(stdout:#"{"decision":"block","reason":"Check result"}"#,stderr:"",status:0,event:"PostToolUse");precondition(feedback.replacement=="Check result")
        let market=try PackageMarketplace.decode(Data(#"{"name":"demo","plugins":[{"name":"test-plugin","source":{"source":"local","path":"./plugin"},"policy":{"installation":"NOT_AVAILABLE"}}]}"#.utf8),root:root)
        precondition(market.count==1 && market[0].unavailable)
        do{_ = try PackageMarketplace.decode(Data(#"{"plugins":[{"name":"escape","source":"../x"}]}"#.utf8),root:root);preconditionFailure("Marketplace traversal allowed")}catch{}
        let helper=URL(fileURLWithPath:CommandLine.arguments[1])
        let output=try await HookCommandRunner().run("/usr/bin/python3 -c 'import json,sys; v=json.load(sys.stdin); print(v[\"prompt\"]); print(\"diagnostic\",file=sys.stderr)'",input:Data(#"{"prompt":"received JSON"}"#.utf8),root:root,environment:[:],timeout:10,helper:helper)
        precondition(output.0.contains("received JSON") && output.1.contains("diagnostic") && output.2==0)
        let start=Date()
        do{_ = try await HookCommandRunner().run("sleep 10",input:Data("{}".utf8),root:root,environment:[:],timeout:0.1,helper:helper);preconditionFailure("Timeout ignored")}catch{}
        precondition(Date().timeIntervalSince(start)<3)
        let store=StudioStore(storageURL:root.appendingPathComponent("archive.json"));store.schedulingEnabled=false
        store.mutateSelected{$0.workspacePath=root.path;$0.agentEnabled=true;$0.approvalMode="automatic";$0.allowCommands=false}
        var rewrite=hook;rewrite.enabled=true;rewrite.trustedDigest=HookDefinition.digest(rewrite);store.archive.settings.hooks=[rewrite]
        let session=store.selected!,message=ChatMessage(role:"assistant",text:#"{"suc_action":{"name":"write_file","arguments":{"path":"original.txt","content":"original"}}}"#)
        store.mutateSelected{$0.messages=[message]};store.handleAgentResponse(sessionID:session.id,messageID:message.id)
        for _ in 0..<200 {if !store.agentRuns.contains(session.id){break};try await Task.sleep(nanoseconds:20_000_000)}
        precondition(FileManager.default.fileExists(atPath:root.appendingPathComponent("safe.txt").path))
        precondition(!FileManager.default.fileExists(atPath:root.appendingPathComponent("original.txt").path))
        let saved=try String(contentsOf:root.appendingPathComponent("safe.txt"),encoding:.utf8);precondition(saved=="rewritten")
        var deny=StudioHook(name:"Deny",event:"PreToolUse",command:"printf 'not allowed' >&2; exit 2");deny.enabled=true;deny.trustedDigest=HookDefinition.digest(deny);store.archive.settings.hooks=[deny]
        let result=await store.invokeHooks("PreToolUse",session:session,action:WorkspaceAction(name:"write_file",arguments:[:]),helperURL:helper);precondition(result.deny=="not allowed")
        var allow=StudioHook(name:"Allow",event:"PermissionRequest",command:#"printf '%s' '{"hookSpecificOutput":{"hookEventName":"PermissionRequest","decision":{"behavior":"allow"}}}'"#);allow.enabled=true;allow.trustedDigest=HookDefinition.digest(allow);store.archive.settings.hooks=[allow]
        var manual=session;manual.approvalMode="manual"
        let approved=await store.approve(WorkspaceAction(name:"read_file",arguments:["path":"safe.txt"]),session:manual,root:root);precondition(approved)
        precondition(store.approvals.isEmpty)
        var background=StudioHook(name:"Async context",event:"UserPromptSubmit",command:"printf 'background ready'")
        background.background=true;background.enabled=true;background.trustedDigest=HookDefinition.digest(background);store.archive.settings.hooks=[background]
        let asynchronous=await store.invokeHooks("UserPromptSubmit",session:session,extra:["prompt":"sample"],helperURL:helper)
        precondition(asynchronous.context.isEmpty && asynchronous.deny==nil)
        for _ in 0..<200 {if store.hookContexts[session.id]?.contains("background ready")==true{break};try await Task.sleep(nanoseconds:20_000_000)}
        precondition(store.hookContexts[session.id]?.contains("background ready")==true)
        let fixture=URL(fileURLWithPath:CommandLine.arguments[2])
        let config=MCPConfiguration(name:"fixture",executable:"/usr/bin/python3",arguments:[fixture.path],enabled:true)
        let connection=MCPConnection(config,credentials:[:]);defer{connection.stop()}
        store.archive.settings.mcpServers=[config];store.mcpConnections[config.id]=connection;store.mcpTools=try await connection.start()
        var mcpHook=StudioHook(name:"MCP context",event:"UserPromptSubmit");mcpHook.handlerType="mcp_tool";mcpHook.server="fixture";mcpHook.tool="echo";mcpHook.inputJSON=#"{"text":"${prompt}"}"#;mcpHook.enabled=true;mcpHook.trustedDigest=HookDefinition.digest(mcpHook);store.archive.settings.hooks=[mcpHook]
        let echoed=await store.invokeHooks("UserPromptSubmit",session:session,extra:["prompt":"hook MCP works"])
        precondition(echoed.context.joined().contains("hook MCP works"))
        var pendingPermission=mcpHook;pendingPermission.event="PermissionRequest";pendingPermission.inputJSON=#"{"text":"waiting","delay":true,"delay_seconds":5}"#
        pendingPermission.trustedDigest=HookDefinition.digest(pendingPermission);store.archive.settings.hooks=[pendingPermission]
        var waitingSession=session;waitingSession.approvalMode="manual"
        let cancellationStart=Date()
        let cancelledCall=Task {await store.approve(WorkspaceAction(name:"read_file",arguments:["path":"safe.txt"]),session:waitingSession,root:root)}
        try await Task.sleep(nanoseconds:50_000_000);cancelledCall.cancel()
        let cancelledApproval=await cancelledCall.value
        precondition(!cancelledApproval && store.approvals.isEmpty)
        precondition(Date().timeIntervalSince(cancellationStart)<0.8);connection.stop()
        let alreadyCancelled=Task {await store.approve(WorkspaceAction(name:"read_file",arguments:[:]),session:waitingSession,root:root)}
        alreadyCancelled.cancel();let immediate=await alreadyCancelled.value;precondition(!immediate && store.approvals.isEmpty)
        let component=StudioSkill(name:"component",description:"Test",instructions:"test")
        let installed=InstalledPlugin(manifest:PluginManifest(formatVersion:1,identifier:"component-plugin",name:"component-plugin",version:"1.0"),source:"test",skills:[component],servers:[],hooks:[])
        store.archive.settings.plugins=[installed];store.setPlugin(installed,enabled:true,review:false)
        precondition(store.archive.settings.skills?.first{$0.id==component.id}?.enabled==true)
        store.archive.settings.skills?[0].enabled=false
        store.setPlugin(installed,enabled:false,review:false);store.setPlugin(installed,enabled:true,review:false)
        precondition(store.archive.settings.skills?.first{$0.id==component.id}?.enabled==false)
        for sample in HookSamples.all where !sample.command.contains("git") && !sample.command.contains("PLUGIN_DATA") {
            let output=try await HookCommandRunner().run(sample.command,input:Data(#"{"stop_hook_active":false}"#.utf8),root:root,environment:[:],timeout:10,helper:helper)
            _ = try HookDefinition.outcome(stdout:output.0,stderr:output.1,status:output.2,event:sample.event)
        }
        let sampleRoot=URL(fileURLWithPath:CommandLine.arguments[3]);let bundled=try PluginPackage.load(sampleRoot,dataRoot:root.appendingPathComponent("sample-data"))
        let sampleOutput=try await HookCommandRunner().run(bundled.hooks[0].command,input:Data("{}".utf8),root:root,environment:[:],timeout:10,helper:helper)
        let sampleContext=try HookDefinition.outcome(stdout:sampleOutput.0,stderr:sampleOutput.1,status:sampleOutput.2,event:"SessionStart");precondition(sampleContext.context.first?.hasPrefix("Consulta README")==true)
        var startHook=StudioHook(name:"Start context",event:"SessionStart",command:"printf 'START_CONTEXT'")
        startHook.matcher="^startup$";startHook.enabled=true;startHook.trustedDigest=HookDefinition.digest(startHook)
        var promptHook=StudioHook(name:"Prompt context",event:"UserPromptSubmit",command:"printf 'PROMPT_CONTEXT'")
        promptHook.enabled=true;promptHook.trustedDigest=HookDefinition.digest(promptHook)
        store.archive.settings.hooks=[startHook,promptHook];store.archive.settings.globalSystemPrompt="GLOBAL_CONTEXT";store.archive.settings.modelIOLogEnabled=true
        store.archive.settings.chatBackend="remote";store.archive.settings.remote=RemoteConfiguration(endpoint:"http://127.0.0.1:1/v1",model:"fixture")
        store.newSession();let hookSession=store.selected!.id;store.mutateSelected{$0.draft="hello";$0.allowWeb=false}
        store.sendMessage()
        for _ in 0..<200 {if store.modelIOLog.entries.contains(where:{$0.direction=="INPUT" && $0.text.contains("PROMPT_CONTEXT")}){break};try await Task.sleep(nanoseconds:20_000_000)}
        let wire=store.modelIOLog.entries.last{$0.direction=="INPUT"}!
        let request=try JSONSerialization.jsonObject(with:Data(wire.text.utf8)) as! [String:Any]
        let messages=request["messages"] as! [[String:Any]]
        precondition(messages.filter{$0["role"] as? String=="system"}.count==1)
        let finalPrompt=messages[0]["content"] as! String
        precondition(finalPrompt.contains("GLOBAL_CONTEXT") && finalPrompt.contains("START_CONTEXT") && finalPrompt.contains("PROMPT_CONTEXT"))
        precondition(store.selected!.messages.last(where:{$0.generation != nil})?.generation?.systemPrompt==finalPrompt)
        precondition(store.hookSessions.contains(hookSession))
        store.cancelSelected()
        for mask in 1...7 {
            var builder=PluginBuilderDraft();builder.name="combined-\(mask)"
            builder.includeSkill=mask & 1 != 0;builder.includeHook=mask & 2 != 0;builder.includeMCP=mask & 4 != 0
            let exported=try builder.export(to:root)
            let combined=try PluginPackage.load(exported,dataRoot:root.appendingPathComponent("builder-data"))
            precondition(combined.skills.count==(builder.includeSkill ? 1:0))
            precondition(combined.hooks.count==(builder.includeHook ? 1:0))
            precondition(combined.servers.count==(builder.includeMCP ? 1:0))
            precondition(FileManager.default.fileExists(atPath:exported.appendingPathComponent("mcp.json").path)==builder.includeMCP)
            precondition(FileManager.default.fileExists(atPath:exported.appendingPathComponent("hooks/hooks.json").path)==builder.includeHook)
        }
        var draft=PluginBuilderDraft();draft.skill+="\nCUSTOM_SKILL";draft.hook="CUSTOM_HOOK";draft.mcp="CUSTOM_MCP"
        draft.includeSkill=false;draft.includeHook=true;draft.includeMCP=true
        draft.includeSkill=true;draft.includeHook=false;draft.includeMCP=false
        precondition(draft.skill.contains("CUSTOM_SKILL") && draft.hook=="CUSTOM_HOOK" && draft.mcp=="CUSTOM_MCP")
        draft.reset("hook");precondition(draft.skill.contains("CUSTOM_SKILL") && draft.mcp=="CUSTOM_MCP")
        draft.name="skill-only-invalid-unselected";_ = try draft.export(to:root)
        draft.name="invalid-combined";draft.includeMCP=true
        do{_ = try draft.export(to:root);preconditionFailure("Invalid selected MCP exported")}catch{}
        precondition(!FileManager.default.fileExists(atPath:root.appendingPathComponent(draft.name).path))
        draft.mcp=#"{"mcpServers":{"invalid":{"type":"sse","url":"http://127.0.0.1:8081/mcp"}}}"#
        do{_ = try draft.export(to:root);preconditionFailure("Invalid MCP transport exported")}catch{}
        precondition(!FileManager.default.fileExists(atPath:root.appendingPathComponent(draft.name).path))
        var renamed=PluginBuilderDraft();renamed.name="renamed-skill";renamed.skill=renamed.skill.replacingOccurrences(of:"name: review-code",with:"name: project-review")
        let renamedRoot=try renamed.export(to:root);precondition(FileManager.default.fileExists(atPath:renamedRoot.appendingPathComponent("skills/project-review/SKILL.md").path))
        draft.includeSkill=false;draft.includeMCP=false
        do{_ = try draft.documents();preconditionFailure("Empty plugin exported")}catch{}
        var existing=PluginBuilderDraft();existing.name="combined-7"
        let marker=root.appendingPathComponent("combined-7/keep.txt");try "keep".write(to:marker,atomically:true,encoding:.utf8)
        do{_ = try existing.export(to:root);preconditionFailure("Existing package overwritten")}catch{}
        let kept=try String(contentsOf:marker,encoding:.utf8);precondition(kept=="keep")
        let remaining=try FileManager.default.contentsOfDirectory(atPath:root.path);precondition(remaining.allSatisfy{!$0.hasPrefix(".suc-plugin-")})
        print("PluginBuilder: all 7 component combinations round-trip; independent drafts/examples, selected-only validation, atomic failure, existing folder protection passed")
        print("Customization14602Tests: portable packages, progressive skills/metadata/resources, explicit calls, MCP placeholders, hook trust/matchers/templates/decisions, JSON stdin, timeout, actual tool rewrite, permission hook, async context, MCP/approval cancellation, HTTP-independent lifecycle prompt assembly/I-O and marketplace confinement passed")
    }
}
