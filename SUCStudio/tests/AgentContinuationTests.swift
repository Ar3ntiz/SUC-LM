import Foundation

@main struct AgentContinuationTests {
    @MainActor static func wait(_ label:String,_ condition:()->Bool) async throws {
        for _ in 0..<600 {if condition(){return};try await Task.sleep(nanoseconds:50_000_000)}
        preconditionFailure("Timed out: "+label)
    }
    @MainActor static func main() async throws {
        let wire=#"{"suc_action":{"name":"read_file","arguments":{"path":"fixture.txt"}}}"#
        for wrapper in ["<tool_call>\n"+wire,"<tool_call>\n"+wire+"\n</tool_call>","<tool_call>\n```json\n"+wire+"\n```\n</tool_call>"] {
            precondition(WorkspaceAction.parse(wrapper)?.arguments["path"]=="fixture.txt")
            for intro in ["","Leggo il file."] {
                let prefix=intro.isEmpty ? "":intro+"\n\n"
                for count in 1...wrapper.count {
                    let message=ChatMessage(role:"assistant",text:prefix+String(wrapper.prefix(count)),state:"generating")
                    precondition(message.chatParts.answer==intro,"Leaked wrapper at prefix \(count)")
                }
                let saved=ChatMessage(role:"assistant",text:prefix+wrapper,kind:"tool_call")
                precondition(saved.chatParts.answer==intro && saved.text==prefix+wrapper)
            }
        }
        precondition(WorkspaceAction.parse("```xml\n<tool_call>\n"+wire+"\n</tool_call>\n```")==nil)
        precondition(ChatMessage(role:"assistant",text:"```xml\n<tool_call>\nexample\n</tool_call>\n```").chatParts.answer.contains("<tool_call>"))
        precondition(ChatMessage(role:"assistant",text:"The marker is `<tool_call>`.").chatParts.answer=="The marker is `<tool_call>`.")
        precondition(WorkspaceAction.parse("<tool_call>\n"+wire+"\n"+wire+"\n</tool_call>")==nil)
        precondition(WorkspaceAction.parse("<tool_call>\n{\"name\":\"read_file\",\"arguments\":{}}\n</tool_call>")==nil)
        let missing=String(wire.dropLast())
        precondition(WorkspaceAction.parse(missing)==nil && WorkspaceAction.parse(missing,allowMissingOuterClose:true)?.arguments["path"]=="fixture.txt")
        for broken in [String(wire.dropLast(2)),"{\"suc_action\":{\"name\":\"write_file\",\"arguments\":{\"content\":\"unfinished",wire+missing,missing+",", "{\"suc_action\":{\"name\":\"read_file\"}"] {
            precondition(WorkspaceAction.parse(broken,allowMissingOuterClose:true)==nil)
        }
        for text in ["Ora cerco come REA carica la configurazione e se esiste un comando doctor.","Sto cercando la configurazione. Procedo con le ricerche.","Hai ragione, procedo subito con le ricerche.","I will check the configuration now.","Let me inspect the files.","I'll run the tests."] {precondition(AgentContinuation.isPromise(text),text)}
        for text in ["Il lavoro è completato.","Vuoi che proceda?","Attendo la tua conferma prima di procedere.","Non posso procedere senza accesso.","I need your permission to continue.","Procedo se confermi.","\"Procedo con le ricerche.\"","> Procedo con le ricerche.","```text\nProcedo con le ricerche.\n```",#"{"suc_action":{"name":"read_file","arguments":{"path":"x"}}}"#] {precondition(!AgentContinuation.isPromise(text),text)}
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
        try "VERIFIED_FIXTURE_CONTENT".write(to:root.appendingPathComponent("fixture.txt"),atomically:true,encoding:.utf8)
        let server=Process(),pipe=Pipe();server.executableURL=URL(fileURLWithPath:"/usr/bin/python3");server.arguments=[CommandLine.arguments[1]];server.standardOutput=pipe
        try server.run();defer{server.terminate()}
        let port=String(decoding:pipe.fileHandleForReading.availableData,as:UTF8.self).trimmingCharacters(in:.whitespacesAndNewlines)
        func store(_ model:String,agent:Bool=true)->StudioStore {
            let value=StudioStore(storageURL:root.appendingPathComponent(UUID().uuidString+".json"));value.schedulingEnabled=false
            value.archive.settings.chatBackend="remote";value.archive.settings.remote=RemoteConfiguration(endpoint:"http://127.0.0.1:\(port)/v1",model:model)
            value.mutateSelected{$0.workspacePath=agent ? root.path:nil;$0.agentEnabled=agent;$0.allowWeb=false;$0.approvalMode="automatic";$0.draft="inspect the fixture"}
            return value
        }
        let recovered=store("recover"),sid=recovered.selected!.id
        recovered.sendMessage();recovered.mutateSelected{$0.draft="next queued turn"};recovered.sendMessage()
        try await wait("real read and queued continuation"){!recovered.sessionTurnBusy(sid) && recovered.selected!.messages.last?.text=="Queued turn completed."}
        precondition(recovered.selected!.messages.contains{$0.text=="Verified fixture file read successfully."})
        precondition(recovered.selected!.messages.last?.text=="Queued turn completed.")
        precondition(recovered.selected!.actions?.filter{$0.name=="read_file" && $0.state=="completed"}.count==1)
        precondition(recovered.selected!.queuedMessages?.isEmpty==true && recovered.agentRepairs[sid]==0)
        let bounded=store("always-promise"),bid=bounded.selected!.id
        bounded.sendMessage();bounded.mutateSelected{$0.draft="next queued turn"};bounded.sendMessage()
        try await wait("bounded recovery stops"){bounded.selected!.queuePaused==true && !bounded.sessionTurnBusy(bid)}
        precondition(bounded.agentRepairs[bid]==2 && bounded.agentIterations[bid]==2)
        precondition(bounded.selected!.messages.filter{$0.kind=="tool_result" && $0.text.hasPrefix("Feedback harness")}.count==2)
        precondition(bounded.selected!.queuedMessages?.count==1 && (bounded.selected!.actions ?? []).isEmpty)
        let manual=store("recover"),mid=manual.selected!.id;manual.mutateSelected{$0.approvalMode="manual"};manual.sendMessage()
        manual.mutateSelected{$0.draft="next queued turn"};manual.sendMessage()
        try await wait("approval still required"){!manual.approvals.isEmpty}
        precondition(manual.selected!.actions?.isEmpty != false)
        manual.cancelSelected();try await wait("cancel recovery/approval"){!manual.sessionTurnBusy(mid)}
        precondition(manual.selected!.queuePaused==true && !manual.selected!.messages.contains{$0.text=="Verified fixture file read successfully."})
        let limited=store("always-promise"),lid=limited.selected!.id;limited.archive.settings.maxConsecutiveActions=1;limited.sendMessage();limited.mutateSelected{$0.draft="next queued turn"};limited.sendMessage()
        try await wait("action budget"){!limited.sessionTurnBusy(lid)}
        precondition(limited.agentRepairs[lid]==1 && limited.selected!.queuePaused==true)
        for (model,agent) in [("question",true),("final",true),("always-promise",false)] {
            let quiet=store(model,agent:agent),id=quiet.selected!.id;quiet.sendMessage()
            try await wait("legitimate final/no tools"){!quiet.sessionTurnBusy(id)}
            precondition(quiet.agentRepairs[id]==0 && !quiet.selected!.messages.contains{$0.kind=="tool_result"})
        }
        let writer=store("broken-write"),wid=writer.selected!.id
        writer.sendMessage();try await wait("corrected Markdown artifact"){!writer.sessionTurnBusy(wid)}
        precondition(writer.agentRepairs[wid]==1)
        let savedMarkdown=try String(contentsOf:root.appendingPathComponent("summary.md"),encoding:.utf8)
        precondition(savedMarkdown=="# Fixture summary\n\n```text\n\"quoted\" \\ path à🙂\n```\n")
        precondition(writer.selected!.artifacts?.count==1 && writer.selected!.actions?.filter{$0.name=="write_file" && $0.state=="completed"}.count==1)
        try FileManager.default.removeItem(at:root.appendingPathComponent("summary.md"))
        let outer=store("missing-outer"),oid=outer.selected!.id
        outer.sendMessage();try await wait("outer wrapper tolerance"){!outer.sessionTurnBusy(oid)}
        precondition(outer.agentRepairs[oid]==0 && outer.selected!.actions?.filter{$0.name=="write_file" && $0.state=="completed"}.count==1)
        precondition(outer.selected!.artifacts?.count==1)
        let outerText=try String(contentsOf:root.appendingPathComponent("summary.md"),encoding:.utf8)
        precondition(outerText==savedMarkdown)
        try FileManager.default.removeItem(at:root.appendingPathComponent("summary.md"))
        for model in ["always-invalid","mixed-repair","truncated-invalid"] {
            let invalid=store(model),id=invalid.selected!.id;invalid.sendMessage()
            try await wait("malformed request stops"){!invalid.sessionTurnBusy(id)}
            precondition(invalid.agentRepairs[id]==(model=="truncated-invalid" ? 0:2))
            precondition((invalid.selected!.actions ?? []).isEmpty && !FileManager.default.fileExists(atPath:root.appendingPathComponent("summary.md").path))
            if model != "truncated-invalid" {precondition(invalid.selected!.messages.last(where:{!$0.isUIReport})?.state=="error")}
        }
        let phase=store("final"),pid=phase.selected!.id;phase.sendMessage()
        let request=phase.requests.keys.first!
        phase.receive(["event":"started","id":request,"payload":[:]])
        phase.receive(["event":"metrics","id":request,"payload":["completion_tokens":0]])
        precondition(phase.activePhase=="Attesa remota")
        phase.receive(["event":"delta","id":request,"payload":["text":"first fragment"]])
        for _ in 0..<5 {phase.receive(["event":"metrics","id":request,"payload":["completion_tokens":0]]);precondition(phase.activePhase=="Decode")}
        phase.receive(["event":"metrics","id":"unrelated-request","payload":["completion_tokens":0]])
        precondition(phase.activePhase=="Decode")
        phase.receive(["event":"reasoning","id":request,"payload":["text":"reasoning"]])
        phase.receive(["event":"metrics","id":request,"payload":["completion_tokens":0]])
        precondition(phase.activePhase=="Ragionamento")
        phase.cancelSelected();try await wait("phase fixture cancelled"){!phase.sessionTurnBusy(pid)}
        print("PASS: IT/EN promise detection; real streamed recovery to authorized file read; FIFO; two-attempt cap/action budget; manual approval and Stop; final/question/no-tools exclusions.")
        print("PASS: zero/missing live completion counts do not revert decode/reasoning or remote wait; unrelated request metrics leave phase unchanged.")
        print("PASS: malformed Markdown action corrected by model to one real write/artifact; persistent/mixed invalid output bounded, no guessed execution; length truncation not retried.")
        print("PASS: every streamed prefix of open/closed/fenced tool_call wrappers withheld; commentary/raw archive and code examples preserved; only normally finished complete actions tolerate one missing outer brace.")
    }
}
