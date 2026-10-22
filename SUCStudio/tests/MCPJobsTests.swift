import Foundation
@main struct MCPJobsTests {
    @MainActor static func main() async throws {
        let fixture=CommandLine.arguments[1]
        let server=Process(),out=Pipe()
        server.executableURL=URL(fileURLWithPath:"/usr/bin/python3");server.arguments=[fixture,"--http"];server.standardOutput=out
        try server.run();defer{server.terminate()}
        let port=Int(String(decoding:out.fileHandleForReading.availableData,as:UTF8.self).trimmingCharacters(in:.whitespacesAndNewlines))!
        for transport in ["stdio","http"] {
            let config=MCPConfiguration(name:"Lifecycle fixture",transport:transport,endpoint:"http://127.0.0.1:\(port)/mcp",executable:"/usr/bin/python3",arguments:[fixture],enabled:true)
            let client=MCPConnection(config,credentials:[:]);defer{client.stop()}
            let tools=try await client.start(),tool=tools.first{$0.name=="echo"}!,stats=tools.first{$0.name=="stats"}!
            if !CommandLine.arguments.contains("--integration-only") {
            let jobs=MCPJobs(),owner=UUID(),other=UUID()
            let long=try jobs.start(client:client,tool:tool,input:["delay":31,"text":"unique-completion"],session:owner,record:UUID(),timeout:nil)
            try await jobs.wait(long,seconds:1);precondition(long.running)
            do {_ = try jobs.start(client:client,tool:tool,input:[:],session:owner,record:UUID(),timeout:nil);preconditionFailure("duplicate") } catch {}
            do {_ = try await jobs.manage(WorkspaceAction(name:"mcp_cancel",arguments:["job_id":long.id.uuidString]),session:other);preconditionFailure("cross-session cancellation") } catch {}
            let observer=Task{try await jobs.wait(long,seconds:60)}
            try await Task.sleep(nanoseconds:100_000_000);observer.cancel()
            do {try await observer.value;preconditionFailure("cancelled observation") }catch{}
            precondition(long.running)
            await long.task?.value
            precondition(long.state=="completed" && long.result=="unique-completion" && (long.finished!-long.started)>=30)
            precondition(!long.delivered);_ = jobs.read(long);precondition(long.delivered)
            let count=try await client.call(stats,arguments:[:]);let object=try JSONSerialization.jsonObject(with:Data(count.utf8)) as! [String:Int]
            precondition(object["calls"]==1,"The original call was repeated")
            print("PASS \(transport): real 31s call, no fixed timeout/retry, ownership, observation cancellation")
            let deadline=try jobs.start(client:client,tool:tool,input:["delay":3],session:owner,record:UUID(),timeout:0.2)
            await deadline.task?.value;precondition(deadline.state=="timed_out" && deadline.json.contains("outcome_uncertain"))
            let cancel=try jobs.start(client:client,tool:tool,input:["delay":3],session:owner,record:UUID(),timeout:nil)
            try await Task.sleep(nanoseconds:100_000_000)
            _ = try await jobs.manage(WorkspaceAction(name:"mcp_cancel",arguments:["job_id":cancel.id.uuidString]),session:owner)
            await cancel.task?.value;precondition(cancel.state=="cancelled")
            try await Task.sleep(nanoseconds:200_000_000)
            let cancelled=try await client.call(stats,arguments:[:]);let c=try JSONSerialization.jsonObject(with:Data(cancelled.utf8)) as! [String:Int]
            precondition(c["calls"]==3 && c["cancelled"]!>=2)
            let alive=try await client.call(tool,arguments:["text":"still connected"]);precondition(alive=="still connected")
            print("PASS \(transport): explicit deadline, cancel notification, no retry, connection survives")
            }
            let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
            let store=StudioStore(storageURL:root.appendingPathComponent("archive.json"))
            // Let the empty archive startup task finish before injecting fixture connections.
            try await Task.sleep(nanoseconds:100_000_000)
            store.archive.settings.chatBackend="remote"
            store.modelState="ready";store.loadedModel="fixture";store.archive.settings.maxConsecutiveActions=1
            var remote=RemoteConfiguration();remote.model="fixture";remote.endpoint="http://127.0.0.1:1/v1";store.archive.settings.remote=remote
            store.archive.settings.mcpServers=[config];store.mcpConnections[config.id]=client;store.mcpTools=tools
            store.mutateSelected{$0.approvalMode="manual"}
            let sid=store.selected!.id
            let action=ChatMessage(role:"assistant",text:"{\"suc_action\":{\"name\":\"mcp_call\",\"arguments\":{\"server\":\"\(config.id)\",\"tool\":\"echo\",\"input\":\"{\\\"delay\\\":3}\",\"wait_seconds\":0}}}")
            store.mutateSelected{$0.messages.append(action)};store.handleAgentResponse(sessionID:sid,messageID:action.id)
            for _ in 0..<100 where store.approvals.isEmpty{try await Task.sleep(nanoseconds:10_000_000)}
            precondition(store.approvals.count==1);store.resolveApproval(store.approvals[0].id,allowed:true);await store.agentTasks[sid]?.value
            guard let active=store.mcpJobs.jobs.first else {fatalError("MCP job missing: \(store.selected?.actions?.map{ $0.name+":"+$0.result } ?? [])")};precondition(active.running && store.agentIterations[sid]==1 && store.selected?.actions?.last?.state=="running")
            let check=ChatMessage(role:"assistant",text:"{\"suc_action\":{\"name\":\"mcp_status\",\"arguments\":{\"job_id\":\"\(active.id)\"}}}")
            store.mutateSelected{$0.messages.append(check)};store.handleAgentResponse(sessionID:sid,messageID:check.id);await store.agentTasks[sid]?.value
            precondition(store.approvals.isEmpty && store.agentIterations[sid]==1 && store.selected?.actions?.last?.name=="mcp_status")
            precondition(store.sessionTurnBusy(sid));store.stopAgent(sid);await active.task?.value
            precondition(active.state=="cancelled" && store.selected?.actions?.first{$0.id==active.recordID}?.state=="cancelled")
            // A premature final answer must wait for the already approved tool result.
            let finalRecord=ActionRecord(name:"mcp_call",details:"fixture",state:"running",result:"")
            store.recordAction(sid,finalRecord)
            let finalJob=try store.mcpJobs.start(client:client,tool:tool,input:["delay":0.3,"text":"final-result-sentinel"],session:sid,record:finalRecord.id,timeout:nil)
            let premature=ChatMessage(role:"assistant",text:"Completed.")
            store.mutateSelected{$0.messages.append(premature)};store.handleAgentResponse(sessionID:sid,messageID:premature.id)
            precondition(store.agentRuns.contains(sid));await store.agentTasks[sid]?.value
            precondition(finalJob.state=="completed" && finalJob.delivered)
            precondition(store.selected?.actions?.first{$0.id==finalRecord.id}?.result.contains("final-result-sentinel")==true)
            // The manual settings invocation shares the same lifecycle and Stop.
            let manual=Task {try await store.runMCP(tool,argumentsJSON:#"{"delay":3}"#,sessionID:sid)}
            for _ in 0..<100 where !store.mcpJobs.running(sid){try await Task.sleep(nanoseconds:10_000_000)}
            precondition(store.mcpJobs.running(sid));store.stopAgent(sid)
            do {try await manual.value;preconditionFailure("Manual invocation ignored Stop")} catch {}
            precondition(!store.mcpJobs.running(sid) && !store.mcpRuns.contains(sid))
            var recovered=store.archive;recovered.recoverInterrupted()
            precondition(recovered.sessions.flatMap{$0.actions ?? []}.allSatisfy{$0.state != "running"})
            try await Task.sleep(nanoseconds:500_000_000)
            print("PASS \(transport): premature final waits for result; manual invocation obeys Stop; archive recovery")
            print("PASS \(transport): manual approval, action-limit exemption for owned observation, busy queue, Stop and archived record")
        }
        precondition(MCPConnection.setupHint("Not authorized to send Apple events to Hopper Disassembler.").contains("Automazione"))
        precondition(MCPConnection.setupHint("missing GHIDRA_INSTALL_DIR").contains("JAVA_HOME"))
        print("MCP lifecycle tests: PASS")
    }
}
