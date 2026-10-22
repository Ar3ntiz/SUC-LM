import AppKit
import Foundation
@main struct ToolRecoveryTests {
 @MainActor static func wait(_ label:String,_ check:()->Bool) async throws {
  for _ in 0..<800 {if check(){return};try await Task.sleep(nanoseconds:20_000_000)}
  fatalError("Timeout: "+label)
 }
 @MainActor static func main() async throws {
  _=NSApplication.shared;NSApp.setActivationPolicy(.accessory)
  let examples=[
   "Verifico lo strumento.<tool_call>run_command(\"printf fixture\", 30, 60)",
   #"Riprovo.<tool_call>suc_action({"name":"read_file","arguments":{"path":"fixture.txt"}})"#,
   #"It's ready.<tool_call>{"suc_action":{"name":"read_file","arguments":{"path":"fixture.txt"}}}</tool_call>"#,
   "Verifico.<tool_call>\nrun_command(\"printf fixture\")",
   "Verifico.\r\n🙂 Nuova riga.<tool_call>run_command(\"printf fixture\", 30, 60)"
  ]
  for sample in examples {
   precondition(WorkspaceAction.parse(sample)==nil && WorkspaceAction.isActionEnvelope(sample))
   let marker=sample.range(of:"<tool_call>")!,comment=String(sample[..<marker.lowerBound])
   let chars=Array(sample)
   for count in 1...chars.count {
    let prefix=String(chars.prefix(count))
    let message=ChatMessage(role:"assistant",text:prefix,state:"generating")
    if prefix.count>=comment.count+2 {precondition(!message.chatParts.answer.contains("<t"),"Leaked prefix: "+prefix)}
   }
   precondition(WorkspaceAction.commentaryBeforeEnvelope(sample,partial:false)==comment)
  }
  let literals=["Use `<tool_call>run_command(...)` as an example.","Example: \"<tool_call>run_command(...)\"", "Example: '<tool_call>run_command(...)'", "```python\n<tool_call>run_command(...)\n```","> <tool_call>run_command(...)","Use <tool_call> to request a tool.","Use \\<tool_call>run_command(...) as literal Markdown."]
  for text in literals {precondition(WorkspaceAction.parse(text)==nil && !WorkspaceAction.isActionEnvelope(text),text)}
  let literalAction=WorkspaceAction(name:"write_file",arguments:["path":"example.md","content":"Example: <tool_call>run_command(...)"])
  let canonical=String(decoding:try JSONSerialization.data(withJSONObject:["suc_action":["name":literalAction.name,"arguments":literalAction.arguments]]),as:UTF8.self)
  precondition(WorkspaceAction.parse(canonical)==literalAction)
  precondition(WorkspaceAction.parse(examples[0]+"\n"+canonical)==nil)
  print("PASS markers: inline functions/JSON stay nonexecutable, every streaming prefix hidden, commentary retained; code/quotes/examples untouched, mixed calls rejected")
  let guarder=ActionLoopGuard(),owner=UUID(),command=WorkspaceAction(name:"run_command",arguments:["command":"printf fixture","wait_seconds":"1"])
  var records=[ActionRecord]()
  for i in 0..<3 {
   var record=ActionRecord(name:"run_command",details:"fixture",state:"completed",result:"")
   record.result="{\"job_id\":\"\(UUID())\",\"elapsed_seconds\":\(i),\"state\":\"completed\",\"exit_code\":0,\"output\":\"same\",\"total_output_bytes\":4,\"omitted_bytes\":0}"
   guarder.note(command,session:owner,recordID:record.id);records.append(record)
  }
  precondition(guarder.blocks(command,session:owner,records:records))
  var variant=command;variant.arguments["wait_seconds"]="60";variant.arguments["timeout_seconds"]="3600"
  precondition(guarder.blocks(variant,session:owner,records:records))
  let poll=WorkspaceAction(name:"command_status",arguments:["job_id":"fixture"])
  guarder.note(poll,session:owner,recordID:UUID());precondition(!guarder.blocks(poll,session:owner,records:records) && guarder.blocks(command,session:owner,records:records))
  var changed=records;changed[2].result=changed[2].result.replacingOccurrences(of:"same",with:"new output")
  precondition(!guarder.blocks(command,session:owner,records:changed));changed=records;changed[2].state="running";precondition(!guarder.blocks(command,session:owner,records:changed))
  guarder.note(WorkspaceAction(name:"read_file",arguments:["path":"other"]),session:owner,recordID:UUID());precondition(!guarder.blocks(command,session:owner,records:records))
  let mcpOwner=UUID(),mcp=WorkspaceAction(name:"mcp_call",arguments:["server":"fixture","tool":"echo","input":"{}"])
  var mcpRecords=[ActionRecord]()
  for i in 0..<3 {let record=ActionRecord(name:"mcp_call",details:"fixture",state:"completed",result:"{\"job_id\":\"\(UUID())\",\"elapsed_seconds\":\(i),\"state\":\"completed\",\"result\":\"same result\"}");mcpRecords.append(record);guarder.note(mcp,session:mcpOwner,recordID:record.id)}
  precondition(guarder.blocks(mcp,session:mcpOwner,records:mcpRecords));precondition(!guarder.blocks(WorkspaceAction(name:"mcp_wait",arguments:["job_id":"fixture"]),session:mcpOwner,records:mcpRecords))
  guarder.reset(mcpOwner);precondition(!guarder.blocks(mcp,session:mcpOwner,records:mcpRecords))
  print("PASS loop guard: ignore job IDs/timing/wait settings; changed output, running jobs, intervening actions, management and reset stay allowed; MCP covered")
  let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
  let server=Process(),pipe=Pipe();server.executableURL=URL(fileURLWithPath:"/usr/bin/python3");server.arguments=[CommandLine.arguments[1]];server.standardOutput=pipe
  try server.run();defer{server.terminate()}
  let port=String(decoding:pipe.fileHandleForReading.availableData,as:UTF8.self).trimmingCharacters(in:.whitespacesAndNewlines)
  let helper=URL(fileURLWithPath:CommandLine.arguments[2])
  func store(_ model:String,manual:Bool=false) throws -> StudioStore {
   let directory=root.appendingPathComponent(UUID().uuidString);try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
   try "FIXTURE_READ_SENTINEL".write(to:directory.appendingPathComponent("fixture.txt"),atomically:true,encoding:.utf8)
   let value=StudioStore(storageURL:directory.appendingPathComponent("archive.json"),commandHelperURL:helper);value.schedulingEnabled=false
   value.archive.settings.chatBackend="remote";value.archive.settings.remote=RemoteConfiguration(endpoint:"http://127.0.0.1:\(port)/v1",model:model);value.archive.settings.maxConsecutiveActions=100
   value.mutateSelected{$0.workspacePath=directory.path;$0.agentEnabled=true;$0.allowCommands=true;$0.allowWeb=false;$0.approvalMode=manual ? "manual":"automatic";$0.draft="START"}
   return value
  }
  for model in ["inline-function","inline-suc","inline-json"] {
   let value=try store(model),id=value.selected!.id;value.sendMessage()
   try await wait("recovered "+model){!value.sessionTurnBusy(id) && value.selected!.messages.contains{$0.text=="RECOVERED_FINAL"}}
   precondition(value.agentRepairs[id]==1 && value.selected!.actions?.filter{$0.state=="completed" && $0.name=="read_file"}.count==1)
   precondition(value.selected!.messages.contains{$0.text.contains("<tool_call>") && $0.kind=="tool_call"})
   precondition(!value.selected!.messages.filter{$0.role=="assistant" && !$0.isUIReport}.contains{$0.chatParts.answer.contains("<tool_call>")})
  }
  let bounded=try store("always-inline"),bid=bounded.selected!.id;bounded.sendMessage()
  try await wait("repair bound"){!bounded.sessionTurnBusy(bid)}
  precondition(bounded.agentRepairs[bid]==2 && (bounded.selected!.actions ?? []).isEmpty && bounded.selected!.messages.last(where:{!$0.isUIReport})?.state=="error")
  let truncated=try store("length-inline"),tid=truncated.selected!.id;truncated.sendMessage();try await wait("length not repaired"){!truncated.sessionTurnBusy(tid)}
  precondition(truncated.agentRepairs[tid]==0 && (truncated.selected!.actions ?? []).isEmpty)
  let manual=try store("inline-suc",manual:true),mid=manual.selected!.id;manual.sendMessage();try await wait("manual permission"){!manual.approvals.isEmpty}
  precondition((manual.selected!.actions ?? []).isEmpty);manual.resolveApproval(manual.approvals[0].id,allowed:true)
  try await wait("manual recovery"){!manual.sessionTurnBusy(mid) && manual.selected!.messages.contains{$0.text=="RECOVERED_FINAL"}}
  let stopped=try store("stop-repair"),sid=stopped.selected!.id;stopped.sendMessage()
  try await wait("repair sent"){stopped.selected!.messages.contains{$0.kind=="tool_result" && $0.text.hasPrefix("Feedback harness")}}
  stopped.mutateSelected{$0.draft="UPDATE"};stopped.sendMessage();stopped.cancelSelected();try await wait("Stop settles"){!stopped.sessionTurnBusy(sid)}
  precondition(stopped.selected!.queuePaused==true && stopped.selected!.queuedMessages?.count==1 && (stopped.selected!.actions ?? []).isEmpty)
  let steering=try store("inline-function"),steerID=steering.selected!.id;steering.sendMessage();steering.mutateSelected{$0.draft="UPDATE"};steering.sendMessage()
  try await wait("steering priority"){!steering.sessionTurnBusy(steerID) && steering.selected!.messages.contains{$0.text=="STEERED_FINAL"}}
  precondition(steering.agentRepairs[steerID]==0 && (steering.selected!.actions ?? []).isEmpty)
  print("PASS recovery: three GLM-like formats to one real read; persistent/length bounds, manual approval, Stop/paused queue and steering priority")
  let loop=try store("loop"),lid=loop.selected!.id;loop.sendMessage();try await wait("loop blocked"){!loop.sessionTurnBusy(lid)}
  precondition(loop.commandJobs.jobs.count==3 && loop.selected!.actions?.last?.state=="blocked")
  let runs=URL(fileURLWithPath:loop.selected!.workspacePath!).appendingPathComponent("runs.txt")
  let first=try String(contentsOf:runs,encoding:.utf8);precondition(first=="xxx")
  loop.mutateSelected{$0.draft="START"};loop.sendMessage();try await wait("fresh turn"){!loop.sessionTurnBusy(lid)}
  let second=try String(contentsOf:runs,encoding:.utf8);precondition(second=="xxxxxx" && loop.commandJobs.jobs.count==6)
  let progress=try store("progress"),pid=progress.selected!.id;progress.sendMessage();try await wait("changing output allowed"){!progress.sessionTurnBusy(pid) && progress.selected!.messages.contains{$0.text=="PROGRESS_FINAL"}}
  precondition(progress.commandJobs.jobs.count==5 && progress.selected!.actions?.allSatisfy{$0.state=="completed"}==true)
  print("PASS real operations: exactly three repeated commands, fourth blocked even at action limit 100; fresh turn allowed; five changed-output operations finish")
 }
}
