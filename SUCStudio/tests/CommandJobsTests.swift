import Foundation
@main struct CommandJobsTests {
 @MainActor static func main() async throws {
  let numericWire=#"{"suc_action":{"name":"run_command","arguments":{"command":"printf fixture","wait_seconds":60,"timeout_seconds":3600}}}"#
  let parsed=WorkspaceAction.parse(numericWire)!
  precondition(parsed.arguments==["command":"printf fixture","wait_seconds":"60","timeout_seconds":"3600"])
  precondition(WorkspaceAction.parse("Commentary\n```bash\nprintf example\n```\n\n"+numericWire)==parsed)
  precondition(WorkspaceAction.parse(#"{"suc_action":{"name":"wait_command","arguments":{"job_id":"fixture","wait_seconds":60}}}"#)?.arguments["wait_seconds"]=="60")
  for bad in ["true","false","null","[]","{}"] {
   precondition(WorkspaceAction.parse("{\"suc_action\":{\"name\":\"run_command\",\"arguments\":{\"wait_seconds\":"+bad+"}}}")==nil)
  }
  precondition(WorkspaceAction.parse(numericWire+"\n"+numericWire)==nil)
  print("PASS: numeric JSON action arguments normalize; ambiguous and non-scalar actions stay rejected")
  let helper=URL(fileURLWithPath:CommandLine.arguments[1]),root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
  defer{try? FileManager.default.removeItem(at:root)}
  let jobs=CommandJobs(helperURL:helper),session=UUID(),other=UUID()
  let unicode=CommandJob(sessionID:session,recordID:UUID(),command:"fixture")
  let text="X"+String(repeating:"🙂€à",count:10000)
  unicode.append(Data(text.utf8));unicode.finish("completed",exitCode:0)
  var offset=0,joined=""
  repeat {let chunk=unicode.snapshot(offset:offset);joined+=chunk.output;offset=chunk.nextOffset} while offset<text.utf8.count
  precondition(joined==text)
  let partial=CommandJob(sessionID:session,recordID:UUID(),command:"fixture")
  partial.append(Data([0xE2,0x82]));precondition(partial.snapshot(offset:0).output.isEmpty && partial.snapshot(offset:0).nextOffset==0)
  partial.append(Data([0xAC]));precondition(partial.snapshot(offset:0).output=="€" && partial.snapshot(offset:0).nextOffset==3)
  let tail=CommandJob(sessionID:session,recordID:UUID(),command:"fixture")
  tail.append(Data(String(repeating:"€",count:100000).utf8));tail.finish("completed",exitCode:0)
  precondition(!tail.snapshot().output.contains("�") && tail.snapshot().omittedBytes>0)
  print("PASS: UTF-8 across read chunks, incomplete scalars and bounded tails")
  let long=try jobs.start(command:"printf BEGIN;sleep 31;printf END",root:root,session:session,recordID:UUID(),timeout:nil)
  try await jobs.wait(long,seconds:1);precondition(long.running && long.snapshot().output.contains("BEGIN"))
  do{_ = try jobs.lookup(long.id.uuidString,session:other);preconditionFailure("Cross-session lookup")}catch{}
  do{_ = try jobs.start(command:"touch duplicate",root:root,session:session,recordID:UUID(),timeout:nil);preconditionFailure("Duplicate allowed")}catch{}
  precondition(!FileManager.default.fileExists(atPath:root.appendingPathComponent("duplicate").path))
  let poll=Task{try await jobs.wait(long,seconds:60)}
  try await Task.sleep(nanoseconds:100_000_000);poll.cancel()
  do{try await poll.value;preconditionFailure("Cancelled wait completed")}catch{}
  precondition(long.running,"Cancelling an observation killed the process")
  print("PASS: wait expiry, cancelled observation, duplicate prevention and session ownership")
  await long.task?.value
  precondition(long.snapshot().state=="completed" && long.snapshot().exitCode==0 && long.snapshot().output=="BEGINEND" && long.snapshot().elapsed>=30)
  print("PASS: real command beyond 30 seconds finished without restart")
  let large=try jobs.start(command:"head -c 300000 /dev/zero | tr '\\000' x;printf TAIL;printf ERR >&2",root:root,session:session,recordID:UUID(),timeout:nil)
  await large.task?.value
  precondition(large.snapshot().state=="completed" && large.snapshot().totalOutputBytes==300007)
  precondition(large.snapshot().output.hasSuffix("TAILERR") && large.snapshot().omittedBytes>0,"tail=\(String(large.snapshot().output.suffix(40)).debugDescription), omitted=\(large.snapshot().omittedBytes)")
  var config=ChatSession();config.id=session;config.workspacePath=root.path;config.agentEnabled=true;config.allowCommands=true
  var lastOffset=0,all=""
  for _ in 0..<6 {
   let text=try await jobs.perform(WorkspaceAction(name:"command_status",arguments:["job_id":large.id.uuidString]),session:config,root:nil,recordID:UUID())
   let value=try JSONSerialization.jsonObject(with:Data(text.utf8)) as! [String:Any]
   let next=value["next_offset"] as! Int;precondition(next>=lastOffset);lastOffset=next;all+=value["output"] as! String
  }
  precondition(all.hasSuffix("TAILERR"))
  let empty=try await jobs.perform(WorkspaceAction(name:"command_status",arguments:["job_id":large.id.uuidString]),session:config,root:nil,recordID:UUID())
  let emptyValue=try JSONSerialization.jsonObject(with:Data(empty.utf8)) as! [String:Any]
  precondition(emptyValue["output"] as? String == "")
  let failed=try jobs.start(command:"printf FAIL >&2;exit 7",root:root,session:session,recordID:UUID(),timeout:nil);await failed.task?.value
  precondition(failed.snapshot().state=="failed" && failed.snapshot().exitCode==7 && failed.snapshot().output=="FAIL")
  let timeout=try jobs.start(command:"sleep 20",root:root,session:session,recordID:UUID(),timeout:0.2);await timeout.task?.value
  precondition(timeout.snapshot().state=="timed_out")
  let cancel=try jobs.start(command:"/bin/sh -c 'trap \"\" TERM; sleep 4; touch child-survived' & wait",root:root,session:session,recordID:UUID(),timeout:nil)
  try await Task.sleep(nanoseconds:200_000_000);jobs.stop(cancel.id,session:session);await cancel.task?.value
  try await Task.sleep(nanoseconds:4_300_000_000)
  precondition(cancel.snapshot().state=="cancelled" && !FileManager.default.fileExists(atPath:root.appendingPathComponent("child-survived").path))
  for bad in ["-1","61","NaN","0.1"] {
   do{_ = try await jobs.perform(WorkspaceAction(name:"run_command",arguments:["command":"touch invalid","wait_seconds":bad]),session:config,root:root,recordID:UUID());preconditionFailure("Invalid wait accepted")}catch{}
  }
  precondition(!FileManager.default.fileExists(atPath:root.appendingPathComponent("invalid").path))
  print("PASS: output tail and incremental offsets, nonzero exit, explicit timeout, descendant cancellation and input validation")
  let store=StudioStore(storageURL:root.appendingPathComponent("archive.json"),commandHelperURL:helper)
  store.modelState="ready";store.loadedModel="fixture";store.archive.settings.maxConsecutiveActions=1
  var remote=RemoteConfiguration();remote.model="fixture";remote.endpoint="http://127.0.0.1:1/v1";store.archive.settings.remote=remote
  store.mutateSelected{$0.workspacePath=root.path;$0.agentEnabled=true;$0.allowCommands=true;$0.approvalMode="manual"}
  let sid=store.selected!.id
  let start=ChatMessage(role:"assistant",text:#"{"suc_action":{"name":"run_command","arguments":{"command":"sleep 60","wait_seconds":0,"timeout_seconds":3600}}}"#)
  store.mutateSelected{$0.messages.append(start)};store.handleAgentResponse(sessionID:sid,messageID:start.id)
  for _ in 0..<100 where store.approvals.isEmpty {try await Task.sleep(nanoseconds:10_000_000)}
  precondition(store.approvals.count==1);store.resolveApproval(store.approvals[0].id,allowed:true);await store.agentTasks[sid]?.value
  let active=store.commandJobs.jobs.first!;precondition(active.running && store.agentIterations[sid]==1)
  for _ in 0..<2 {
   let check=ChatMessage(role:"assistant",text:#"{"suc_action":{"name":"command_status","arguments":{"job_id":""#+active.id.uuidString+#"","output_offset":0}}}"#)
   store.mutateSelected{$0.messages.append(check)};store.handleAgentResponse(sessionID:sid,messageID:check.id);await store.agentTasks[sid]?.value
   precondition(store.approvals.isEmpty && store.agentIterations[sid]==1)
   precondition(store.selected?.actions?.last?.name=="command_status")
  }
  store.mutateSelected{$0.draft="next queued request"};store.sendMessage()
  precondition(store.selected?.queuedMessages?.count==1)
  store.cancelSelected();await store.commandJobs.shutdown()
  precondition(!store.commandJobs.running(sid) && store.selected?.queuePaused==true)
  precondition(store.selected?.actions?.first(where:{$0.id==active.recordID})?.state=="cancelled")
  precondition(store.selected?.queuedMessages?.count==1)
  let defaults=AppSettings();precondition(defaults.context==16000)
  var old=defaults;old.context=4096
  let decodedOld=try JSONDecoder().decode(AppSettings.self,from:JSONEncoder().encode(old))
  precondition(decodedOld.context==4096)
  store.archive.settings.chatBackend="remote";store.archive.settings.remote=RemoteConfiguration();store.mutateSelected{$0.contextWindow=nil}
  precondition(store.contextLimit(store.selected!)==16000)
  store.archive.settings.remote?.model="fixture";store.archive.settings.remote?.modelContextWindows=["fixture":32000]
  precondition(store.contextLimit(store.selected!)==32000)
  store.mutateSelected{$0.contextWindow=8192};precondition(store.contextLimit(store.selected!)==8192)
  var archive=Archive();var restored=ChatSession();restored.actions=[ActionRecord(name:"run_command",details:"test",state:"running",result:"running")];archive.sessions=[restored];archive.recoverInterrupted()
  precondition(archive.sessions[0].actions?[0].state=="interrupted")
  let premature=StudioStore(storageURL:root.appendingPathComponent("early.json"),commandHelperURL:helper)
  premature.modelState="ready";premature.loadedModel="fixture";premature.archive.settings.remote=remote
  premature.mutateSelected{$0.workspacePath=root.path;$0.agentEnabled=true;$0.allowCommands=true}
  let earlyJob=try premature.commandJobs.start(command:"sleep 1;printf verified",root:root,session:premature.selected!.id,recordID:UUID(),timeout:nil)
  let earlyText=ChatMessage(role:"assistant",text:"A preliminary reply while the command is running.")
  premature.mutateSelected{$0.messages.append(earlyText)}
  premature.handleAgentResponse(sessionID:premature.selected!.id,messageID:earlyText.id)
  await premature.agentTasks[premature.selected!.id]?.value
  precondition(!earlyJob.running && premature.selected!.messages.contains{$0.kind=="tool_result" && $0.text.contains("verified")})
  let errorRecord=ActionRecord(name:"run_command",details:"fixture",state:"error",result:failed.snapshot().json)
  store.recordAction(sid,errorRecord)
  precondition(store.selected?.messages.last?.kind=="action_notice" && !(store.selected?.messages.last?.text.contains("job_id") ?? true))
  await jobs.shutdown()
  print("PASS: launch approval once, management past action limit, queue/Stop integration, archive recovery and 16000 context defaults with saved limits preserved")
 }
}
