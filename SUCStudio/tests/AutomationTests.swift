import Foundation
@main struct AutomationTests {
 @MainActor static func main() async throws {
  let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
  let store=StudioStore(storageURL:root.appendingPathComponent("archive.json"));store.modelState="ready";store.loadedModel="fixture"
  store.mutateSelected{$0.workspacePath=root.path;$0.agentEnabled=true;$0.allowSubagents=true;$0.approvalMode="automatic"}
  let parent=store.selected!
  let task=Task{try await store.delegateTask(WorkspaceAction(name:"delegate_task",arguments:["task":"Explain the project"]),session:parent)}
  await Task.yield()
  guard let child=store.archive.sessions.first(where:{$0.parentSessionID==parent.id}),let request=store.requests.first(where:{$0.value.session==child.id})?.key else{preconditionFailure("Child request missing")}
  precondition(child.allowSubagents==false && child.workspacePath==parent.workspacePath && child.approvalMode==parent.approvalMode && store.selected!.id==parent.id)
  store.receive(["event":"delta","id":request,"payload":["text":"Subtask complete"]]);store.receive(["event":"finished","id":request,"payload":[:]])
  let result=try await task.value;precondition(result=="Subtask complete" && store.selected!.subtasks!.last!.state=="completed")
  do{_ = try await store.delegateTask(WorkspaceAction(name:"delegate_task",arguments:["task":"nested"]),session:child);preconditionFailure("Nested delegate")}catch{}
  store.subtaskCounts[parent.id]=4
  do{_ = try await store.delegateTask(WorkspaceAction(name:"delegate_task",arguments:["task":"excess"]),session:parent);preconditionFailure("Delegate limit")}catch{}
  store.archive.settings.hooks=[StudioHook(name:"Environment",event:"before_tool",command:"printf '%s' \"$SUC_HOOK_EVENT:$SUC_TOOL_NAME\"",enabled:true)]
  let helper=URL(fileURLWithPath:CommandLine.arguments[1])
  try await store.runHooks("before_tool",session:parent,action:WorkspaceAction(name:"read_file",arguments:["path":"x"]),helperURL:helper)
  precondition(store.selected!.actions!.last!.result.contains("before_tool:read_file"))
  store.archive.settings.hooks=[StudioHook(name:"Block",event:"before_tool",command:"exit 3",enabled:true)]
  do{try await store.runHooks("before_tool",session:parent,helperURL:helper);preconditionFailure("Hook should block")}catch{}
  precondition(store.selected!.actions!.last!.state=="error")
  store.archive.settings.hooks?[0].enabled=false;try await store.runHooks("before_tool",session:parent,helperURL:helper)
  print("AutomationTests: isolated subagent, inherited permissions, results, nesting/count limits, hook environment/error/disable passed")
 }
}
