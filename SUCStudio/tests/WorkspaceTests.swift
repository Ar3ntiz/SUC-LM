import Foundation
@main struct WorkspaceTests {
 // tests/WebToolsTests.swift is not listed in the (frozen) test scripts: build and run it here with the same core sources.
 static func runWebToolsTests() throws {
  let tests=URL(fileURLWithPath:#filePath).deletingLastPathComponent(), sources=tests.deletingLastPathComponent().appendingPathComponent("Sources")
  let out=URL(fileURLWithPath:CommandLine.arguments[1]).deletingLastPathComponent(), binary=out.appendingPathComponent("web-tools-tests")
  let core=try FileManager.default.contentsOfDirectory(atPath:sources.path).filter{$0.hasSuffix(".swift") && !$0.hasSuffix("Views.swift") && $0 != "SUCStudioApp.swift"}.sorted().map{sources.appendingPathComponent($0).path}
  func run(_ executable:String,_ arguments:[String]) throws { let p=Process();p.executableURL=URL(fileURLWithPath:executable);p.arguments=arguments;try p.run();p.waitUntilExit();precondition(p.terminationStatus==0,"\(executable) failed") }
  try run("/usr/bin/xcrun",["swiftc","-swift-version","5","-parse-as-library","-module-cache-path",out.appendingPathComponent("swift-cache").path]+core+[tests.appendingPathComponent("WebToolsTests.swift").path,"-o",binary.path])
  try run(binary.path,[tests.appendingPathComponent("web_fixture.py").path])
 }
 @MainActor static func main() async throws {
  let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString), outside=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);try Data("outside".utf8).write(to:outside)
  defer{try? FileManager.default.removeItem(at:root);try? FileManager.default.removeItem(at:outside)}
  // A single fenced object (```json or ```) and surrounding whitespace are accepted; JSON inside prose or extra keys are not.
  precondition(WorkspaceAction.parse("```json\n{\"suc_action\":{\"name\":\"read_file\",\"arguments\":{}}}\n```")?.name=="read_file")
  precondition(WorkspaceAction.parse("\n  ```\n{\"suc_action\":{\"name\":\"web_search\",\"arguments\":{\"query\":\"q\"}}}\n```  \n")?.arguments==["query":"q"])
  precondition(WorkspaceAction.parse("  {\"suc_action\":{\"name\":\"read_file\",\"arguments\":{}}}\n")?.name=="read_file")
  for text in ["Ecco: {\"suc_action\":{\"name\":\"read_file\",\"arguments\":{}}}","Cerco.\n```json\n{\"suc_action\":{\"name\":\"read_file\",\"arguments\":{}}}\n```","```json\n{\"suc_action\":{\"name\":\"read_file\",\"arguments\":{}}}\n```\nFatto.","```python\n{\"suc_action\":{\"name\":\"read_file\",\"arguments\":{}}}\n```","```json\n{\"suc_action\":{\"name\":\"read_file\",\"arguments\":{}}}\n```\n```json\n{}\n```","{\"suc_action\":{\"name\":\"read_file\",\"arguments\":{}},\"x\":1}","```json\n"+String(repeating:" ",count:128*1024)+"{\"suc_action\":{\"name\":\"read_file\",\"arguments\":{}}}\n```"] { precondition(WorkspaceAction.parse(text)==nil,"Accepted: \(text.prefix(80))") }
  precondition(WorkspaceAction.parse("{\"suc_action\":{\"name\":\"read_file\",\"arguments\":{\"path\":\"x\"}}}")?.name=="read_file")
  for path in ["../escape","/etc/passwd"] {do{_ = try WorkspaceTools.resolve(path,root:root);preconditionFailure("Traversal accepted")}catch{}}
  try FileManager.default.createSymbolicLink(at:root.appendingPathComponent("link"),withDestinationURL:outside)
  do{_ = try WorkspaceTools.resolve("link",root:root);preconditionFailure("Symlink accepted")}catch{}
  _ = try await WorkspaceTools.execute(WorkspaceAction(name:"write_file",arguments:["path":"test.txt","content":"hello"]),root:root,allowCommands:false)
  let read=try await WorkspaceTools.execute(WorkspaceAction(name:"read_file",arguments:["path":"test.txt"]),root:root,allowCommands:false);precondition(read=="hello")
  do{_ = try await WorkspaceTools.execute(WorkspaceAction(name:"run_command",arguments:["command":"echo forbidden"]),root:root,allowCommands:false);preconditionFailure("Shell not disabled")}catch{}
  let runner=WorkspaceCommand(helperURL:URL(fileURLWithPath:CommandLine.arguments[1]))
  let result=try await runner.run("printf shell-ok",root:root);precondition(result.contains("shell-ok") && result.contains("Exit status: 0"))
  let slow=WorkspaceCommand(helperURL:URL(fileURLWithPath:CommandLine.arguments[1]))
  let work=Task{try await slow.run("sleep 30",root:root)};try await Task.sleep(nanoseconds:100_000_000);work.cancel()
  do{_ = try await work.value;preconditionFailure("Cancel expected")}catch{}
  let store=StudioStore(storageURL:root.appendingPathComponent("archive.json"));let sessionID=store.selected!.id
  store.mutateSelected{$0.workspacePath=root.path;$0.agentEnabled=true;$0.approvalMode="manual"}
  let action=WorkspaceAction(name:"write_file",arguments:["path":"denied.txt","content":"never"])
  let prompt=Task{await store.approve(action,session:store.selected!,root:root)}
  await Task.yield();precondition(store.approvals.count==1);store.resolveApproval(store.approvals[0].id,allowed:false);let allowed=await prompt.value;precondition(!allowed)
  store.mutateSelected{$0.approvalMode="automatic"}
  let allowedAuto=await store.approve(action,session:store.selected!,root:root);precondition(allowedAuto)
  store.modelState="ready";store.loadedModel="test"
  let message=ChatMessage(role:"assistant",text:"{\"suc_action\":{\"name\":\"write_file\",\"arguments\":{\"path\":\"actual.txt\",\"content\":\"real result\"}}}")
  store.mutateSelected{$0.messages.append(message)};store.handleAgentResponse(sessionID:sessionID,messageID:message.id)
  store.newSession();await store.agentTasks[sessionID]?.value
  let actual=try String(contentsOf:root.appendingPathComponent("actual.txt"),encoding:.utf8);precondition(actual=="real result")
  precondition(store.archive.sessions.first{$0.id==sessionID}!.actions?.last?.state=="completed")
  precondition(store.selected!.messages.isEmpty)
  // Without work directory and model actions off, non-web actions are still ignored.
  store.newSession();let bareID=store.selected!.id
  let bareAction=ChatMessage(role:"assistant",text:"{\"suc_action\":{\"name\":\"write_file\",\"arguments\":{\"path\":\"bare.txt\",\"content\":\"never\"}}}")
  store.mutateSelected{$0.approvalMode="automatic";$0.messages.append(bareAction)};store.handleAgentResponse(sessionID:bareID,messageID:bareAction.id)
  precondition(store.agentTasks[bareID]==nil && store.approvals.isEmpty && store.selected!.actions==nil && store.selected!.messages.last?.kind==nil && store.agentIterations[bareID]==nil)
  precondition(bareAction.visibleInChat(agent:false,web:true) && !bareAction.visibleInChat(agent:true))
  try runWebToolsTests()
  print("WorkspaceTests: strict action envelope (single fence accepted, prose rejected), non-web actions ignored without agent, paths/symlinks, file IO, disabled shell, command/cancellation, manual/automatic approval and session routing passed")
 }
}
