import Foundation
@main struct TerminalArtifactTests {
 @MainActor static func main() async throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer{try? FileManager.default.removeItem(at:root)}
    let helper=URL(fileURLWithPath:CommandLine.arguments[1])
    let terminal=InteractiveTerminal(command:"test -t 0 || exit 9; printf 'READY\\n'; read answer; printf 'CONFIRMED:%s\\n' \"$answer\"; stty -echo; printf 'PASSWORD\\n'; read password; stty echo; test \"$password\" = 'secret-fixture' || exit 8; printf 'DONE\\n'")
    try terminal.start(root:root,helper:helper)
    func until(_ condition:()->Bool) async throws {for _ in 0..<200{if condition(){return};try await Task.sleep(nanoseconds:25_000_000)};throw MCPConnection.failure("Test timeout")}
    try await until{terminal.output.contains("READY") && terminal.awaitingInput};terminal.send("yes")
    try await until{terminal.output.contains("PASSWORD")};terminal.send("secret-fixture",secret:true)
    let status=await terminal.wait();assert(status==0 && terminal.output.contains("CONFIRMED:yes") && terminal.output.contains("DONE"));assert(!terminal.output.contains("secret-fixture"))
    let echo=InteractiveTerminal(command:"read value; printf '%s' \"$value\"")
    try echo.start(root:root,helper:helper);assert(!echo.send(String(repeating:"x",count:4097)) && echo.inputError != nil);echo.send("private-fixture",secret:true)
    let echoStatus=await echo.wait();assert(echoStatus==0 && !echo.output.contains("private-fixture"))
    try Data((String(repeating:"Testo UTF-8 🚀 è\n",count:20000)+"FINAL 🚀").utf8).write(to:root.appendingPathComponent("burst.txt"))
    let burst=InteractiveTerminal(command:"cat burst.txt");try burst.start(root:root,helper:helper)
    let burstStatus=await burst.wait();assert(burstStatus==0 && burst.output.hasSuffix("FINAL 🚀"));assert(burst.output.utf8.count<=131080 && !burst.output.dropFirst(4).contains("�"))
    assert(!burst.awaitingInput)
    let sleeping=InteractiveTerminal(command:"sleep 90")
    try sleeping.start(root:root,helper:helper)
    let wait=Task{await sleeping.wait()};try await Task.sleep(nanoseconds:100_000_000);wait.cancel();_ = await wait.value;assert(!sleeping.running)
    let source=root.appendingPathComponent("report.txt");try Data("artifact original".utf8).write(to:source)
    let archiveURL=root.appendingPathComponent("storage/sessions.json")
    let store=StudioStore(storageURL:archiveURL);store.mutateSelected{$0.workspacePath=root.path}
    _ = try await store.publishArtifact(path:"report.txt",session:store.selected!)
    let artifact=store.selected!.artifacts!.first!,saved=try store.artifactURL(artifact)
    try Data("changed".utf8).write(to:source);let savedText=try String(contentsOf:saved,encoding:.utf8);assert(savedText=="artifact original")
    let encoded=try JSONEncoder().encode(store.archive),decoded=try JSONDecoder().decode(Archive.self,from:encoded);assert(decoded.sessions.first!.artifacts!.first!.id==artifact.id)
    do{_ = try ArtifactFiles.snapshot(path:"../outside",root:root,archiveURL:archiveURL);fatalError("Traversal accepted")}catch{}
    let link=root.appendingPathComponent("outside-link");try FileManager.default.createSymbolicLink(atPath:link.path,withDestinationPath:"/etc/hosts")
    do{_ = try ArtifactFiles.snapshot(path:"outside-link",root:root,archiveURL:archiveURL);fatalError("Symlink escape accepted")}catch{}
    do{_ = try ArtifactFiles.url(ChatArtifact(name:"../escape",byteCount:1),archiveURL:archiveURL);fatalError("Unsafe metadata accepted")}catch{}
    let large=root.appendingPathComponent("large.bin");FileManager.default.createFile(atPath:large.path,contents:nil);let handle=try FileHandle(forWritingTo:large);try handle.truncate(atOffset:UInt64(ArtifactFiles.maximumSize+1));try handle.close()
    do{_ = try ArtifactFiles.snapshot(path:"large.bin",root:root,archiveURL:archiveURL);fatalError("Large artifact accepted")}catch{}
    do{_ = try await store.runInteractive(WorkspaceAction(name:"run_interactive",arguments:["command":"echo unauthorized"]),session:store.selected!);fatalError("Disabled commands accepted")}catch{}
    assert(store.terminals.isEmpty)
    store.mutateSelected{$0.agentEnabled=true;$0.approvalMode="automatic";$0.messages.append(ChatMessage(role:"assistant",text:"{\"suc_action\":{\"name\":\"write_file\",\"arguments\":{\"path\":\"auto.txt\",\"content\":\"generated\"}}}"))}
    store.handleAgentResponse(sessionID:store.selected!.id,messageID:store.selected!.messages.last!.id)
    try await until{store.selected?.artifacts?.count==2}
    assert(store.selected!.artifacts!.last!.name=="auto.txt")
    assert(AgentProtocol.instructions.contains("run_interactive") && AgentProtocol.instructions.contains("publish_artifact"))
    print("Terminal/artifact tests passed: actual PTY, interactive input/limit, no-echo password, bounded UTF-8 burst, cancellation, snapshot independence, persistence, traversal/symlink/size rejection")
 }
}
