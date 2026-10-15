import AppKit
import CryptoKit
import Foundation
import WebKit

// Built and run by WorkspaceTests (scripts/test-workspace.sh, called by scripts/test-swift.sh): the test scripts are frozen.
// No network: every page comes from tests/web_fixture.py on 127.0.0.1.
@main struct WebToolsTests {
 static func expectError(_ text:String,_ body:() async throws -> Any) async {
  do { _ = try await body();preconditionFailure("Error expected: \(text)") } catch { precondition(error.localizedDescription.contains(text),"Unexpected error: \(error.localizedDescription) / expected: \(text)") }
 }
 /// Windows of this process that intersect a display (CGWindowList, on-screen windows only).
 static func onDisplay() -> [CGRect] {
  let pid=ProcessInfo.processInfo.processIdentifier, mainHeight=NSScreen.screens.first?.frame.height ?? 0, screens=NSScreen.screens.map{$0.frame}
  let info=(CGWindowListCopyWindowInfo([.optionOnScreenOnly],kCGNullWindowID) as? [[String:Any]]) ?? []
  return info.filter{($0[kCGWindowOwnerPID as String] as? Int32)==pid}.compactMap{
   guard let b=$0[kCGWindowBounds as String] as? [String:CGFloat],let x=b["X"],let y=b["Y"],let w=b["Width"],let h=b["Height"] else {return nil}
   let rect=CGRect(x:x,y:mainHeight-y-h,width:w,height:h);return screens.contains{$0.intersects(rect)} ? rect:nil
  }
 }
 /// 1.4.5_Final: during a web action the browser is never key, never on a display, and the app is never activated.
 @MainActor static func assertHidden(_ browser:WebBrowser,_ context:String) {
  precondition(onDisplay().isEmpty,"Browser window on a display (\(context)): \(onDisplay())")
  precondition(browser.window?.isKeyWindow != true && browser.window?.isMainWindow != true && browser.viewer==nil && !NSApp.isActive && NSApp.keyWindow==nil,"Browser fronted (\(context))")
 }
 /// Page-side timer/rAF/visibility probe (not from ds4): 4 × setTimeout(250), 2 × setTimeout(900), rAF count over the 250 ms chain (2.8 s, inside the 5-step scroll of /lazy).
 static let timerProbe="return await new Promise(resolve=>{const out={vis:document.visibilityState,t250:[],t900:[],raf:0};let raf=0;const tick=()=>{raf++;requestAnimationFrame(tick)};requestAnimationFrame(tick);const seq=(ms,n,arr,done)=>{let last=performance.now();const step=()=>{setTimeout(()=>{const now=performance.now();arr.push(now-last);last=now;if(arr.length<n)step();else done();},ms)};step();};seq(250,4,out.t250,()=>{out.raf=raf;seq(900,2,out.t900,()=>resolve(JSON.stringify(out)));});});"
 struct TimerSample: Decodable { let vis:String; let t250:[Double]; let t900:[Double]; let raf:Int }
 static func wait(line:Int=#line,_ condition:@MainActor () -> Bool) async throws { for _ in 0..<1500 { if await condition() {return};try await Task.sleep(nanoseconds:10_000_000) };preconditionFailure("Timed out waiting (line \(line))") }
 @MainActor static func main() async throws {
  _ = NSApplication.shared;NSApp.setActivationPolicy(.accessory)
  // 9. Scripts byte-identical to ds4 (hashes from docs/WEB_TOOLS_SPEC_IT.md §4).
  let hashes=[WebTools.probeScript:"e778d2b1990fd0e7999bf6ffb4d40a816f6505ddfac4ff030c58d07025e3c517",WebTools.consentScript:"8ce0075fd425c8b196855ad0401f5be2b99a3137ef864481e09d305c21cec784",WebTools.searchScript:"49d9c714fcad6f1475f4c4fcaa0407b7a98fe3466780c4f38024743aa2fbc67f",WebTools.pageScript:"f35e3203f1c8b756e5b754a6b952f8b3cbc3b6f3836134b44a102fabee8138f2",WebTools.scrollScript:"72f5ad19920fa0a4476743d911a1e2939b0fdd8b7fcf2a5a94c0ed7d91e791c2"]
  precondition(hashes.count==5)
  for (script,hash) in hashes { precondition(SHA256.hash(data:Data(script.utf8)).map{String(format:"%02x",$0)}.joined()==hash,"Script hash mismatch") }
  // Pure helpers: ds4 URL encoding, result formats, 8 KiB byte limit on a UTF-8 boundary.
  precondition(WebTools.encode("a b&c/è~")=="a%20b%26c%2F%C3%A8~")
  precondition(WebTools.formatVisit(url:"u",path:"p.md",markdown:"x\ny")=="visit_page url=u\noutput_path=p.md (3 bytes, 2 lines)\n<markdown>\nx\ny\n</markdown>\n")
  let wide=WebTools.formatVisit(url:"u",path:"p.md",markdown:"a"+String(repeating:"é",count:5000))
  precondition(wide.hasPrefix("visit_page url=u\noutput_path=p.md (10001 bytes, 1 lines)\n<head -100 p.md>\n") && wide.hasSuffix("\n</head>\nUsa read_file con path=p.md per leggere il resto del Markdown.\n"))
  precondition(wide.components(separatedBy:"\n")[3].utf8.count==8191)
  // 7. Arguments and URL schemes.
  for url in ["file:///etc/hosts","javascript:alert(1)","about:blank","data:text/html,x","ftp://example.com/x"] {
   await expectError("Tool error: visit_page failed:"){try WebTools.validate(WorkspaceAction(name:"visit_page",arguments:["url":url]),allowWeb:true)}
  }
  await expectError("Tool error: visit_page requires url"){try WebTools.validate(WorkspaceAction(name:"visit_page",arguments:[:]),allowWeb:true)}
  await expectError("Tool error: web_search requires query"){try WebTools.validate(WorkspaceAction(name:"web_search",arguments:["query":""]),allowWeb:true)}
  await expectError("Tool error: google_search requires query"){try WebTools.validate(WorkspaceAction(name:"google_search",arguments:[:]),allowWeb:true)}
  precondition(WebTools.journalTitle(WorkspaceAction(name:"web_search",arguments:["query":"swift"]))=="Ricerca Google: swift")
  var session=ChatSession();session.agentEnabled=true;session.workspacePath="/tmp"
  precondition(PromptComposer.compose(settings:AppSettings(),session:session).contains("web_search(query), visit_page(url)"))
  session.allowWeb=false;precondition(!PromptComposer.compose(settings:AppSettings(),session:session).contains("web_search"))
  // 1.4.5_01: web-only protocol without work directory / with model actions off; full instructions unchanged with the agent.
  let full=PromptComposer.compose(settings:AppSettings(),session:{var value=ChatSession();value.agentEnabled=true;value.workspacePath="/tmp";return value}())
  precondition(full.contains(AgentProtocol.instructions(web:true)) && full.contains("read_file(path)") && full.contains("usa web_search invece di dire che non puoi navigare") && !full.contains("Strumenti web SUC disponibili"))
  for bare in [ChatSession(),{var value=ChatSession();value.workspacePath="/tmp";return value}(),{var value=ChatSession();value.agentEnabled=true;return value}()] {
   let prompt=PromptComposer.compose(settings:AppSettings(),session:bare)
   precondition(prompt.contains("web_search(query), visit_page(url)") && prompt.contains("{\"suc_action\":{\"name\":\"web_search\"") && prompt.contains("usa web_search invece di dire che non puoi navigare") && prompt.contains("Massimo 12 azioni consecutive.") && !prompt.contains("read_file") && !prompt.contains("run_command"),"Web-only prompt:\n"+prompt)
   var off=bare;off.allowWeb=false;precondition(!PromptComposer.compose(settings:AppSettings(),session:off).contains("suc_action"))
  }
  precondition(WebTools.formatVisit(url:"u",path:"/a/p.md",markdown:String(repeating:"x\n",count:150),workspace:false).hasSuffix("</head>\n"+WebTools.standaloneHint+"\n"))

  let fixture=Process(), output=Pipe(), input=Pipe()
  fixture.executableURL=URL(fileURLWithPath:"/usr/bin/python3");fixture.arguments=[CommandLine.arguments[1]];fixture.standardOutput=output;fixture.standardInput=input
  try fixture.run();defer{fixture.terminate()}
  guard let port=Int(String(decoding:output.fileHandleForReading.availableData,as:UTF8.self).trimmingCharacters(in:.whitespacesAndNewlines)) else {preconditionFailure("Web fixture not available")}
  let base="http://127.0.0.1:\(port)"
  let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
  var logs:[String]=[]
  let browser=WebBrowser{WKWebsiteDataStore.nonPersistent()};browser.searchEndpoint=base+"/results?q="
  // Button state model: every transition is recorded and checked against the window state.
  var states:[WebActivity?]=[], phaseTimes:[(String,Date)]=[], checkHidden=true // off while the test itself opens the viewer
  browser.onActivity={activity in
   states.append(activity);if let activity,phaseTimes.last?.0 != activity.phase {phaseTimes.append((activity.phase,Date()))}
   if checkHidden {assertHidden(browser,activity?.phase ?? "finished")}
  }
  precondition(!NSApp.isActive)
  func run(_ name:String,_ key:String,_ value:String) async throws -> String { try await WebTools.execute(WorkspaceAction(name:name,arguments:[key:value]),root:root,browser:browser){logs.append($0)} }
  func stored(_ result:String) throws -> (String,String) {
   let line=result.components(separatedBy:"\n")[1];let path=String(line.dropFirst("output_path=".count).prefix{$0 != " "})
   precondition(path.hasPrefix(".suc-web/visit-") && path.hasSuffix(".md"));return (path,try String(contentsOf:root.appendingPathComponent(path),encoding:.utf8))
  }
  // 1. Search results: Google/gstatic, hidden, short and duplicate links dropped, /url?q= rewritten.
  let search=try await run("web_search","query","fixture query")
  precondition(search=="# Google search results\n\nURL: \(base)/results?q=fixture%20query\n\n## Visible links\n- [First result](https://example.com/one)\n- [Redirected result](https://example.org/target)\n\n## Text snapshot\nImpostazioni Google Static asset First result First result again Redirected result ab Snippet text.","Search output:\n"+search)
  precondition(browser.hasWindow && browser.activeWebView==nil && browser.activity==nil && browser.window?.isVisible==false && browser.viewer==nil && onDisplay().isEmpty)
  precondition(states.first??.phase==WebPhase.navigation && states.first??.title=="Ricerca Google: fixture query" && states.contains{$0?.phase==WebPhase.loading} && states.contains{$0?.phase==WebPhase.extraction} && states.last! == nil && !states.contains{$0?.verification==true},"States: \(states)")
  precondition(states.compactMap{$0}.allSatisfy{$0.sessionID==nil && $0.label().hasPrefix("Ricerca Google: fixture query · ")})
  // 5. Short page: <markdown> format, full file in .suc-web.
  let short=try await run("visit_page","url",base+"/short")
  let (shortPath,shortMarkdown)=try stored(short)
  precondition(shortMarkdown=="# Short page\n\nURL: \(base)/short\n\n## Content\n\n# Titolo\n\nUn paragrafo con [un link](https://example.com/a) e `codice`.\n\n- voce uno\n\n## Visible links\n- [un link](https://example.com/a)","Short page:\n"+shortMarkdown)
  precondition(short=="visit_page url=\(base)/short\noutput_path=\(shortPath) (\(shortMarkdown.utf8.count) bytes, 14 lines)\n<markdown>\n\(shortMarkdown)\n</markdown>\n")
  // 4. Long page: exact <head -100 …> format.
  let long=try await run("visit_page","url",base+"/long")
  let (longPath,longMarkdown)=try stored(long);let lines=longMarkdown.components(separatedBy:"\n")
  precondition(lines.count>100 && longMarkdown.utf8.count>8192 && longMarkdown.contains("Line 149: ") && longPath != shortPath)
  precondition(long=="visit_page url=\(base)/long\noutput_path=\(longPath) (\(longMarkdown.utf8.count) bytes, \(lines.count) lines)\n<head -100 \(longPath)>\n"+lines[0..<100].joined(separator:"\n")+"\n</head>\nUsa read_file con path=\(longPath) per leggere il resto del Markdown.\n")
  // 2. Consent button clicked, pause, new wait: the extraction sees the new content.
  logs=[];let consent=try await run("visit_page","url",base+"/consent");let (_,consentMarkdown)=try stored(consent)
  precondition(logs.contains("clicked Accetta tutto") && consentMarkdown.contains("Contenuto dopo il consenso") && !consentMarkdown.contains("Prima di continuare"),"Consent: \(logs) \(consentMarkdown)")
  // 3. Lazy loading on scroll: the shim counts the addEventListener handler, scrolling loads blocks.
  // 1.4.5_Final: in the hidden window the page stays "visible" for WebKit: timers and rAF are not throttled while ds4's scroll runs.
  logs=[];phaseTimes=[];var sample:TimerSample?
  let lazyRun=Task{try await run("visit_page","url",base+"/lazy")}
  try await wait{browser.activity?.phase==WebPhase.scrolling};assertHidden(browser,"scroll")
  sample=try JSONDecoder().decode(TimerSample.self,from:Data(try await browser.evaluate(browser.activeWebView!,timerProbe).utf8))
  let lazy=try await lazyRun.value;let (_,lazyMarkdown)=try stored(lazy)
  precondition(logs.contains{$0.hasPrefix("scrolled ")} && lazyMarkdown.contains("Lazy block 100") && lazyMarkdown.contains("Lazy block 200"),"Lazy: \(logs)")
  let steps=Double(logs.first{$0.hasPrefix("scrolled ")}!.components(separatedBy:" ")[1])!
  let scrollStart=phaseTimes.first{$0.0==WebPhase.scrolling}!.1, scrollEnd=phaseTimes.first{$0.0==WebPhase.extraction}!.1, perStep=scrollEnd.timeIntervalSince(scrollStart)/steps*1000
  precondition(sample!.vis=="visible" && sample!.raf>=10 && sample!.t250.allSatisfy{$0>=245 && $0<=300} && sample!.t900.allSatisfy{$0>=895 && $0<=960} && perStep>=895 && perStep<=980,"Hidden timings: \(sample!) \(perStep) ms/step")
  print(String(format:"WebToolsTests hidden timings: visibility=%@ rAF=%d setTimeout(250) max %.0f ms, setTimeout(900) max %.0f ms, scroll %.0f ms/step over %.0f steps",sample!.vis,sample!.raf,sample!.t250.max()!,sample!.t900.max()!,perStep,steps))
  // IntersectionObserver lazy loading and content fetched after load, hidden.
  let observed=try stored(try await run("visit_page","url",base+"/iolazy")).1;precondition(observed.contains("Observed block 200"))
  let fetched=try stored(try await run("visit_page","url",base+"/xhr")).1;precondition(fetched.contains("\nDati caricati dopo il load") && fetched.contains("XHR Dati caricati dopo il load"),fetched)
  precondition(browser.window?.isVisible==false && onDisplay().isEmpty && browser.activity==nil)
  // 6. Verification page: error, page kept loaded in the hidden window (not fronted), orange button state, no automatic click.
  browser.searchEndpoint=base+"/sorry/index?q="
  await expectError("Tool error: web_search failed: "+WebTools.verificationMessage){try await run("web_search","query","blocked")}
  precondition(WebTools.verificationMessage.contains("pulsante globo") && browser.parkedWebView != nil && browser.parkedWebView?.window===browser.window && browser.activity?.verification==true && states.last??.verification==true && browser.activity?.label()==WebTools.verificationLabel)
  assertHidden(browser,"verification")
  // Opening from the button shows the page with its status bar; closing the viewer puts it back in the hidden window, still pending.
  checkHidden=false;browser.showViewer(activate:false)
  try await wait{!onDisplay().isEmpty} // WindowServer updates its list asynchronously
  precondition(browser.viewer?.isVisible==true && browser.parkedWebView?.window===browser.viewer && browser.window?.isVisible==false,"Viewer: \(String(describing:browser.viewer?.isVisible)) \(String(describing:browser.parkedWebView?.window)) \(String(describing:browser.window?.isVisible))")
  browser.viewer?.close()
  precondition(browser.viewer==nil && browser.parkedWebView?.window===browser.window && browser.activity?.verification==true);try await wait{onDisplay().isEmpty};assertHidden(browser,"viewer closed");checkHidden=true
  // Verification completed (the page leaves /sorry/): the button disappears and the page is released.
  browser.parkedWebView?.load(URLRequest(url:URL(string:base+"/short")!))
  try await wait{browser.activity==nil && browser.parkedWebView==nil};precondition(browser.window?.isVisible==false && onDisplay().isEmpty)
  let hits=try String(contentsOf:URL(string:base+"/hits")!,encoding:.utf8);precondition(hits.contains("/sorry/index") && !hits.contains("/clicked"))
  // 8. Cancellation while waiting: interrupted, WebView removed.
  let slow=Task{try await run("visit_page","url",base+"/slow")}
  try await wait{browser.activeWebView != nil};let slowView=browser.activeWebView
  precondition(browser.activity?.phase==WebPhase.navigation && browser.activity?.url==base+"/slow");assertHidden(browser,"slow") // the server holds the response: still navigating
  // The globe button during a run: the live page moves to the viewer, the tool goes on.
  checkHidden=false;browser.showViewer(activate:false);try await wait{!onDisplay().isEmpty}
  precondition(browser.viewer?.isVisible==true && slowView?.window===browser.viewer)
  try await Task.sleep(nanoseconds:700_000_000);slow.cancel()
  await expectError("Tool error: visit_page failed: interrupted"){try await slow.value}
  precondition(browser.activeWebView==nil && browser.parkedWebView==nil && slowView?.superview==nil && browser.viewer==nil && browser.activity==nil && states.last! == nil)
  try await wait{onDisplay().isEmpty};precondition(browser.window?.isVisible==false,"Window left on screen after cancel")
  // Viewer closed by the user during a run: the tool completes in the hidden window.
  let watched=Task{try await run("visit_page","url",base+"/short")}
  try await wait{browser.activeWebView != nil};browser.showViewer(activate:false);browser.viewer?.close()
  precondition(browser.activeWebView?.window===browser.window || browser.activeWebView==nil)
  let watchedResult=try await watched.value;precondition(watchedResult.contains("<markdown>") && browser.viewer==nil);try await wait{onDisplay().isEmpty};checkHidden=true
  browser.showViewer(activate:false);precondition(browser.viewer==nil) // nothing running: no empty window
  browser.shutdown();precondition(!browser.hasWindow)

  // 10. Gating: allowWeb, browser start confirmation (denied, routine), per-action approval; then the full agent path.
  func store(_ mode:String) -> StudioStore {
   let value=StudioStore(storageURL:root.appendingPathComponent(UUID().uuidString+".json"));value.modelState="ready";value.loadedModel="fixture"
   value.mutateSelected{$0.workspacePath=root.path;$0.agentEnabled=true;$0.approvalMode=mode};return value
  }
  let disabled=store("automatic");disabled.mutateSelected{$0.allowWeb=false}
  await expectError(WebTools.disabledMessage){try await disabled.runWebTool(WorkspaceAction(name:"visit_page",arguments:["url":base+"/short"]),session:disabled.selected!,root:root)}
  let denied=store("automatic");let deniedBrowser=WebBrowser{WKWebsiteDataStore.nonPersistent()};denied.webBrowser=deniedBrowser
  let deniedRun=Task{try await denied.runWebTool(WorkspaceAction(name:"visit_page",arguments:["url":base+"/short"]),session:denied.selected!,root:root)}
  try await wait{!denied.approvals.isEmpty};precondition(denied.approvals[0].action.name=="avvio_browser" && denied.approvals[0].action.arguments["richiesta"]==WebTools.startPrompt)
  denied.resolveApproval(denied.approvals[0].id,allowed:false)
  await expectError("Tool error: visit_page failed: avvio del browser non consentito"){try await deniedRun.value}
  precondition(!deniedBrowser.hasWindow && !denied.webBrowserAllowed)
  let routine=store("automatic");routine.webBrowser=WebBrowser{WKWebsiteDataStore.nonPersistent()};routine.activeRoutines[UUID()]=routine.selected!.id
  await expectError("l’avvio del browser richiede approvazione interattiva"){try await routine.runWebTool(WorkspaceAction(name:"web_search",arguments:["query":"x"]),session:routine.selected!,root:root)}
  precondition(routine.approvals.isEmpty && routine.webBrowser?.hasWindow==false)
  let manual=store("manual");let manualID=manual.selected!.id
  let request=ChatMessage(role:"assistant",text:"{\"suc_action\":{\"name\":\"web_search\",\"arguments\":{\"query\":\"swift\"}}}")
  manual.mutateSelected{$0.messages.append(request)};manual.handleAgentResponse(sessionID:manualID,messageID:request.id)
  try await wait{!manual.approvals.isEmpty};precondition(manual.approvals[0].action.name=="web_search" && manual.operationJournal.entries.last?.title=="Ricerca Google: swift")
  manual.resolveApproval(manual.approvals[0].id,allowed:false);await manual.agentTasks[manualID]?.value
  precondition(manual.selected!.actions?.last?.state=="denied" && manual.webBrowser==nil)
  // Full agent path with the dedicated persistent data store: start confirmation once, window reopened without asking, data removal.
  let agent=store("automatic");let agentID=agent.selected!.id
  let visit=ChatMessage(role:"assistant",text:"{\"suc_action\":{\"name\":\"visit_page\",\"arguments\":{\"url\":\"\(base)/short\"}}}")
  agent.mutateSelected{$0.messages.append(visit)};agent.handleAgentResponse(sessionID:agentID,messageID:visit.id)
  try await wait{!agent.approvals.isEmpty};precondition(agent.approvals[0].action.name=="avvio_browser")
  agent.resolveApproval(agent.approvals[0].id,allowed:true);await agent.agentTasks[agentID]?.value
  let record=agent.archive.sessions.first{$0.id==agentID}!.actions!.last!
  precondition(record.state=="completed" && record.result.contains("<markdown>\n# Short page") && agent.webBrowserAllowed && agent.webBrowser?.hasWindow==true,"Agent record: \(record.state) \(record.result)")
  precondition(agent.operationJournal.entries.contains{$0.title=="Apertura pagina: \(base)/short" && $0.state=="Completata"})
  precondition(agent.archive.sessions.first{$0.id==agentID}!.messages.contains{$0.kind=="tool_result" && $0.text.contains("visit_page url=")})
  // 1.4.5_01: web actions without work directory (model actions off): approvals, journal, record, page file in the injected App Support directory.
  let standalone=root.appendingPathComponent("app-support-web");WebTools.standaloneDirectory=standalone
  func bare(_ mode:String?) -> StudioStore {
   let value=StudioStore(storageURL:root.appendingPathComponent(UUID().uuidString+".json"));value.modelState="ready";value.loadedModel="fixture"
   value.mutateSelected{$0.approvalMode=mode};let web=WebBrowser{WKWebsiteDataStore.nonPersistent()};web.searchEndpoint=base+"/results?q=";value.webBrowser=web;return value
  }
  var storeStates:[WebActivity]=[]
  func approveAll(_ value:StudioStore,_ id:UUID,_ seen:inout [ActionApproval]) async throws {
   while value.agentTasks[id] != nil { if let item=value.approvals.first {seen.append(item);value.resolveApproval(item.id,allowed:true)};if let activity=value.webActivity {storeStates.append(activity)};try await Task.sleep(nanoseconds:10_000_000) }
  }
  let plain=bare(nil);let plainID=plain.selected!.id;var seen:[ActionApproval]=[]
  precondition(plain.selected!.workspacePath==nil && plain.selected!.agentEnabled != true)
  let searchRequest=ChatMessage(role:"assistant",text:"```json\n{\"suc_action\":{\"name\":\"web_search\",\"arguments\":{\"query\":\"fixture query\"}}}\n```")
  precondition(!searchRequest.visibleInChat(agent:false,web:true) && searchRequest.visibleInChat(agent:false,web:false))
  plain.mutateSelected{$0.messages.append(searchRequest)};plain.handleAgentResponse(sessionID:plainID,messageID:searchRequest.id)
  try await wait{!plain.approvals.isEmpty};try await approveAll(plain,plainID,&seen)
  precondition(seen.map{$0.action.name}==["web_search","avvio_browser"] && seen[0].directory=="Nessuna directory di lavoro","Approvals: \(seen.map{$0.action.name})")
  precondition(plain.webActivity==nil && plain.webBrowser?.activity==nil && onDisplay().isEmpty)
  var plainSession=plain.archive.sessions.first{$0.id==plainID}!
  precondition(plainSession.actions?.last?.state=="completed" && plainSession.actions!.last!.result.contains("[First result](https://example.com/one)") && plainSession.messages.contains{$0.kind=="tool_result" && $0.text.contains("First result")},"Plain search: \(String(describing:plainSession.actions?.last))")
  precondition(plain.operationJournal.entries.contains{$0.title=="Ricerca Google: fixture query" && $0.state=="Completata"} && plainSession.messages.first{$0.id==searchRequest.id}?.kind=="tool_call")
  let longVisit=ChatMessage(role:"assistant",text:"{\"suc_action\":{\"name\":\"visit_page\",\"arguments\":{\"url\":\"\(base)/long\"}}}")
  plain.mutateSession(plainID){$0.messages.append(longVisit)};plain.handleAgentResponse(sessionID:plainID,messageID:longVisit.id)
  try await wait{!plain.approvals.isEmpty};seen=[];try await approveAll(plain,plainID,&seen)
  precondition(seen.map{$0.action.name}==["visit_page"] && plain.agentIterations[plainID]==2)
  // Store-side button state: tied to the session, running → nil on finish.
  precondition(!storeStates.isEmpty && storeStates.allSatisfy{$0.sessionID==plainID && !$0.verification} && plain.webActivity==nil,"Store states: \(storeStates)")
  plainSession=plain.archive.sessions.first{$0.id==plainID}!
  let visitResult=plainSession.actions!.last!.result, visitPath=String(visitResult.components(separatedBy:"\n")[1].dropFirst("output_path=".count).prefix{$0 != " "})
  precondition(plainSession.actions!.last!.state=="completed" && visitPath.hasPrefix(standalone.path+"/visit-") && visitPath.hasSuffix(".md") && visitResult.contains("<head -100 \(visitPath)>") && visitResult.hasSuffix("</head>\n"+WebTools.standaloneHint+"\n") && !visitResult.contains("read_file con"),"Standalone visit: \(visitResult.prefix(300))")
  let visitFile=try String(contentsOfFile:visitPath,encoding:.utf8)
  precondition(visitFile.contains("Line 149: ") && !FileManager.default.fileExists(atPath:root.appendingPathComponent(".suc-web").path+"/"+URL(fileURLWithPath:visitPath).lastPathComponent))
  plain.webBrowser?.shutdown()
  // Web disabled without the agent: the request is ignored like any other text.
  let blocked=bare("automatic");blocked.mutateSelected{$0.allowWeb=false};let blockedRequest=ChatMessage(role:"assistant",text:"{\"suc_action\":{\"name\":\"web_search\",\"arguments\":{\"query\":\"x\"}}}")
  blocked.mutateSelected{$0.messages.append(blockedRequest)};blocked.handleAgentResponse(sessionID:blocked.selected!.id,messageID:blockedRequest.id)
  precondition(blocked.agentTasks.isEmpty && blocked.approvals.isEmpty && blocked.selected!.actions==nil && blocked.selected!.messages.last?.kind==nil)
  let storeID=agent.archive.settings.webDataStoreID;precondition(storeID != nil)
  agent.webBrowser?.window?.close();precondition(agent.webBrowser?.hasWindow==false)
  let again=try await agent.runWebTool(WorkspaceAction(name:"visit_page",arguments:["url":base+"/short"]),session:agent.selected!,root:root)
  precondition(again.contains("<markdown>") && agent.approvals.isEmpty && agent.webBrowser?.hasWindow==true)
  await agent.clearWebData();precondition(agent.webBrowser==nil && agent.error==nil)
  print("WebToolsTests: hidden browser (never key/front/on a display, unthrottled timers and rAF, lazy scroll, IntersectionObserver, fetch/XHR), globe button state and viewer, verification kept hidden, script hashes, encoding/formats, URL schemes, search extraction, short/long pages, consent, lazy scroll with listener shim, verification page, cancellation, allowWeb/start confirmation/routine/approval gating, agent path, web-only prompt and web actions without work directory passed")
 }
}
