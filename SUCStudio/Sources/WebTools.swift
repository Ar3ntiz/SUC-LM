// Web extraction scripts and browsing logic adapted from ds4 (https://github.com/antirez/ds4), MIT License, Copyright (c) 2026 The ds4.c authors.
// Full licence text: licenses/ds4-MIT.txt. The five JavaScript scripts at the end of this file are byte-identical copies of ds4_web.c (commit 9139e2a).
import AppKit
import Foundation
import WebKit

/// web_search / visit_page: ds4's web module on WKWebView (timings, limits and output formats as in docs/WEB_TOOLS_SPEC_IT.md §1).
enum WebTools {
    static let names=["web_search","google_search","visit_page"]
    static let viewport=CGSize(width:1365,height:900)
    static let evaluationTimeout:UInt64=20_000_000_000
    static let maxResultBytes=1024*1024 // ds4 caps a CDP message at 4 × this value
    static let headLines=100, headBytes=8*1024
    static let startPrompt="Lo strumento web vuole usare il browser integrato (nascosto; puoi guardarlo dal pulsante globo). Consenti?"
    static let verificationMessage="Google ha chiesto una verifica: apri il browser integrato dal pulsante globo arancione nella barra della chat, completala e ripeti la ricerca."
    static let verificationLabel="Verifica Google richiesta: clicca per completarla"
    static let verificationStatus="Verifica Google richiesta: completala in questa pagina, poi ripeti la ricerca."
    static let completedStatus="Verifica completata: ripeti la ricerca. Puoi chiudere questa finestra."
    static let disabledMessage="Strumenti web disabilitati per questa sessione."
    static func isSearch(_ name:String)->Bool { name != "visit_page" }
    static func failure(_ text:String)->NSError { NSError(domain:"SUCStudio.Web",code:1,userInfo:[NSLocalizedDescriptionKey:text]) }
    static func failure(_ tool:String,_ reason:String)->NSError { failure("Tool error: \(tool) failed: \(reason)") }
    /// Checks argument, URL scheme and session permission; returns the query or URL.
    @discardableResult static func validate(_ action:WorkspaceAction,allowWeb:Bool) throws -> String {
        guard names.contains(action.name) else { throw MCPConnection.failure("Strumento sconosciuto: \(action.name)") }
        let key=isSearch(action.name) ? "query":"url"
        guard let value=action.arguments[key],!value.isEmpty else { throw failure("Tool error: \(action.name) requires \(key)") }
        if !isSearch(action.name) {
            guard let url=URL(string:value),let scheme=url.scheme?.lowercased(),["http","https"].contains(scheme),url.host?.isEmpty==false else { throw failure(action.name,"solo URL http e https sono ammessi") }
        }
        guard allowWeb else { throw MCPConnection.failure(disabledMessage) }
        return value
    }
    static func journalTitle(_ action:WorkspaceAction)->String? {
        guard names.contains(action.name) else{return nil}
        return isSearch(action.name) ? "Ricerca Google: "+(action.arguments["query"] ?? "") : "Apertura pagina: "+(action.arguments["url"] ?? "")
    }
    /// ds4 web_url_encode: unreserved ASCII kept, every other byte percent-encoded.
    static func encode(_ value:String)->String {
        var out=""
        for byte in value.utf8 {
            if (byte>=48 && byte<=57) || (byte>=65 && byte<=90) || (byte>=97 && byte<=122) || [45,95,46,126].contains(byte) { out.append(Character(UnicodeScalar(byte))) }
            else { out += String(format:"%%%02X",byte) }
        }
        return out
    }
    /// ds4 agent_tool_visit_page result: full Markdown in a file, head of 100 lines / 8 KiB for the model.
    /// Without a work directory the page goes to an absolute path for the user (read_file is unavailable there).
    static let standaloneHint="Il file completo è salvato per l’utente; senza directory di lavoro read_file non è disponibile."
    /// Pages visited without a work directory; tests inject a temporary directory.
    static var standaloneDirectory=FileManager.default.urls(for:.applicationSupportDirectory,in:.userDomainMask)[0].appendingPathComponent(StudioPlatform.dataDirectory,isDirectory:true).appendingPathComponent("web",isDirectory:true)
    static func formatVisit(url:String,path:String,markdown:String,workspace:Bool=true)->String {
        let bytes=Array(markdown.utf8)
        var total=bytes.reduce(0){$0+($1==10 ? 1:0)};if let last=bytes.last,last != 10{total+=1}
        var used=0,lines=0
        while used<bytes.count && used<headBytes && lines<headLines { if bytes[used]==10{lines+=1};used+=1 }
        let byteLimited=used<bytes.count && used>=headBytes
        if byteLimited { while used>0 && used<bytes.count && bytes[used] & 0xC0 == 0x80 { used-=1 } } // never split a UTF-8 sequence
        if used>0 && bytes[used-1] != 10 && lines<headLines { lines+=1 }
        let head=String(decoding:bytes[0..<used],as:UTF8.self), newline=head.isEmpty || head.hasSuffix("\n") ? "":"\n"
        var out="visit_page url=\(url)\noutput_path=\(path) (\(bytes.count) bytes, \(total) lines\(workspace ? "":"; file per l’utente"))\n"
        if byteLimited || lines<total { out += "<head -\(headLines) \(path)>\n"+head+newline+"</head>\n"+(workspace ? "Usa read_file con path=\(path) per leggere il resto del Markdown.":standaloneHint)+"\n" }
        else { out += "<markdown>\n"+head+newline+"</markdown>\n" }
        return out
    }
    /// Writes the rendered page to <workdir>/.suc-web/visit-<yyyyMMdd-HHmmss>-<n>.md and returns the relative path.
    static func store(_ markdown:String,root:URL,date:Date=Date()) throws -> String {
        let directory=try WorkspaceTools.resolve(".suc-web",root:root)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        return try write(markdown,date:date){(".suc-web/"+$0,try WorkspaceTools.resolve(".suc-web/"+$0,root:root))}
    }
    /// Writes the page to <standaloneDirectory>/visit-<yyyyMMdd-HHmmss>-<n>.md and returns the absolute path.
    static func storeStandalone(_ markdown:String,date:Date=Date()) throws -> String {
        let directory=standaloneDirectory
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        return try write(markdown,date:date){(directory.appendingPathComponent($0).path,directory.appendingPathComponent($0))}
    }
    private static func write(_ markdown:String,date:Date,target:(String) throws -> (String,URL)) throws -> String {
        let formatter=DateFormatter();formatter.locale=Locale(identifier:"en_US_POSIX");formatter.dateFormat="yyyyMMdd-HHmmss"
        let stamp=formatter.string(from:date)
        for n in 1...10_000 {
            let name="visit-\(stamp)-\(n).md"
            let (path,url)=try target(name)
            do { try Data(markdown.utf8).write(to:url,options:.withoutOverwriting);return path } catch let error as CocoaError where error.code == .fileWriteFileExists { continue }
        }
        throw MCPConnection.failure("could not store rendered page")
    }
    /// root nil: no work directory (web-only sessions); visit_page then stores the page in standaloneDirectory.
    @MainActor static func execute(_ action:WorkspaceAction,root:URL?,browser:WebBrowser,sessionID:UUID?=nil,log:@escaping @MainActor (String)->Void={_ in}) async throws -> String {
        let target=try validate(action,allowWeb:true)
        if isSearch(action.name) {
            guard let url=URL(string:browser.searchEndpoint+encode(target)) else { throw failure(action.name,"URL di ricerca non valido") }
            return try await browser.run(tool:action.name,target:target,sessionID:sessionID,url:url,extractor:searchScript,scroll:false,log:log)
        }
        let markdown=try await browser.run(tool:action.name,target:target,sessionID:sessionID,url:URL(string:target)!,extractor:pageScript,scroll:true,log:log)
        let path: String
        do { path=try root.map{try store(markdown,root:$0)} ?? storeStandalone(markdown) } catch { throw failure(action.name,error.localizedDescription) }
        return formatVisit(url:target,path:path,markdown:markdown,workspace:root != nil)
    }
}

/// Resumes a continuation at most once (JavaScript completion, timeout or cancellation).
private final class ResumeOnce: @unchecked Sendable {
    private let lock=NSLock()
    private var continuation: CheckedContinuation<String,Error>?
    func set(_ value:CheckedContinuation<String,Error>) { lock.lock();continuation=value;lock.unlock() }
    func resume(_ result:Result<String,Error>) { lock.lock();let value=continuation;continuation=nil;lock.unlock();value?.resume(with:result) }
}

/// Live state of one web action, shown by the globe button in the chat toolbar and by the viewer status bar (1.4.5_Final).
struct WebActivity: Equatable {
    var sessionID: UUID?
    var tool: String
    var target: String
    var phase: String
    var url: String
    var startedAt: Date
    /// Google asked for a verification: the action has failed, the page stays loaded for the user.
    var verification=false
    var title: String { WebTools.isSearch(tool) ? "Ricerca Google: "+target : "Apertura pagina: "+target }
    func elapsed(_ now:Date)->Int { max(0,Int(now.timeIntervalSince(startedAt))) }
    /// Button tooltip and accessibility label.
    func label(now:Date=Date())->String { verification ? WebTools.verificationLabel : "\(title) · \(phase) · \(elapsed(now)) s" }
    /// Viewer status bar: phase, URL, elapsed time.
    func status(now:Date=Date())->String { verification ? WebTools.verificationStatus : "\(phase) · \(url) · \(elapsed(now)) s" }
}
enum WebPhase { static let navigation="Navigazione", loading="Attesa caricamento", consent="Consenso cookie", scrolling="Scorrimento", extraction="Estrazione" }

/// Host of the tool's WKWebView while nobody watches: it can never become key or main and never activates the app.
final class HiddenBrowserPanel: NSPanel {
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }
}
/// Top-left anchored container: the web view keeps ds4's 1365×900 viewport whatever the viewer size.
private final class FlippedView: NSView { override var isFlipped: Bool { true } }

/// ds4's browser on WKWebView with a dedicated, persistent website data store.
///
/// 1.4.5_Final, hidden but not throttled. Measured on tests/web_fixture.py (docs/WEB_TOOLS_SPEC_IT.md §2bis): a WKWebView whose window is
/// never ordered in, ordered out, detached, or simply covered by another app's windows (the 1.4.5_01 visible window while the user works
/// elsewhere) reports visibilityState "hidden": requestAnimationFrame stops and WebKit aligns nested timers to 1 s, so ds4's 900 ms scroll
/// steps become 1000 ms (one of them 1.2-1.7 s). Chosen approach: a borderless, non-activating HiddenBrowserPanel (alpha 0, ignores the
/// mouse, on all Spaces) placed beyond every display and ordered in behind all windows, plus WebKit's _setWindowOcclusionDetectionEnabled:NO
/// on each web view, so the necessarily occluded window is not treated as hidden. Result: visibilityState "visible", rAF ~60 Hz, setTimeout
/// 250/900 ms at 250-253/900-905 ms, lazy scroll, IntersectionObserver, fetch/XHR and consent pages as with an unoccluded visible window;
/// no window on any display, never key, the app never activated. If a future WebKit drops the selector the tool still works (throttled).
/// The visible "Browser SUC Studio" window (viewer) exists only when the user opens it from the globe button.
@MainActor final class WebBrowser: NSObject, WKNavigationDelegate, NSWindowDelegate {
    static let title="Browser SUC Studio"
    static let statusHeight: CGFloat=24
    var searchEndpoint="https://www.google.com/search?q="
    private let makeDataStore: () -> WKWebsiteDataStore
    private var configuration: WKWebViewConfiguration?
    /// The hidden host panel (ordered in, off-screen, only while it holds a web view).
    private(set) var window: NSWindow?
    /// The viewer opened by the user from the globe button; nil otherwise.
    private(set) var viewer: NSWindow?
    private var viewerContainer: NSView?
    private var viewerStatus: NSTextField?
    private var viewerTimer: Timer?
    private(set) var activeWebView: WKWebView?
    /// The page kept loaded after a verification request, so that the user can complete it from the viewer.
    private(set) var parkedWebView: WKWebView?
    /// Current web action; nil when idle. Published to the store through onActivity.
    private(set) var activity: WebActivity? { didSet { if activity != oldValue { onActivity?(activity) };updateViewerStatus() } }
    var onActivity: (@MainActor (WebActivity?) -> Void)?
    private var navigationError: Error?
    private var busy=false
    private var waiters: [CheckedContinuation<Void,Never>]=[]
    var hasWindow: Bool { window != nil }
    init(dataStore: @escaping () -> WKWebsiteDataStore) { makeDataStore=dataStore }
    /// Closes every window and drops the configuration so that the data store can be removed.
    func shutdown() { closeViewer();releaseParked();activity=nil;window?.delegate=nil;window?.close();window=nil;configuration=nil }
    func windowWillClose(_ notification: Notification) {
        guard let closing=notification.object as? NSWindow else {return}
        if closing===viewer {
            viewerTimer?.invalidate();viewerTimer=nil;viewer=nil;viewerContainer=nil;viewerStatus=nil
            // Closing the viewer never affects the tool: the live (or verification) page goes back to the hidden host.
            if let webView=activeWebView ?? (activity != nil ? parkedWebView:nil) { hostHidden(webView) } else { releaseParked() }
        } else if closing===window { parkedWebView=nil;if activeWebView==nil {activity=nil};window=nil }
    }
    private func acquire() async { if !busy {busy=true;return};await withCheckedContinuation{waiters.append($0)} }
    private func release() { if waiters.isEmpty {busy=false} else {waiters.removeFirst().resume()} }
    private func ensureWindow() -> NSWindow {
        if let window {return window}
        let value=HiddenBrowserPanel(contentRect:NSRect(origin:.zero,size:WebTools.viewport),styleMask:[.borderless,.nonactivatingPanel],backing:.buffered,defer:false)
        value.title=Self.title;value.isReleasedWhenClosed=false;value.delegate=self;value.contentView=FlippedView(frame:NSRect(origin:.zero,size:WebTools.viewport))
        value.hidesOnDeactivate=false;value.ignoresMouseEvents=true;value.alphaValue=0;value.hasShadow=false;value.isExcludedFromWindowsMenu=true
        value.collectionBehavior=[.canJoinAllSpaces,.transient,.ignoresCycle,.fullScreenAuxiliary]
        window=value;return value
    }
    /// Beyond the union of all displays (recomputed every time: displays change).
    static func offscreenOrigin() -> NSPoint {
        let frame=NSScreen.screens.reduce(NSRect.zero){$0.union($1.frame)}
        return NSPoint(x:frame.maxX+4000,y:frame.maxY+4000)
    }
    private static func keepVisibleWhenOccluded(_ webView:WKWebView) {
        let selector=NSSelectorFromString("_setWindowOcclusionDetectionEnabled:")
        if webView.responds(to:selector) { webView.perform(selector,with:nil) } // nil is BOOL NO
    }
    private func move(_ webView:WKWebView,into container:NSView) {
        if webView.superview !== container { webView.removeFromSuperview();webView.frame=NSRect(origin:.zero,size:WebTools.viewport);container.addSubview(webView) }
    }
    /// Puts the page in the hidden panel, ordered in behind every window and off every display; never ordered front, never key.
    private func hostHidden(_ webView:WKWebView) {
        let window=ensureWindow()
        move(webView,into:window.contentView!)
        window.setFrameOrigin(Self.offscreenOrigin());window.order(.below,relativeTo:0)
    }
    private func hideHost() { window?.orderOut(nil) }
    private func releaseParked() {
        guard let webView=parkedWebView else {return}
        parkedWebView=nil;webView.stopLoading();webView.navigationDelegate=nil;webView.removeFromSuperview()
        if activeWebView==nil {hideHost()}
    }
    private func closeViewer() { viewer?.close() }
    private func updateViewerStatus() { viewerStatus?.stringValue=activity?.status() ?? (parkedWebView != nil ? WebTools.completedStatus:"") }
    private func makeViewer() -> NSWindow {
        let size=NSSize(width:WebTools.viewport.width,height:WebTools.viewport.height+Self.statusHeight)
        let value=NSWindow(contentRect:NSRect(origin:.zero,size:size),styleMask:[.titled,.closable,.miniaturizable,.resizable],backing:.buffered,defer:false)
        value.title=Self.title;value.isReleasedWhenClosed=false;value.delegate=self
        let root=NSView(frame:NSRect(origin:.zero,size:size))
        let container=FlippedView(frame:NSRect(x:0,y:Self.statusHeight,width:size.width,height:WebTools.viewport.height));container.autoresizingMask=[.width,.height]
        let status=NSTextField(labelWithString:"");status.frame=NSRect(x:10,y:4,width:size.width-20,height:16);status.autoresizingMask=[.width,.maxYMargin]
        status.font = .systemFont(ofSize:11);status.textColor = .secondaryLabelColor;status.lineBreakMode = .byTruncatingMiddle
        root.addSubview(container);root.addSubview(status);value.contentView=root;value.center()
        viewer=value;viewerContainer=container;viewerStatus=status
        viewerTimer=Timer.scheduledTimer(withTimeInterval:1,repeats:true){[weak self] _ in MainActor.assumeIsolated{self?.updateViewerStatus()}}
        return value
    }
    /// The globe button: shows the live page (or the pending verification) with a status bar. Only the user's click brings it to the front.
    func showViewer(activate:Bool=true) {
        guard let webView=activeWebView ?? parkedWebView else {return} // nothing to watch: never an empty window
        let value=viewer ?? makeViewer()
        move(webView,into:viewerContainer!);hideHost();updateViewerStatus()
        if activate {NSApplication.shared.activate()}
        value.makeKeyAndOrderFront(nil)
    }
    private func setPhase(_ phase:String,_ webView:WKWebView) {
        activity?.phase=phase
        if let url=webView.url?.absoluteString,!url.isEmpty,!url.hasPrefix("about:") {activity?.url=url}
    }
    private func sharedConfiguration() -> WKWebViewConfiguration {
        if let configuration {return configuration}
        let value=WKWebViewConfiguration();value.websiteDataStore=makeDataStore()
        // Counts scroll listeners for the dynamic-scroll script (Chrome's getEventListeners exists only in its Command Line API).
        value.userContentController.addUserScript(WKUserScript(source:WebTools.listenerShim,injectionTime:.atDocumentStart,forMainFrameOnly:true,in:.page))
        configuration=value;return value
    }
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        let code=(error as NSError).code
        guard webView===activeWebView, !(code==NSURLErrorCancelled && (error as NSError).domain==NSURLErrorDomain), !(code==102 && (error as NSError).domain=="WebKitErrorDomain") else{return}
        navigationError=error
    }
    /// The user completed the verification in the viewer: the button disappears, the page stays until the viewer is closed.
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        guard webView===parkedWebView, activeWebView==nil else {return}
        Task { @MainActor in
            guard webView===self.parkedWebView, self.activeWebView==nil, (try? await self.verificationRequired(webView))==false else {return}
            self.activity=nil
            if self.viewer==nil {self.releaseParked()} else {self.updateViewerStatus()}
        }
    }
    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction, decisionHandler: @escaping @MainActor (WKNavigationActionPolicy) -> Void) {
        let scheme=navigationAction.request.url?.scheme?.lowercased() ?? ""
        decisionHandler(["http","https","about","blob","data"].contains(scheme) ? .allow:.cancel)
    }
    private func pause(_ milliseconds:UInt64) async throws { try await Task.sleep(nanoseconds:milliseconds*1_000_000) }
    private static func javaScript(_ webView:WKWebView,_ body:String) async throws -> String {
        let box=ResumeOnce()
        return try await withTaskCancellationHandler(operation:{
            try await withCheckedThrowingContinuation { continuation in
                box.set(continuation)
                if Task.isCancelled { box.resume(.failure(CancellationError()));return }
                webView.callAsyncJavaScript(body,arguments:[:],in:nil,in:.page) { result in
                    switch result {
                    case .success(let value): box.resume(value is String ? .success(value as! String) : .failure(WebTools.failure("JavaScript evaluation did not return a string")))
                    case .failure: box.resume(.failure(WebTools.failure("JavaScript evaluation failed")))
                    }
                }
            }
        },onCancel:{ box.resume(.failure(CancellationError())) })
    }
    /// One evaluation in the page world with ds4's 20 s timeout and result size cap.
    func evaluate(_ webView:WKWebView,_ body:String) async throws -> String {
        let value=try await withThrowingTaskGroup(of:String.self) { group in
            group.addTask { @MainActor in try await Self.javaScript(webView,body) }
            group.addTask { try await Task.sleep(nanoseconds:WebTools.evaluationTimeout);throw WebTools.failure("timeout") }
            defer{group.cancelAll()}
            return try await group.next()!
        }
        guard value.utf8.count<=WebTools.maxResultBytes*4 else { throw WebTools.failure("result too large") }
        return value
    }
    /// ds4 web_wait_navigated_ready: up to 100 probes every 250 ms.
    func waitNavigatedReady(_ webView:WKWebView) async throws {
        var lastLength = -1, stable=0, sawRealURL=false
        for i in 0..<100 {
            try Task.checkCancellation()
            if let navigationError { throw WebTools.failure(navigationError.localizedDescription) }
            let probe: String
            do { probe=try await evaluate(webView,"return "+WebTools.probeScript+";") } catch { try Task.checkCancellation();try await pause(250);continue }
            let parts=probe.components(separatedBy:"\n")
            guard parts.count>=3 else { try await pause(250);continue }
            let href=parts[0], ready=parts[1], length=Int(parts[2]) ?? 0
            if !href.isEmpty && href != "about:blank" && !href.hasPrefix("about:") && !href.hasPrefix("chrome://") { sawRealURL=true;if webView===activeWebView && activity?.phase==WebPhase.navigation {setPhase(WebPhase.loading,webView)} }
            let readyState=ready=="complete" || ready=="interactive"
            if length>0 && length==lastLength {stable+=1} else {stable=0}
            lastLength=length
            if sawRealURL && readyState && length>0 && stable>=2 { try await pause(500);return }
            if sawRealURL && readyState && i>=24 { return }
            try await pause(250)
        }
    }
    /// Google's /sorry/ page or a reCAPTCHA: never bypassed, the user completes it in the window.
    func verificationRequired(_ webView:WKWebView) async throws -> Bool {
        do {
            let value=try await evaluate(webView,"return "+WebTools.verificationProbe+";")
            let parts=value.components(separatedBy:"\n")
            return parts[0].contains("/sorry/") || parts.dropFirst().contains("1")
        } catch { try Task.checkCancellation();return false }
    }
    /// One ds4 "tab": a new WKWebView per call, removed at the end together with the hidden host and the viewer.
    func run(tool:String,target:String="",sessionID:UUID?=nil,url:URL,extractor:String,scroll:Bool,log:@escaping @MainActor (String)->Void) async throws -> String {
        await acquire();defer{release()}
        releaseParked() // a new action replaces a pending verification page
        let webView=WKWebView(frame:NSRect(origin:.zero,size:WebTools.viewport),configuration:sharedConfiguration())
        webView.navigationDelegate=self;Self.keepVisibleWhenOccluded(webView)
        activeWebView=webView;navigationError=nil
        activity=WebActivity(sessionID:sessionID,tool:tool,target:target.isEmpty ? url.absoluteString:target,phase:WebPhase.navigation,url:url.absoluteString,startedAt:Date())
        if let viewerContainer {move(webView,into:viewerContainer)} else {hostHidden(webView)}
        defer {
            activeWebView=nil
            // Finished (success, error or cancel): no page left behind, no empty window on screen.
            if parkedWebView !== webView { webView.stopLoading();webView.navigationDelegate=nil;webView.removeFromSuperview();activity=nil;closeViewer();hideHost() }
        }
        do {
            try Task.checkCancellation()
            webView.load(URLRequest(url:url))
            try await waitNavigatedReady(webView)
            if try await verificationRequired(webView) { park(webView);throw WebTools.failure(tool,WebTools.verificationMessage) }
            var clicked=""
            setPhase(WebPhase.consent,webView)
            do { clicked=try await evaluate(webView,"return "+WebTools.consentScript+";") } catch { try Task.checkCancellation() }
            if !clicked.isEmpty {
                log(clicked);try await pause(1500)
                setPhase(WebPhase.loading,webView)
                do { try await waitNavigatedReady(webView) } catch { try Task.checkCancellation() }
                if try await verificationRequired(webView) { park(webView);throw WebTools.failure(tool,WebTools.verificationMessage) }
            }
            if scroll {
                setPhase(WebPhase.scrolling,webView)
                do { log(try await evaluate(webView,"return await "+WebTools.scrollScript+";")) } catch { try Task.checkCancellation();log("scroll: "+error.localizedDescription) }
            }
            try Task.checkCancellation()
            setPhase(WebPhase.extraction,webView)
            return try await evaluate(webView,"return "+extractor+";")
        } catch {
            if error is CancellationError || Task.isCancelled { throw WebTools.failure(tool,"interrupted") }
            if (error as NSError).domain=="SUCStudio.Web",error.localizedDescription.hasPrefix("Tool error:") { throw error }
            throw WebTools.failure(tool,error.localizedDescription)
        }
    }
    /// Google's verification: never bypassed and never brought to the front; the page stays loaded and the globe button turns orange.
    private func park(_ webView:WKWebView) { parkedWebView=webView;setPhase(activity?.phase ?? WebPhase.loading,webView);activity?.verification=true }
}

extension StudioStore {
    var webDataStoreID: UUID {
        if let id=archive.settings.webDataStoreID {return id}
        let id=UUID();archive.settings.webDataStoreID=id;scheduleSave();return id
    }
    /// Scheduled routines (and their subagents) have nobody to answer the browser start confirmation.
    func webSessionUnattended(_ session:ChatSession)->Bool {
        let routines=Set(activeRoutines.values)
        return routines.contains(session.id) || session.parentSessionID.map{routines.contains($0)}==true
    }
    func runWebTool(_ action:WorkspaceAction,session:ChatSession,root:URL?) async throws -> String {
        try WebTools.validate(action,allowWeb:session.allowWeb != false)
        let browser: WebBrowser
        if let webBrowser {browser=webBrowser} else { let id=webDataStoreID;browser=WebBrowser{WKWebsiteDataStore(forIdentifier:id)};webBrowser=browser }
        if !webBrowserAllowed && !browser.hasWindow {
            if session.approvalMode=="automatic" {
                // The enabled web action already follows the session approval policy,
                // including the first browser launch in interactive chats.
                webBrowserAllowed=true
            } else {
                let request=WorkspaceAction(name:"avvio_browser",arguments:["richiesta":WebTools.startPrompt])
                let allowed=await approve(request,session:session,root:root,always:true)
                guard allowed,!Task.isCancelled else { throw WebTools.failure(action.name,Task.isCancelled ? "interrupted":"avvio del browser non consentito dall’utente") }
                webBrowserAllowed=true
            }
        }
        browser.onActivity={[weak self] in self?.webActivity=$0}
        return try await WebTools.execute(action,root:root,browser:browser,sessionID:session.id){[weak self] in self?.appendLog("Web · "+$0)}
    }
    /// Globe button in the chat toolbar.
    func showWebBrowser() { webBrowser?.showViewer() }
    func clearWebData() async {
        guard let id=archive.settings.webDataStoreID else { appendLog("Nessun dato del browser web da cancellare.");return }
        webBrowser?.shutdown();webBrowser=nil
        for attempt in 0..<20 {
            do { try await WKWebsiteDataStore.remove(forIdentifier:id);appendLog("Dati del browser web cancellati.");return }
            catch { if attempt==19 {self.error="Impossibile cancellare i dati del browser web: "+error.localizedDescription};try? await Task.sleep(nanoseconds:250_000_000) }
        }
    }
}

extension WebTools {
    /// Own addition (not from ds4): detects Google's verification page without interacting with it.
    static let verificationProbe=#"location.href+'\n'+(document.querySelector('iframe[src*="recaptcha"]')?'1':'0')+'\n'+(/unusual traffic|traffico insolito/i.test((document.body&&document.body.innerText)||'')?'1':'0')"#
    static let listenerShim=##"""
(() => {
  const counts = new WeakMap();
  const add = EventTarget.prototype.addEventListener, rm = EventTarget.prototype.removeEventListener;
  EventTarget.prototype.addEventListener = function(t, l, o) {
    if (t === 'scroll') counts.set(this, (counts.get(this) || 0) + 1);
    return add.call(this, t, l, o); };
  EventTarget.prototype.removeEventListener = function(t, l, o) {
    if (t === 'scroll' && counts.get(this)) counts.set(this, counts.get(this) - 1);
    return rm.call(this, t, l, o); };
  if (typeof window.getEventListeners !== 'function')
    window.getEventListeners = o => ({ scroll: new Array(counts.get(o) || 0).fill({}) });
})();
"""##
    /// ds4 §4.1 readiness probe.
    static let probeScript=##"""
location.href+'\n'+document.readyState+'\n'+((document.body&&document.body.innerText)||'').length
"""##
    /// ds4 §4.2 Google cookie consent.
    static let consentScript=##"""
(() => {const clean=s=>(s||'').replace(/\s+/g,' ').trim();const pats=[/accept all/i,/i agree/i,/agree/i,/accetta tutto/i,/tout accepter/i,/aceptar todo/i,/alle akzeptieren/i];const els=[...document.querySelectorAll('button,[role=button],input[type=submit],a')];for (const el of els){const t=clean(el.innerText||el.value||el.textContent);if(!t)continue; if(pats.some(p=>p.test(t))){el.click(); return 'clicked '+t;}}return '';})()
"""##
    /// ds4 §4.3 search results extraction (web_search).
    static let searchScript=##"""
(() => {const clean=s=>(s||'').replace(/\s+/g,' ').trim();const esc=s=>clean(s).replace(/\\/g,'\\\\').replace(/\[/g,'\\[').replace(/\]/g,'\\]').replace(/\n/g,' ');const visible=el=>{const r=el.getBoundingClientRect();const st=getComputedStyle(el);return r.width>0&&r.height>0&&st.display!=='none'&&st.visibility!=='hidden'&&st.opacity!=='0';};const bad=h=>(/(^|\.)google\./.test(h)||/(^|\.)gstatic\./.test(h)||/(^|\.)googleusercontent\./.test(h));const lines=['# Google search results','',`URL: ${location.href}`,'','## Visible links'];const seen=new Set();for(const a of document.querySelectorAll('a[href]')){if(!visible(a))continue;let href=a.href||'';try{const u=new URL(href);if(u.pathname==='/url'&&u.searchParams.get('q'))href=u.searchParams.get('q');}catch{}let u;try{u=new URL(href);}catch{continue;}if(!/^https?:$/.test(u.protocol))continue;if(bad(u.hostname))continue;const text=esc(a.innerText||a.textContent);if(text.length<3)continue;if(seen.has(u.href))continue;seen.add(u.href);lines.push(`- [${text.slice(0,180)}](${u.href})`);if(seen.size>=20)break;}lines.push('','## Text snapshot',clean(document.body.innerText).slice(0,1200));return lines.join('\n');})()
"""##
    /// ds4 §4.4 page extraction (visit_page).
    static let pageScript=##"""
(() => {const clean=s=>(s||'').replace(/\s+/g,' ').trim();const esc=s=>clean(s).replace(/\\/g,'\\\\').replace(/\[/g,'\\[').replace(/\]/g,'\\]').replace(/\n/g,' ');const visible=el=>{const r=el.getBoundingClientRect();const st=getComputedStyle(el);return r.width>0&&r.height>0&&st.display!=='none'&&st.visibility!=='hidden'&&st.opacity!=='0';};const inline=n=>{if(!n)return'';if(n.nodeType===3)return n.nodeValue;if(n.nodeType!==1)return'';const el=n;if(el.tagName==='SCRIPT'||el.tagName==='STYLE'||el.tagName==='NOSCRIPT')return'';if(el.tagName==='A'){const t=esc(el.innerText||el.textContent);const h=el.href||'';return t&&h?`[${t}](${h})`:t;}if(el.tagName==='CODE')return '`'+clean(el.innerText||el.textContent).replace(/`/g,'\\`')+'`';return [...el.childNodes].map(inline).join('');};const lines=[`# ${clean(document.title)||location.href}`,'',`URL: ${location.href}`,'','## Content'];const blocks=[...document.body.querySelectorAll('h1,h2,h3,h4,h5,h6,p,li,pre,blockquote,td,th,[id="content-text"],[class*="comment-body"],[class*="comment-content"],[data-testid*="comment-text"]')];const seen=new Set();for(const el of blocks){if(!visible(el))continue;let s='';const tag=el.tagName;if(/^H[1-6]$/.test(tag)){s='#'.repeat(Number(tag[1]))+' '+inline(el);}else if(tag==='LI'){s='- '+inline(el);}else if(tag==='PRE'){s='```\n'+(el.innerText||el.textContent||'').trimEnd()+'\n```';}else if(tag==='BLOCKQUOTE'){s='> '+clean(el.innerText||el.textContent);}else{s=inline(el);}s=s.trim();if(!s||seen.has(s))continue;seen.add(s);lines.push('',s);if(lines.join('\n').length>900000){lines.push('','[Content truncated by browser extractor.]');break;}}lines.push('','## Visible links');let n=0;const linkSeen=new Set();for(const a of document.querySelectorAll('a[href]')){if(!visible(a))continue;const t=esc(a.innerText||a.textContent);if(t.length<3)continue;let u;try{u=new URL(a.href);}catch{continue;}if(!/^https?:$/.test(u.protocol)||linkSeen.has(u.href))continue;linkSeen.add(u.href);lines.push(`- [${t.slice(0,160)}](${u.href})`);if(++n>=80)break;}return lines.join('\n');})()
"""##
    /// ds4 §4.5 dynamic scroll (visit_page only).
    static let scrollScript=##"""
(() => new Promise(resolve => {const root=()=>document.scrollingElement||document.documentElement||document.body;const blockSel='h1,h2,h3,h4,h5,h6,p,li,pre,blockquote,td,th,[id="content-text"],[class*="comment-body"],[class*="comment-content"],[data-testid*="comment-text"]';const lazySel='[onscroll],[loading="lazy"],[data-src],[data-lazy],[class*="lazy"],[class*="infinite"],[class*="virtual"],[role="feed"],[id*="comment"],[class*="comment"],[data-testid*="comment"]';const hookCount=()=>{let n=0;try{if(window.onscroll)n++;if(document.onscroll)n++;if(document.body&&document.body.onscroll)n++;}catch(e){}try{if(typeof getEventListeners==='function'){for(const o of [window,document,document.body]){if(!o)continue;const ev=getEventListeners(o);if(ev&&ev.scroll)n+=ev.scroll.length;}}}catch(e){}try{n+=document.querySelectorAll(lazySel).length;}catch(e){}return n;};const metrics=()=>{const r=root();return {height:r?r.scrollHeight:0,view:innerHeight||900,y:scrollY||(r&&r.scrollTop)||0,text:((document.body&&document.body.innerText)||'').length,links:document.links?document.links.length:0,blocks:document.body?document.body.querySelectorAll(blockSel).length:0,hooks:hookCount()};};const sig=m=>[m.height,m.text,m.links,m.blocks].join('|');const grew=(a,b)=>b.height>a.height+20||b.text>a.text+200||b.links>a.links+2||b.blocks>a.blocks+2;const scrollOnce=()=>{const r=root();if(!r)return;const h=Math.max(700,Math.floor((innerHeight||900)*0.85));window.scrollTo(0,Math.min(r.scrollHeight,(scrollY||r.scrollTop||0)+h));};let last=metrics(),lastSig=sig(last),same=0,steps=0;const scrollable=last.height>last.view*1.35;if(!scrollable||last.hooks===0){resolve('scroll skipped hooks='+last.hooks+' text='+last.text);return;}const tick=()=>{if(steps>=28){resolve('scrolled '+steps+' text='+last.text);return;}const before=last;scrollOnce();steps++;setTimeout(()=>{const now=metrics(),nowSig=sig(now);if(nowSig===lastSig)same++;else same=0;const loaded=grew(before,now);last=now;lastSig=nowSig;if(steps===1&&!loaded){resolve('scroll probe unchanged text='+now.text);return;}const atBottom=now.y+now.view+20>=now.height;if(same>=4||(atBottom&&same>=1)){resolve('scrolled '+steps+' text='+now.text);return;}tick();},900);};tick();}))()
"""##
}
