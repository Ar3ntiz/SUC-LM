import SwiftUI
import AppKit
import Darwin
@main struct NativeScrollTests {
 @MainActor static func main() throws {
    setbuf(stdout,nil)
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer{try? FileManager.default.removeItem(at:root)}
    _=NSApplication.shared;NSApplication.shared.setActivationPolicy(.prohibited)
    let store=StudioStore(storageURL:root.appendingPathComponent("sessions.json"))
    store.archive.sessions=(0..<120).map { i in var s=ChatSession();s.title="Sessione \(i)";s.updated=Date(timeIntervalSince1970:Double(i));return s }
    store.archive.selected=store.archive.sessions.last!.id
    store.mutateSelected { s in s.messages=(0..<50).map { i in ChatMessage(role:i%2==0 ? "user":"assistant",text:"Messaggio \(i)\n\n"+String(repeating:"Testo della conversazione. ",count:20)) } }
    let selected=store.archive.selected!
    let host=NSHostingView(rootView:StudioRoot().environmentObject(store))
    let window=NSWindow(contentRect:NSRect(x:0,y:0,width:820,height:560),styleMask:[.borderless],backing:.buffered,defer:false)
    window.contentView=host;window.setFrameOrigin(NSPoint(x:-10000,y:-10000));window.orderFront(nil);defer{window.close()};host.frame=NSRect(x:0,y:0,width:820,height:560)
    func pump(_ seconds:Double=0.4){host.layoutSubtreeIfNeeded();RunLoop.current.run(until:Date().addingTimeInterval(seconds));host.layoutSubtreeIfNeeded()}
    func scrolls(_ v:NSView)->[NSScrollView] { (v as? NSScrollView).map{[$0]} ?? v.subviews.flatMap{scrolls($0)} }
    pump(1)
    let candidates=scrolls(host)
    for v in candidates {print("Scroll:",v.frame,"document:",v.documentView?.frame as Any,"visible:",v.documentVisibleRect)}
    guard let sessions=candidates.first(where:{$0.frame.width<260 && $0.frame.height>200}),let chat=candidates.first(where:{$0.frame.width>400 && $0.frame.height>200}) else{fatalError("Native scroll containers missing")}
    func bottomGap(_ v:NSScrollView)->CGFloat { (v.documentView?.bounds.height ?? 0)-v.documentVisibleRect.maxY }
    assert((sessions.documentView?.bounds.height ?? 0)>sessions.contentView.bounds.height*3)
    let end=max(0,(sessions.documentView?.bounds.height ?? 0)-sessions.contentView.bounds.height)
    sessions.contentView.scroll(to:NSPoint(x:0,y:end));sessions.reflectScrolledClipView(sessions.contentView);pump()
    assert(sessions.documentVisibleRect.minY>100,"Session list did not scroll")
    store.newSession();pump();assert(sessions.documentVisibleRect.minY<40,"New chat not visible")
    store.archive.selected=selected;pump(1)
    assert(bottomGap(chat)<60,"Initial transcript not at bottom: \(bottomGap(chat))")
    store.mutateSelected{$0.messages[$0.messages.count-1].text+=String(repeating:"\nNuova riga in streaming",count:30)};pump(1)
    assert(bottomGap(chat)<60,"Streaming did not follow bottom: \(bottomGap(chat))")
    // Clip-view movement verifies native scrolling; gesture-driven pause is covered
    // by policy tests and requires an interactive trackpad check on the target Mac.
    chat.contentView.scroll(to:NSPoint(x:0,y:max(0,chat.documentVisibleRect.minY-400)))
    chat.reflectScrolledClipView(chat.contentView);pump()
    assert(bottomGap(chat)>100,"Transcript could not be scrolled away from bottom")
    store.archive.selected=store.archive.sessions.first!.id;pump();store.archive.selected=selected;pump(1)
    assert(bottomGap(chat)<60,"Session selection did not resume following")
    print("Native scroll tests passed: 120 sessions, new selection, long transcript, streaming, native movement and selection resume")
 }
}
