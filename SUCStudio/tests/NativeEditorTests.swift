import SwiftUI
import AppKit
@main struct NativeEditorTests {
 @MainActor static func main() throws {
    _=NSApplication.shared;NSApplication.shared.setActivationPolicy(.prohibited)
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer{try? FileManager.default.removeItem(at:root)}
    let store=StudioStore(storageURL:root.appendingPathComponent("sessions.json"));store.loadedModel="fixture";store.modelState="ready"
    let host=NSHostingView(rootView:StudioRoot().environmentObject(store))
    let window=NSWindow(contentRect:NSRect(x:0,y:0,width:960,height:720),styleMask:[.borderless],backing:.buffered,defer:false)
    window.contentView=host;host.frame=NSRect(x:0,y:0,width:960,height:720)
    func pump(_ duration:Double=0.3){host.layoutSubtreeIfNeeded();RunLoop.current.run(until:Date().addingTimeInterval(duration));host.layoutSubtreeIfNeeded()}
    func editor(_ view:NSView)->ComposeEditor.SendTextView? {if let editor=view as? ComposeEditor.SendTextView{return editor};return view.subviews.compactMap{editor($0)}.first}
    pump();guard let input=editor(host) else{fatalError("Composer missing")}
    let trigger=ComposeEditor.SendTextView(frame:NSRect(x:0,y:0,width:500,height:82));var codeOffset:Int?;trigger.onCode={codeOffset=$0}
    for _ in 0..<3{trigger.insertText("`",replacementRange:NSRange(location:NSNotFound,length:0))}
    pump();assert(codeOffset==0 && trigger.string.isEmpty)
    let page="<!doctype html><html><head><style>"+String(repeating:".x{color:red;margin:0}",count:1000)+"</style></head><body>"+String(repeating:"<a href=\"https://example.com/long/path?q=1&amp;v=2\">test</a>\n",count:2000)+"</body></html>"
    let twoPages=page+"\n"+page
    let start=Date();input.insertText(twoPages,replacementRange:NSRange(location:0,length:0));pump()
    assert(store.selected!.draft==twoPages)
    store.sendMessage();pump(1)
    assert(store.selected!.messages.first?.text==twoPages && store.selected!.draft.isEmpty)
    input.insertText("Interfaccia ancora reattiva",replacementRange:NSRange(location:0,length:0));pump()
    assert(store.selected!.draft=="Interfaccia ancora reattiva")
    let elapsed=Date().timeIntervalSince(start)
    print("Native editor tests passed: triple backtick callback, insert/send",twoPages.utf8.count,"HTML bytes unchanged and subsequent typing in",elapsed,"seconds (including 1.6s run-loop waits)")
 }
}
