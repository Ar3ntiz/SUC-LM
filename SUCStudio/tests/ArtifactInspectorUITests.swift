import AppKit
import SwiftUI
@main struct ArtifactInspectorUITests {
    @MainActor static func main() async throws {
        _ = NSApplication.shared;NSApp.setActivationPolicy(.accessory)
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer{try? FileManager.default.removeItem(at:root)}
        let store=StudioStore(storageURL:root.appendingPathComponent("sessions.json"))
        store.mutateSelected{$0.artifacts=[ChatArtifact(name:"fixture.txt",byteCount:42)]}
        let originalSession=store.selected!.id
        store.artifactsVisible=true
        let host=NSHostingController(rootView:ChatView().environmentObject(store))
        let window=NSWindow(contentRect:NSRect(x:0,y:0,width:1180,height:720),styleMask:[.titled,.closable,.resizable],backing:.buffered,defer:false)
        window.contentViewController=host;window.setContentSize(NSSize(width:1180,height:720));window.orderBack(nil)
        defer{window.close()}
        func find(_ view:NSView)->NSButton? {
            if let button=view as? NSButton,button.identifier?.rawValue=="close-artifacts" {return button}
            for child in view.subviews {if let found=find(child){return found}}
            return nil
        }
        func closeButton()->NSButton? {window.contentView.flatMap{find($0)}}
        func wait(_ label:String,_ condition:()->Bool) async throws {
            for _ in 0..<100 {if condition(){return};try await Task.sleep(nanoseconds:50_000_000)}
            throw MCPConnection.failure("Artifact UI test timeout: "+label)
        }
        try await wait("visible sidebar"){closeButton() != nil}
        FileHandle.standardError.write(Data("Artifact UI: panel visible\n".utf8))
        closeButton()!.performClick(nil)
        try await wait("closed by x"){!store.artifactsVisible && closeButton()==nil}
        FileHandle.standardError.write(Data("Artifact UI: x removed panel\n".utf8))
        precondition(window.isVisible)
        precondition(store.archive.sessions.first{$0.id==originalSession}?.artifacts?.count==1)
        try Data("{\"passed\":true,\"close_button_pressed\":true,\"panel_removed\":true,\"main_window_preserved\":true,\"artifacts_preserved\":true}\n".utf8).write(to:URL(fileURLWithPath:CommandLine.arguments[1]),options:.atomic)
        FileHandle.standardError.write(Data("Artifact sidebar UI PASS: real x removes panel, keeps main window and artifacts.\n".utf8))
    }
}
