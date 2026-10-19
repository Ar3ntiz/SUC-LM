import SwiftUI
import AppKit

@MainActor final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    var store: StudioStore?
    var approvedQuit = false
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if approvedQuit { return .terminateNow }
        guard let store else { return .terminateNow }
        store.quit { [weak self] allowed in self?.approvedQuit = allowed; NSApplication.shared.reply(toApplicationShouldTerminate: allowed) }
        return .terminateLater
    }
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApplication.shared.setActivationPolicy(.regular)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            for window in NSApplication.shared.windows where window.title == StudioPlatform.name {
                window.setFrameAutosaveName("SUCStudioUniversal.main"); window.minSize = NSSize(width: 820, height: 560); window.delegate = self
                if let screen = NSScreen.main, !NSScreen.screens.contains(where: { $0.visibleFrame.intersects(window.frame) }) { window.setFrameOrigin(screen.visibleFrame.origin) }
            }
        }
    }
}
@main struct SUCStudioApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var store = StudioStore()
    @Environment(\.openWindow) private var openWindow
    var body: some Scene {
        Window(StudioPlatform.name, id: "main") {
            StudioRoot().environmentObject(store).onAppear { delegate.store = store }
        }.defaultSize(width: 1180, height: 780).windowResizability(.contentMinSize)
        .commands {
            CommandGroup(replacing: .newItem) { Button("Nuova sessione") { store.newSession(); show() }.keyboardShortcut("n"); Button("Importa sessione…") { store.importSession() } }
            CommandGroup(replacing: .appSettings) { Button("Settings…") { store.tab = "settings"; show() }.keyboardShortcut(",") }
            CommandMenu("Navigazione") {
                Button("Chat") { store.tab = "chat"; show() }.keyboardShortcut("1")
                Button("Server") { store.tab = "server"; show() }.keyboardShortcut("2")
                Button("Settings") { store.tab = "settings"; show() }.keyboardShortcut("3")
                Button("Cerca sessioni") { store.tab = "chat"; show(); NotificationCenter.default.post(name: .init("SUCStudioSearch"), object: nil) }.keyboardShortcut("f")
                Button("Interrompi risposta") { store.cancelSelected() }.keyboardShortcut(.escape, modifiers: [])
            }
        }
        MenuBarExtra(StudioPlatform.name, systemImage: store.serverRunning ? "server.rack" : "sparkles.rectangle.stack") { MenuBarView(show: show).environmentObject(store) }.menuBarExtraStyle(.window)
    }
    private func show() { openWindow(id: "main"); NSApplication.shared.activate(ignoringOtherApps: true); NSApplication.shared.windows.first { $0.title == StudioPlatform.name }?.deminiaturize(nil) }
}
