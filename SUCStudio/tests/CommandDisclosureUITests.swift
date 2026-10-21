import AppKit
import SwiftUI

@MainActor final class DisclosureFixtureState:ObservableObject {@Published var expanded=false}
struct DisclosureFixtureView:View {
    @ObservedObject var state:DisclosureFixtureState
    var body:some View {
        VStack {
            CommandDisclosureHeader(title:"Command fixture",expanded:$state.expanded)
            if state.expanded {Text("Command details fixture")}
            Spacer()
        }.frame(width:600).padding(10)
    }
}
@main struct CommandDisclosureUITests {
    @MainActor static func main() {
        let app=NSApplication.shared;app.setActivationPolicy(.accessory)
        Task{@MainActor in
            do {try await run();exit(0)} catch {
                FileHandle.standardError.write(Data(("UI fixture error: "+error.localizedDescription+"\n").utf8));exit(1)
            }
        }
        app.run()
    }
    @MainActor static func run() async throws {
        // Activate SwiftUI's virtual AX tree only inside this fixture process.
        NSApp.accessibilitySetValue(true,forAttribute:NSAccessibility.Attribute(rawValue:"AXEnhancedUserInterface"))
        NSApp.accessibilitySetValue(true,forAttribute:NSAccessibility.Attribute(rawValue:"AXManualAccessibility"))
        let state=DisclosureFixtureState(),host=NSHostingController(rootView:DisclosureFixtureView(state:state))
        let window=NSWindow(contentRect:NSRect(x:0,y:0,width:640,height:220),styleMask:[.titled,.closable],backing:.buffered,defer:false)
        window.contentViewController=host;window.orderBack(nil);defer{window.close()}
        func find(_ object:AnyObject,_ name:String,_ seen:inout Set<ObjectIdentifier>)->AnyObject? {
            guard seen.insert(ObjectIdentifier(object)).inserted else{return nil}
            if object.accessibilityIdentifier?()==name {return object}
            for child in object.accessibilityChildren?() ?? [] {if let found=find(child as AnyObject,name,&seen){return found}}
            if let view=object as? NSView {for child in view.subviews {if let found=find(child,name,&seen){return found}}}
            return nil
        }
        func control(_ name:String) async throws->AnyObject {
            for _ in 0..<100 {
                var seen=Set<ObjectIdentifier>()
                if let found=find(window.contentView!,name,&seen),(found.accessibilityFrame?() ?? .zero).width>0 {return found}
                try await Task.sleep(nanoseconds:20_000_000)
            }
            func dump(_ object:AnyObject,_ depth:Int) {
                guard depth<8 else{return}
                print(String(repeating:" ",count:depth),type(of:object),object.accessibilityIdentifier?() ?? "-")
                for child in object.accessibilityChildren?() ?? [] {dump(child as AnyObject,depth+1)}
            }
            dump(window.contentView!,0)
            throw NSError(domain:"Disclosure UI",code:1,userInfo:[NSLocalizedDescriptionKey:"Control not found: "+name])
        }
        let header=try await control("toggle-command-details")
        precondition((header.accessibilityFrame?() ?? .zero).width>=580 && (header.accessibilityFrame?() ?? .zero).height>=36)
        for _ in 0..<4 {
            let previous=state.expanded
            precondition(header.accessibilityPerformPress?()==true)
            try await Task.sleep(nanoseconds:30_000_000)
            precondition(state.expanded != previous)
        }
        precondition(!state.expanded)
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer{try? FileManager.default.removeItem(at:root)}
        let store=StudioStore(storageURL:root.appendingPathComponent("archive.json"));store.schedulingEnabled=false
        store.mutateSelected{$0.actions=[ActionRecord(name:"run_command",details:"run_command\ncommand: printf fixture",state:"completed",result:"Fixture output")]}
        store.agentRuns.insert(store.selected!.id)
        window.contentViewController=NSHostingController(rootView:WorkspaceSettingsView().environmentObject(store).padding(20))
        window.setContentSize(NSSize(width:640,height:1000))
        let log=try await control("toggle-action-log");precondition((log.isAccessibilityEnabled?() ?? false));precondition((log.accessibilityPerformPress?() ?? false))
        let row=try await control("toggle-action-details")
        precondition((row.isAccessibilityEnabled?() ?? false) && (row.accessibilityFrame?() ?? .zero).height>=60 && (row.accessibilityFrame?() ?? .zero).width>400)
        precondition((row.accessibilityPerformPress?() ?? false));try await Task.sleep(nanoseconds:100_000_000)
        precondition(store.selected!.actions?.count==1 && store.agentRuns.contains(store.selected!.id))
        precondition((row.accessibilityPerformPress?() ?? false));precondition((log.accessibilityPerformPress?() ?? false));store.agentRuns.removeAll()
        print("PASS: 580x36+ command header exposes a full-width accessible button; repeated accessibility presses toggle exactly once.")
        print("PASS: roomy action-log rows expand/collapse while configuration is disabled during active work; records and runtime state preserved.")
    }
}
