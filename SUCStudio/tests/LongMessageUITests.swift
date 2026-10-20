import AppKit
import SwiftUI
@MainActor final class LongMessageModel:ObservableObject {
    @Published var text=String(repeating:"riga di testo ",count:570)
    @Published var height:CGFloat=0
}
struct LongMessageHarness:View {
    @ObservedObject var model:LongMessageModel
    var body:some View {
        BoundedMessageBody(text:model.text).frame(width:700).fixedSize(horizontal:false,vertical:true)
            .onGeometryChange(for:CGFloat.self){$0.size.height} action:{model.height=$0}
    }
}
@main struct LongMessageUITests {
    @MainActor static func main() async throws {
        _=NSApplication.shared;NSApp.setActivationPolicy(.accessory)
        let report=CommandLine.arguments[1],root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".json")
        let store=StudioStore(storageURL:root);store.schedulingEnabled=false
        let model=LongMessageModel(),view=NSHostingView(rootView:LongMessageHarness(model:model).environmentObject(store))
        let window=NSWindow(contentRect:NSRect(x:0,y:0,width:740,height:500),styleMask:[.titled,.closable,.resizable],backing:.buffered,defer:false)
        window.contentView=view;window.orderBack(nil)
        for _ in 0..<100 {if model.height>600 {break};try await Task.sleep(nanoseconds:30_000_000)}
        let before=model.height
        model.text+=String(repeating:"riga di testo ",count:8)
        for _ in 0..<100 {if model.height>=before {try await Task.sleep(nanoseconds:200_000_000);break};try await Task.sleep(nanoseconds:30_000_000)}
        let after=model.height
        guard before>600,after>=before*0.9,model.text.count>8000 else{FileHandle.standardError.write(Data("Long message height regression: \(before) -> \(after)\n".utf8));exit(1)}
        let result:[String:Any]=["passed":true,"height_before_8000":Double(before),"height_after_8000":Double(after),"characters":model.text.count]
        try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:report),options:.atomic)
        FileHandle.standardError.write(Data("Long-message UI PASS: 8000-character threshold retains message height (\(before) -> \(after)).\n".utf8))
        window.close()
    }
}
