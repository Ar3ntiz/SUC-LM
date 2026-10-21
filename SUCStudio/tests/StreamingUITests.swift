import AppKit
import SwiftUI
@main struct StreamingUITests {
    @MainActor static func main() async throws {
        _=NSApplication.shared;NSApp.setActivationPolicy(.accessory)
        let temporary=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:temporary,withIntermediateDirectories:true)
        defer{try? FileManager.default.removeItem(at:temporary)}
        let store=StudioStore(storageURL:temporary.appendingPathComponent("sessions.json"));store.schedulingEnabled=false
        store.archive.settings.chatBackend="remote";store.archive.settings.remote=RemoteConfiguration(endpoint:"http://127.0.0.1:1/v1",model:"ui-fixture")
        let history=(0..<100).map{i in ChatMessage(reasoning:String(repeating:"Historical reasoning. ",count:500),role:"assistant",text:"Message \(i)\n"+String(repeating:"A paragraph with **formatted text** and a [link](https://example.com).\n",count:80))}
        var sink=0
        let uncachedStart=Date()
        for _ in 0..<10 {for message in history{sink+=message.uncachedChatParts.answer.utf8.count}}
        let uncached=Date().timeIntervalSince(uncachedStart)
        for message in history{_=message.chatParts}
        let cachedStart=Date()
        for _ in 0..<10 {for message in history{sink-=message.chatParts.answer.utf8.count}}
        let cached=Date().timeIntervalSince(cachedStart);precondition(sink==0)
        let response=ChatMessage(role:"assistant",text:"",state:"generating",model:"ui-fixture")
        store.mutateSelected{$0.messages=history+[response];$0.contextWindow=262144}
        let session=store.selected!.id
        let view=NSHostingView(rootView:ChatView().environmentObject(store))
        let window=NSWindow(contentRect:NSRect(x:0,y:0,width:1200,height:800),styleMask:[.titled,.closable,.resizable],backing:.buffered,defer:false)
        window.contentView=view;window.orderBack(nil)
        try await Task.sleep(nanoseconds:1_000_000_000)
        let start=Date();var maxDelay:Double=0
        for _ in 0..<150 {
            let step=Date()
            store.mutateMessage(session,response.id){$0.text+="visible stream "};store.scheduleSave()
            try await Task.sleep(nanoseconds:10_000_000)
            maxDelay=max(maxDelay,Date().timeIntervalSince(step))
        }
        let duration=Date().timeIntervalSince(start)
        store.mutateMessage(session,response.id){$0.state="complete";$0.finishReason="stop"}
        precondition(store.selected!.messages.last(where:{$0.id==response.id})!.text==String(repeating:"visible stream ",count:150))
        let result:[String:Any]=["passed":maxDelay<2.5,"historical_messages":history.count,"history_bytes":history.reduce(0){$0+$1.text.utf8.count+($1.reasoning?.utf8.count ?? 0)},"stream_deltas":150,"stream_seconds":duration,"max_ui_tick_seconds":maxDelay,"uncached_presentation_seconds":uncached,"cached_presentation_seconds":cached]
        try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:CommandLine.arguments[1]),options:.atomic)
        print("Streaming UI PASS: \(duration)s, max tick \(maxDelay)s; presentation \(uncached)s -> \(cached)s")
        window.close()
        precondition(maxDelay<2.5,"UI streaming heartbeat stalled; see result.json")
    }
}
