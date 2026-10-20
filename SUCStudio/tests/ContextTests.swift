import AppKit
import Foundation
@main struct ContextTests {
    static func check(_ value:Bool,line:UInt = #line) {if !value{FileHandle.standardError.write(Data("Context check failed at line \(line)\n".utf8));fatalError("Context check failed")}}
    @MainActor static func wait(_ label:String,_ condition:()->Bool) async throws {
        for _ in 0..<1000 {if condition(){return};try await Task.sleep(nanoseconds:30_000_000)}
        FileHandle.standardError.write(Data("Context timed out: \(label)\n".utf8));fatalError(label)
    }
    @MainActor static func main() async throws {
        _=NSApplication.shared;NSApp.setActivationPolicy(.accessory)
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer{try? FileManager.default.removeItem(at:root)}
        let fixture=Process(),pipe=Pipe();fixture.executableURL=URL(fileURLWithPath:"/usr/bin/python3");fixture.arguments=[CommandLine.arguments[1]];fixture.standardOutput=pipe
        try fixture.run();defer{fixture.terminate()}
        let port=String(decoding:pipe.fileHandleForReading.availableData,as:UTF8.self).trimmingCharacters(in:.whitespacesAndNewlines)
        func store(_ model:String="context-fixture")->StudioStore {
            let s=StudioStore(storageURL:root.appendingPathComponent(UUID().uuidString+".json"));s.schedulingEnabled=false;s.archive.settings.chatBackend="remote";s.archive.settings.remote=RemoteConfiguration(endpoint:"http://127.0.0.1:\(port)/v1",model:model)
            s.mutateSelected{$0.contextWindow=4096;$0.maxTokens=400};return s
        }
        check(TranscriptWindow.initial(500)==420)
        check(TranscriptWindow.updated(start:420,count:700,following:false)==420)
        check(TranscriptWindow.updated(start:420,count:700,following:true)==620)
        check(TranscriptWindow.earlier(420)==380)
        let original=String(repeating:"Città 🌍 中文 RAW_HISTORICAL_MARKER ",count:400)
        let pieces=ContextBudget.chunks(original,maxBytes:91);check(pieces.joined()==original && pieces.allSatisfy{$0.utf8.count<=91})
        check(ContextBudget.payload([["type":"image_url"],["type":"text","text":"abc"]])==1025)
        check(!ContextBudget.needsCompact(used:989,reserve:0,limit:1000));check(ContextBudget.needsCompact(used:990,reserve:0,limit:1000))
        let detect=store(),client=RemoteInference(detect.archive.settings.remote!,key:"");_=try await client.models();check(client.modelContextWindows["context-fixture"]==4096);client.stop()
        for automatic in [false,true] {
            FileHandle.standardError.write(Data("Context test: \(automatic ? "automatic":"manual") compaction\n".utf8))
            let s=store(),id=s.selected!.id
            s.mutateSelected{$0.messages=[ChatMessage(role:"user",text:original),ChatMessage(role:"assistant",text:"queued.swift: preserva i vincoli.")]}
            let before=s.selected!.messages
            if automatic{s.mutateSelected{$0.draft="after-compact"};s.sendMessage();check(!s.hookSessions.contains(id))} else{s.mutateSelected{$0.draft="/compact preserva i file"};s.sendMessage();s.mutateSelected{$0.draft="after-compact"};s.sendMessage()}
            try await wait("compacted continuation"){s.selected!.messages.contains{$0.text=="AFTER_COMPACT_OK" && $0.state=="complete"}}
            check(s.selected!.messages.prefix(before.count).map(\.text)==before.map(\.text))
            check(s.selected!.activeContextSummary.contains("MEMORY:") && s.selected!.activeContextMessages.count==2)
            check(s.selected!.queuedMessages?.isEmpty==true && !s.compacting.contains(id))
            check(s.contextUsage(s.selected!,includeDraft:false).used==108 && !s.contextUsage(s.selected!,includeDraft:false).estimated)
            let restored=try JSONDecoder().decode(ChatSession.self,from:JSONEncoder().encode(s.selected!));check(restored.activeContextSummary==s.selected!.activeContextSummary)
        }
        FileHandle.standardError.write(Data("Context test: automatic threshold at end of generation\n".utf8))
        let full=store("fill-context");full.mutateSelected{$0.draft="fill"};full.sendMessage()
        try await wait("99 percent post-generation compact"){!(full.selected!.contextSummary ?? "").isEmpty && full.compacting.isEmpty}
        for model in ["summary-error","summary-slow"] {
            let s=store(model),id=s.selected!.id
            s.mutateSelected{$0.messages=[ChatMessage(role:"user",text:original)]}
            s.requestCompaction(id);s.mutateSelected{$0.draft="pending"};s.sendMessage()
            if model=="summary-slow" {try await Task.sleep(nanoseconds:60_000_000);s.cancelSelected()}
            try await wait("failed/cancelled compaction"){!s.compacting.contains(id)}
            check(s.selected!.contextSummary==nil && s.selected!.messages.first?.text==original && s.selected!.queuePaused==true && s.selected!.queuedMessages?.count==1)
        }
        FileHandle.standardError.write(Data("Context test: commands and explicit/implicit plugin\n".utf8))
        let s=store();let manifest=PluginManifest(formatVersion:1,identifier:"plug.test",name:"Test plugin",version:"1",skills:[PluginSkill(name:"PluginSkill",description:"Check plugin",instructions:"PLUGIN_BODY_SECRET")],description:"Test plugin catalog")
        try s.installPlugin(manifest,source:"fixture");s.setPlugin(s.archive.settings.plugins![0],enabled:true,review:false)
        for (prompt,answer) in [("/plugin plug.test controlla","EXPLICIT_PLUGIN_OK"),("auto-plugin","AUTO_PLUGIN_OK")] {
            s.mutateSelected{$0.draft=prompt};s.sendMessage()
            try await wait("plugin invocation"){s.selected!.messages.contains{$0.text==answer && $0.state=="complete"}}
        }
        let count=s.selected!.messages.count
        for command in ["/help","/status","/context 8192","/auto off","/plugins","/skills","/queue"] {s.mutateSelected{$0.draft=command};s.sendMessage()}
        check(s.selected!.contextWindow==8192 && s.selected!.autoCompact==false && s.requests.isEmpty && s.selected!.messages.count==count+7)
        check(!s.selected!.activeContextMessages.contains{$0.text==StudioStore.commandHelp})
        s.mutateSelected{$0.draft="/clear"};s.sendMessage();check(s.selected!.activeContextMessages.isEmpty && !s.selected!.messages.isEmpty)
        let local=store(),localID=local.selected!.id
        let summaryTask=Task{try await local.summarizeContext(localID,body:["messages":[]],remote:nil)}
        try await wait("local summary waiter"){!local.summaryWaiters.isEmpty}
        let requestID=local.summaryWaiters.keys.first!
        check(local.consumeCompactionEvent(["event":"delta","id":requestID,"payload":["text":"LOCAL_SUMMARY"]]))
        check(local.consumeCompactionEvent(["event":"finished","id":requestID,"payload":[:]]))
        let localText=try await summaryTask.value;check(localText=="LOCAL_SUMMARY" && local.summaryWaiters.isEmpty)
        let cancelledSummary=Task{try await local.summarizeContext(localID,body:["messages":[]],remote:nil)}
        try await wait("local cancel waiter"){!local.summaryWaiters.isEmpty}
        let cancelledID=local.summaryWaiters.keys.first!;cancelledSummary.cancel()
        do{_=try await cancelledSummary.value;check(false)}catch{}
        check(local.summaryWaiters.isEmpty && local.consumeCompactionEvent(["event":"finished","id":cancelledID]))
        print("Context PASS: lossless Unicode chunking, remote limits, manual/auto compaction, 99% trigger, queued continuation, actual usage, archive preservation, cancellation/error rollback, commands and explicit/model plugin invocation.")
    }
}
