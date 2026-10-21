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
            s.archive.settings.remote?.sendExtensions=true;s.archive.settings.remote?.telemetryProfile="ds4"
            s.mutateSelected{$0.contextWindow=4096;$0.maxTokens=400};return s
        }
        var edited=ChatMessage(role:"assistant",text:"answer A")
        check(edited.chatParts.answer=="answer A");edited.text="answer B";check(edited.chatParts.answer=="answer B")
        check(ContextBudget.estimate(ChatMessage(reasoning:String(repeating:"thinking",count:1000),role:"assistant",text:"ok"))==ContextBudget.estimate(ChatMessage(role:"assistant",text:"ok")))
        check(TranscriptWindow.initial(500)==480)
        check(TranscriptWindow.updated(start:420,count:700,following:false)==420)
        check(TranscriptWindow.updated(start:420,count:700,following:true)==680)
        check(TranscriptWindow.earlier(420)==400)
        let original=String(repeating:"Città 🌍 中文 RAW_HISTORICAL_MARKER ",count:400)
        let pieces=ContextBudget.chunks(original,maxBytes:91);check(pieces.joined()==original && pieces.allSatisfy{$0.utf8.count<=91})
        check(ContextBudget.payload([["type":"image_url"],["type":"text","text":"abc"]])==1025)
        check(!ContextBudget.needsCompact(used:989,limit:1000));check(ContextBudget.needsCompact(used:990,limit:1000))
        let threshold=store();threshold.mutateSelected{$0.contextWindow=10000;$0.maxTokens=4000;$0.allowWeb=false}
        let thresholdSystem=threshold.contextSystemPrompt(threshold.selected!)
        let anchor=ChatMessage(role:"assistant",text:String(repeating:"x",count:33000),model:threshold.chatModel,
            metrics:Metrics(promptTokens:8699,completionTokens:1),generation:GenerationSnapshot(temperature:0.1,maxTokens:4000,context:0,thinking:false,systemPrompt:thresholdSystem))
        threshold.mutateSelected{$0.messages=[anchor]}
        let input:[[String:Any]]=[["role":"system","content":thresholdSystem],["role":"assistant","content":anchor.text],["role":"user","content":"next"]]
        check(threshold.contextUsage(threshold.selected!,includeDraft:false).used==8700)
        check(threshold.requestContextUsage(threshold.selected!,messages:input)==8710)
        check(!threshold.maybeCompactBeforeSend(threshold.selected!,messages:input,maxTokens:4000,queued:false,toolResult:nil,isTool:false))
        check(threshold.contextOutputLimit(threshold.selected!,history:input)==1190)
        check(threshold.compacting.isEmpty)
        threshold.mutateSelected{$0.messages[0].metrics=Metrics(promptTokens:9888,completionTokens:1)}
        check(threshold.requestContextUsage(threshold.selected!,messages:input)==9899)
        check(!ContextBudget.needsCompact(used:9899,limit:10000))
        threshold.mutateSelected{$0.messages[0].metrics=Metrics(promptTokens:9889,completionTokens:1)}
        check(threshold.requestContextUsage(threshold.selected!,messages:input)==9900)
        check(ContextBudget.needsCompact(used:9900,limit:10000))
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
        check(CompactionPolicy.outputBudget(limit:262144,inputTokens:100000)==16384)
        check(CompactionPolicy.outputBudget(limit:262144,inputTokens:20000)>CompactionPolicy.outputBudget(limit:262144,inputTokens:1000))
        check(CompactionPolicy.outputBudget(limit:4096,inputTokens:100000)<=4096/3)
        for model in ["summary-retry","summary-truncated","summary-only-thinking"] {
            let value=store(model),id=value.selected!.id
            value.mutateSelected{$0.contextWindow=32768;$0.messages=[ChatMessage(role:"user",text:"Preserva il file notes.txt e il risultato 42.")]}
            value.requestCompaction(id)
            try await wait("summary thinking/truncation retry"){!value.compacting.contains(id)}
            if model=="summary-only-thinking" {check(value.selected!.contextSummary==nil && value.selected!.messages.contains{$0.text.contains("esaurito i token")})}
            else {check(value.selected!.contextSummary?.contains("MEMORY:")==true)}
        }
        let thinking=store("thinking-only");thinking.mutateSelected{$0.draft="test"};thinking.sendMessage()
        try await wait("thinking-only is error"){thinking.requests.isEmpty}
        check(thinking.selected!.messages.contains{$0.role=="assistant" && $0.state=="error" && $0.finishReason=="length"})
        let background=store("thinking-only"),backgroundID=background.selected!.id,failedRoutineID=UUID()
        background.archive.settings.routines=[StudioRoutine(id:failedRoutineID,name:"Thinking regression",prompt:"test",sessionID:backgroundID,backend:"remote",nextRun:Date(),enabled:true)]
        background.schedulingEnabled=true;background.tickRoutines()
        try await wait("routine failure"){background.requests.isEmpty}
        background.tickRoutines()
        check(background.archive.settings.routines?.first?.status=="Interrotta o errore")
        let preflight=store(),preflightID=preflight.selected!.id,preflightRoutineID=UUID()
        preflight.mutateSelected{$0.messages=[ChatMessage(role:"user",text:original)];$0.maxTokens=512}
        preflight.archive.settings.routines=[StudioRoutine(id:preflightRoutineID,name:"Preflight compaction",prompt:"test",sessionID:preflightID,backend:"remote",nextRun:Date(),enabled:true)]
        preflight.schedulingEnabled=true;preflight.tickRoutines()
        try await wait("routine pending preflight"){preflight.compacting.contains(preflightID)}
        check(preflight.compacting.contains(preflightID) && preflight.activeRoutines[preflightRoutineID]==preflightID && preflight.archive.settings.routines?.first?.status=="In esecuzione")
        preflight.cancelSelected();try await wait("routine preflight cancelled"){!preflight.compacting.contains(preflightID)}
        preflight.tickRoutines();check(preflight.archive.settings.routines?.first?.status=="Interrotta o errore")
        FileHandle.standardError.write(Data("Context test: commands and explicit/implicit plugin\n".utf8))
        let s=store();let manifest=PluginManifest(formatVersion:1,identifier:"plug.test",name:"Test plugin",version:"1",skills:[PluginSkill(name:"PluginSkill",description:"Check plugin",instructions:"PLUGIN_BODY_SECRET")],description:"Test plugin catalog")
        try s.installPlugin(manifest,source:"fixture");s.setPlugin(s.archive.settings.plugins![0],enabled:true,review:false)
        for (prompt,answer) in [("/plugin plug.test controlla","EXPLICIT_PLUGIN_OK"),("auto-plugin","AUTO_PLUGIN_OK")] {
            s.mutateSelected{$0.draft=prompt};s.sendMessage()
            try await wait("plugin invocation"){s.selected!.messages.contains{$0.text==answer && $0.state=="complete"}}
        }
        let routineID=UUID(),routineSession=s.selected!.id
        s.mutateSelected{$0.maxTokens=512;$0.contextWindow=32768}
        s.archive.settings.routines=[StudioRoutine(id:routineID,name:"Tool regression",prompt:"routine-prompt",sessionID:routineSession,backend:"remote",nextRun:Date(),enabled:true)]
        s.schedulingEnabled=true;s.tickRoutines()
        try await wait("scheduled tool result"){s.selected!.messages.contains{$0.text=="ROUTINE_RESULT_OK" && $0.state=="complete"}}
        s.tickRoutines();check(s.archive.settings.routines?.first?.status=="Completata")
        s.schedulingEnabled=false
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
        let localText=try await summaryTask.value;check(localText.text=="LOCAL_SUMMARY" && local.summaryWaiters.isEmpty)
        let cancelledSummary=Task{try await local.summarizeContext(localID,body:["messages":[]],remote:nil)}
        try await wait("local cancel waiter"){!local.summaryWaiters.isEmpty}
        let cancelledID=local.summaryWaiters.keys.first!;cancelledSummary.cancel()
        do{_=try await cancelledSummary.value;check(false)}catch{}
        check(local.summaryWaiters.isEmpty && local.consumeCompactionEvent(["event":"finished","id":cancelledID]))
        print("Context PASS: lossless Unicode chunking, remote limits, manual/auto compaction, 99% trigger, queued continuation, actual usage, archive preservation, cancellation/error rollback, commands and explicit/model plugin invocation.")
    }
}
