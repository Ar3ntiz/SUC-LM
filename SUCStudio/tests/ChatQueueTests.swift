import AppKit
import Foundation

@main struct ChatQueueTests {
    static func check(_ value:Bool,line:UInt = #line) {
        if !value {FileHandle.standardError.write(Data("Queue check failed at line \(line)\n".utf8));fatalError("Queue check failed")}
    }
    @MainActor static func wait(_ label:String,_ condition:()->Bool) async throws {
        for _ in 0..<600 {if condition(){return};try await Task.sleep(nanoseconds:50_000_000)}
        preconditionFailure("Timed out: "+label)
    }
    @MainActor static func main() async throws {
        _=NSApplication.shared;NSApp.setActivationPolicy(.accessory)
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer{try? FileManager.default.removeItem(at:root)}
        let fixture=Process(),pipe=Pipe();fixture.executableURL=URL(fileURLWithPath:"/usr/bin/python3");fixture.arguments=[CommandLine.arguments[1]];fixture.standardOutput=pipe
        try fixture.run();defer{fixture.terminate()}
        let port=String(decoding:pipe.fileHandleForReading.availableData,as:UTF8.self).trimmingCharacters(in:.whitespacesAndNewlines)
        func makeStore(_ model:String)->StudioStore {
            let store=StudioStore(storageURL:root.appendingPathComponent(UUID().uuidString+".json"));store.schedulingEnabled=false
            store.archive.settings.chatBackend="remote";store.archive.settings.remote=RemoteConfiguration(endpoint:"http://127.0.0.1:\(port)/v1",model:model)
            return store
        }
        for model in ["queue","queue-tool"] {
            FileHandle.standardError.write(Data("Queue test: \(model)\n".utf8))
            let store=makeStore(model),id=store.selected!.id
            store.archive.settings.skills=[StudioSkill(name:"QueueSkill",description:"Queue fixture",instructions:"QUEUE_SKILL_BODY",enabled:true)]
            store.mutateSelected{$0.draft="queue-first"};store.sendMessage()
            try await wait("stream started"){store.selected!.messages.last?.text.isEmpty==false && store.sessionTurnBusy(id)}
            store.mutateSelected{$0.draft="queue-second";$0.draftAttachments=[ChatAttachment(name:"queued.txt",text:"ATTACHED_QUEUE",originalBytes:14)]}
            check(store.canSendSelected);store.sendMessage()
            store.mutateSelected{$0.draft="queue-third"};store.sendMessage()
            check(store.selected!.queuedMessages?.count==2 && store.selected!.draft.isEmpty)
            check(store.requests.values.filter{$0.session==id}.count<=1)
            check(!store.selected!.messages.contains{$0.text=="queue-second"})
            store.mutateSelected{$0.draft="BOZZA_NON_INVIATA";$0.draftAttachments=[ChatAttachment(name:"draft.txt",text:"UNSENT",originalBytes:6)]}
            store.newSession();let other=store.selected!.id
            store.mutateSelected{$0.draft="ALTRA_SESSIONE"}
            try await wait("FIFO final answer") {store.archive.sessions.first{$0.id==id}!.messages.contains{$0.text=="ACK:queue-third" && $0.state=="complete"}}
            let session=store.archive.sessions.first{$0.id==id}!
            let users=session.messages.filter{$0.role=="user" && !$0.isOperational}
            check(users.map(\.text)==["queue-first","queue-second","queue-third"])
            check(users[1].attachments?.first?.name=="queued.txt" && users[2].attachments==nil)
            check(session.draft=="BOZZA_NON_INVIATA" && session.draftAttachments?.first?.name=="draft.txt")
            check(session.queuedMessages?.isEmpty==true && session.hasUnreadMessages==true)
            check(store.archive.selected==other && store.selected!.draft=="ALTRA_SESSIONE")
        }
        FileHandle.standardError.write(Data("Queue test: error and resume\n".utf8))
        let paused=makeStore("queue-error"),id=paused.selected!.id
        paused.mutateSelected{$0.draft="queue-first"};paused.sendMessage()
        paused.mutateSelected{$0.draft="queue-second"};paused.sendMessage()
        try await wait("error pauses queue"){paused.selected!.queuePaused==true}
        check(paused.selected!.queuedMessages?.count==1)
        paused.resumeChatQueue(id)
        try await wait("resume after error"){paused.selected!.messages.contains{$0.text=="ACK:queue-second" && $0.state=="complete"}}

        FileHandle.standardError.write(Data("Queue test: cancellation and resume\n".utf8))
        let stopped=makeStore("queue"),stopID=stopped.selected!.id
        stopped.mutateSelected{$0.draft="queue-first"};stopped.sendMessage()
        stopped.mutateSelected{$0.draft="queue-second"};stopped.sendMessage()
        stopped.cancelSelected()
        try await wait("cancel settles"){!stopped.sessionTurnBusy(stopID)}
        check(stopped.selected!.queuePaused==true && stopped.selected!.queuedMessages?.count==1)
        stopped.resumeChatQueue(stopID)
        try await wait("resume after cancel"){stopped.selected!.messages.contains{$0.text=="ACK:queue-second" && $0.state=="complete"}}

        var saved=ChatSession();saved.queuedMessages=[ChatMessage(role:"user",text:"RESTORED")]
        var archive=Archive();archive.sessions=[saved];archive.selected=saved.id
        let file=root.appendingPathComponent("restore.json");try JSONEncoder().encode(archive).write(to:file)
        let restored=StudioStore(storageURL:file);restored.schedulingEnabled=false
        check(restored.selected!.queuedMessages?.first?.text=="RESTORED" && restored.selected!.queuePaused==true)
        restored.mutateSelected{$0.draft="PRESERVE"};restored.editQueuedMessage(saved.queuedMessages![0].id,sessionID:saved.id)
        check(restored.selected!.queuedMessages?.count==1 && restored.selected!.draft=="PRESERVE")
        restored.mutateSelected{$0.draft=""};restored.editQueuedMessage(saved.queuedMessages![0].id,sessionID:saved.id)
        check(restored.selected!.queuedMessages?.isEmpty==true && restored.selected!.draft=="RESTORED")
        print("Chat queue PASS: streaming FIFO, complete tool chain, attachments/drafts, background session, error/cancel pause and resume, persisted queue paused on relaunch, safe edit.")
    }
}
