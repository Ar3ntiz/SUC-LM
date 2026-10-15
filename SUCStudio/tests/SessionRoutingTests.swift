import Foundation
@main struct SessionRoutingTests {
    @MainActor static func main() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = StudioStore(storageURL: root.appendingPathComponent("sessions.json"))
        store.modelState = "ready"; store.loadedModel = "test-model"
        let firstID = store.archive.selected!
        store.mutateSelected { $0.draft = "first" }; store.sendMessage()
        let request = store.requests.keys.first!
        store.newSession(); let secondID = store.archive.selected!
        store.mutateSelected { $0.draft = "independent draft" }
        store.receive(["event":"delta", "id":request,"payload":["text":"answer for first"]])
        precondition(store.selected!.id == secondID)
        precondition(store.selected!.draft == "independent draft")
        precondition(store.selected!.messages.isEmpty)
        let original = store.archive.sessions.first { $0.id == firstID }!
        precondition(original.messages.last!.text == "answer for first")
        let payload: [String: Any] = ["prompt_tokens":10,"completion_tokens":3,"prefill_s":2.0,"decode_s":1.0]
        let wire: [String: Any] = ["event":"metrics","id":request,"payload":payload]
        store.receive(try JSONSerialization.jsonObject(with: JSONSerialization.data(withJSONObject: wire)) as! [String: Any])
        store.receive(["event":"finished","id":request,"payload":[:]])
        precondition(store.requests.isEmpty)
        precondition(store.archive.sessions.first { $0.id == firstID }!.messages.last!.metrics!.decodeRate == 2)
        store.duplicate(original)
        precondition(store.selected!.id != firstID)
        precondition(store.selected!.messages.last!.id != original.messages.last!.id)
        precondition(store.selected!.messages.last!.state == "interrupted")
        store.receive(["event":"started", "id":"api-old", "payload":[:]])
        store.receive(["event":"started", "id":"api-new", "payload":[:]])
        store.receive(["event":"idle", "id":"api-old", "payload":[:]])
        precondition(store.latestIsCurrent && store.activePhase == "Prefill")
        precondition(store.latestMetrics == nil)
        store.receive(["event":"idle", "id":"api-new", "payload":[:]])
        precondition(!store.latestIsCurrent)
        store.receive(["event":"memory_plan", "id":"load", "payload":["mode":"expert_ssd", "limit_bytes":Double(20 * 1_073_741_824), "resident_bytes":Double(4 * 1_073_741_824), "reserve_bytes":Double(4 * 1_073_741_824), "expert_budget_bytes":Double(12 * 1_073_741_824)]])
        precondition(store.memoryPlan.contains("Streaming SSD expert") && store.memoryPlan.contains("20.00 GiB"))
        store.receive(["event":"started", "id":"api-stats", "payload":[:]])
        store.receive(["event":"expert_stats", "id":"api-stats", "payload":["phase":"prefill", "bytes_read":Double(1_073_741_824), "hit_rate":0.5, "wait_s":1.0]])
        precondition(store.expertStats.contains("Fine prefill") && store.expertStats.contains("50.0%"))
        let stats = store.expertStats
        store.receive(["event":"expert_stats", "id":"api-old", "payload":["phase":"completed"]])
        precondition(store.expertStats == stats)
        store.receive(["event":"idle", "id":"api-stats", "payload":[:]])
        print("Memory UI and SSD phase snapshot routing passed")
        try await Task.sleep(nanoseconds: 600_000_000)
        let disk = try ArchiveDisk.load(root.appendingPathComponent("sessions.json"))
        precondition(disk.sessions.count == 3)
        let localFile = root.appendingPathComponent("allegato.estensione-libera")
        try Data("Contenuto allegato da conservare".utf8).write(to: localFile)
        store.archive.selected = firstID
        store.attachFiles([localFile])
        precondition(store.selectedAttachmentsLoading)
        store.archive.selected = secondID
        for _ in 0..<500 { if !store.attachmentLoads.contains(firstID) { break }; try await Task.sleep(nanoseconds: 10_000_000) }
        precondition(!store.attachmentLoads.contains(firstID))
        precondition(store.selected!.draftAttachments == nil)
        let attached = store.archive.sessions.first { $0.id == firstID }!.draftAttachments!
        precondition(attached.count == 1 && attached[0].text.contains("da conservare"))
        store.archive.selected = firstID; store.mutateSelected { $0.draft = "" }
        precondition(store.canSendSelected)
        store.sendMessage()
        let user = store.selected!.messages[store.selected!.messages.count - 2]
        precondition(user.attachments == attached && user.modelContent.contains("da conservare"))
        precondition(store.selected!.draftAttachments == nil)
        let attachmentRequest = store.requests.keys.first!
        store.receive(["event":"finished", "id":attachmentRequest, "payload":[:]])
        store.mutateSelected { $0.draftAttachments = attached }
        store.removeAttachment(attached[0].id)
        precondition(store.selected!.draftAttachments!.isEmpty)
        try await Task.sleep(nanoseconds: 600_000_000)
        let savedAttachments = try ArchiveDisk.load(root.appendingPathComponent("sessions.json"))
        precondition(savedAttachments.sessions.first { $0.id == firstID }!.messages.contains { $0.attachments == attached })
        print("Attachment session routing: async import, switch, attachment-only send, removal and persistence passed")
        print("SessionRoutingTests: concurrent session selection, drafts, metrics routing and duplication passed")
    }
}
