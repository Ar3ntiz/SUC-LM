import Foundation

@main struct UniversalPlatformTests {
    @MainActor static func main() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let archiveURL = root.appendingPathComponent("sessions.json")
        var archive = Archive()
        archive.settings.chatBackend = "local" // Moving an archive from ARM must not enable Intel inference.
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try JSONEncoder().encode(archive).write(to: archiveURL)
        let store = StudioStore(storageURL: archiveURL)
        precondition(StudioPlatform.name == "SUC Studio")
        precondition(StudioPlatform.dataDirectory == "SUC Studio Universal Preview")
        precondition(KeyStore.service != "local.sucstudio.apikey")
        precondition(StudioPlatform.version == "1.4.7.04")
        precondition(PrefillStatus(["processed":2048,"total":2165,"cache":2048])!.fraction==0)
        precondition(PrefillStatus(["processed":2165,"total":2165,"cache":2048])!.fraction==1)
        precondition(PrefillStatus(["processed":0,"total":0])==nil)
        precondition(PrefillStatus(["processed":20,"total":10])==nil)
        precondition(PrefillStatus(["processed":1,"total":10,"cache":2])==nil)
        var telemetry=RemoteTelemetry()
        telemetry.ingest(["suc_metrics":["prompt_tokens":2165,"cached_tokens":2048,"prefill_tokens":117,"completion_tokens":41,"prefill_s":1.0,"decode_s":2.0]])
        precondition(telemetry.payload()["decode_tps"] as? Double == 20)
        precondition(telemetry.payload()["prefill_tps"] as? Double == 117)
        precondition(telemetry.payload()["decode_source"] as? String == "server")
        print("Progress and real live SUC token-rate tests: PASS")
        #if arch(arm64) && !SUC_REMOTE_ONLY
        precondition(StudioPlatform.localInferenceAvailable)
        precondition(!store.usesRemote)
        store.archive.settings.chatBackend = "remote"
        precondition(store.usesRemote)
        print("UniversalPlatformTests: ARM retains local/remote choice; data and credentials isolated")
        #else
        precondition(!StudioPlatform.localInferenceAvailable)
        precondition(store.archive.settings.chatBackend == "remote")
        store.archive.settings.chatBackend = "local"
        precondition(store.usesRemote)
        store.loadModel()
        precondition(store.engine.process == nil && store.modelState != "loading")
        precondition(store.error == StudioPlatform.localUnavailableMessage && store.tab == "server")
        store.modelState = "ready"
        store.startServer()
        precondition(!store.serverBusy && !store.serverRunning && store.engine.process == nil)
        do { try store.engine.start(); preconditionFailure("Intel engine launch must be blocked") }
        catch { precondition(error.localizedDescription == StudioPlatform.localUnavailableMessage) }
        print("UniversalPlatformTests: Intel routes remote despite saved local selection; model/server/helper launch blocked")
        #endif
    }
}
