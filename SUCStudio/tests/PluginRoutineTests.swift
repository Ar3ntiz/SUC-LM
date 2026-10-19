import Foundation
import CryptoKit
@main struct PluginRoutineTests {
 @MainActor static func main() async throws {
  let data=Data("{\"formatVersion\":1,\"identifier\":\"sample.test\",\"name\":\"Example\",\"version\":\"1\",\"skills\":[{\"name\":\"Clarity\",\"instructions\":\"Be clear\"}]}".utf8)
  let digest=SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined()
  let manifest=try PluginValidation.verified(data,sha256:digest)
  do{_ = try PluginValidation.verified(data,sha256:"bad");preconditionFailure("Bad digest accepted")}catch{}
  var invalid=manifest;invalid.identifier="../escape"
  do{_ = try PluginValidation.decode(JSONEncoder().encode(invalid));preconditionFailure("Bad identifier accepted")}catch{}
  let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:root)}
  let store=StudioStore(storageURL:root.appendingPathComponent("archive.json"));store.schedulingEnabled=false
  try store.installPlugin(manifest,source:"fixture");let plugin=store.archive.settings.plugins![0]
  precondition(!plugin.enabled && store.archive.settings.skills==nil)
  store.setPlugin(plugin,enabled:true,review:false);precondition(store.archive.settings.skills?.first?.enabled==true)
  store.setPlugin(plugin,enabled:false,review:false);precondition(store.archive.settings.skills?.isEmpty==true)
  store.removePlugin(plugin);precondition(store.archive.settings.plugins!.isEmpty)
  var routine=StudioRoutine();routine.name="Test";routine.prompt="Run scheduled prompt";routine.sessionID=store.selected!.id;routine.enabled=true;routine.intervalMinutes=60;routine.repeatEnabled=true;routine.repeatCount=60;routine.nextRun=Date(timeIntervalSince1970:1)
  try store.saveRoutine(routine);store.modelState="ready";store.loadedModel="test";store.mutateSelected{$0.draft="preserved"}
  let now=Date();store.schedulingEnabled=true;store.tickRoutines(now:now);store.schedulingEnabled=false
  precondition(store.requests.count==1 && store.selected!.draft=="preserved")
  precondition(abs(store.archive.settings.routines![0].nextRun.timeIntervalSince(now)-3600)<1)
  let request=store.requests.keys.first!;store.receive(["event":"delta","id":request,"payload":["text":"Done"]]);store.receive(["event":"finished","id":request,"payload":[:]])
  store.tickRoutines(now:now);precondition(store.archive.settings.routines![0].history.last?.state=="completed")
  store.schedulingEnabled=true;store.tickRoutines(now:now);precondition(store.requests.isEmpty)
  var archived=store.archive;archived.settings.routines?[0].status="In esecuzione";archived.recoverInterrupted();precondition(archived.settings.routines![0].status=="Interrotta al riavvio")
  print("PluginRoutineTests: manifest/checksum validation, inactive install, enable/remove, scheduling, draft preservation, no backlog/replay and recovery passed")
 }
}
