import Foundation
@main struct LoadMemoryTests {
 static func main(){
  precondition(LoadMemoryStatus(["resident_bytes":512.0,"target_bytes":1024.0,"initial_resident_bytes":256.0])!.fraction==0.5)
  precondition(LoadMemoryStatus(["resident_bytes":1024.0,"target_bytes":1024.0])!.fraction==1)
  for p:[String:Any] in [["resident_bytes":-1.0,"target_bytes":1024.0],["resident_bytes":2048.0,"target_bytes":1024.0],["resident_bytes":0.0,"target_bytes":0.0],["target_bytes":1024.0],["resident_bytes":Double.nan,"target_bytes":1024.0]] {precondition(LoadMemoryStatus(p)==nil)}
  precondition(LoadMemoryStatus(["resident_bytes":1.0,"target_bytes":1024.0,"initial_resident_bytes":Double.nan])!.initial==0)
  print("Load memory telemetry bounds, missing samples and resident fraction: PASS")
 }
}
