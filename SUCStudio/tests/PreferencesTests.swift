import Foundation
@main struct PreferencesTests {
 @MainActor static func main() async throws {
  var settings=AppSettings();settings.uiFont="Helvetica Neue";settings.chatFont="Georgia";settings.uiFontSize=99;settings.chatFontSize=1;settings.keepAwake=true
  StudioTypography.apply(settings);precondition(StudioTypography.uiSize==18 && StudioTypography.chatSize==12)
  precondition(StudioTypography.uiName=="Helvetica Neue" && StudioTypography.chatName=="Georgia")
  let decoded=try JSONDecoder().decode(AppSettings.self,from:JSONEncoder().encode(settings));precondition(decoded.keepAwake==true && decoded.chatFont=="Georgia")
  StudioTypography.apply(AppSettings());precondition(StudioTypography.uiSize==13 && StudioTypography.chatSize==14.5)
  let power=PowerController();try power.set(false);precondition(!power.active)
  let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
  defer{try? FileManager.default.removeItem(at:root)}
  let url=root.appendingPathComponent("sessions.json")
  let first=StudioStore(storageURL:url)
  precondition(first.archive.settings.effectiveLanguage=="en")
  first.setLanguage("it")
  try await first.disk.save(first.archive)
  let reloaded=StudioStore(storageURL:url)
  precondition(reloaded.archive.settings.effectiveLanguage=="it" && StudioL10n.language=="it")
  precondition(L("Consenti comandi shell")=="Consenti comandi shell")
  reloaded.setLanguage("en")
  try await reloaded.disk.save(reloaded.archive)
  let savedEnglish=try ArchiveDisk.load(url)
  precondition(savedEnglish.settings.effectiveLanguage=="en")
  print("Language archive: real StudioStore setter, ArchiveDisk save/reload and UI language restoration PASS")
  print("PreferencesTests: independent font settings, size bounds, persistence/defaults and inactive power controller passed; no login or power assertion enabled")
 }
}
