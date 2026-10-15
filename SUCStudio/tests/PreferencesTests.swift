import Foundation
@main struct PreferencesTests {
 @MainActor static func main() throws {
  var settings=AppSettings();settings.uiFont="Helvetica Neue";settings.chatFont="Georgia";settings.uiFontSize=99;settings.chatFontSize=1;settings.keepAwake=true
  StudioTypography.apply(settings);precondition(StudioTypography.uiSize==18 && StudioTypography.chatSize==12)
  precondition(StudioTypography.uiName=="Helvetica Neue" && StudioTypography.chatName=="Georgia")
  let decoded=try JSONDecoder().decode(AppSettings.self,from:JSONEncoder().encode(settings));precondition(decoded.keepAwake==true && decoded.chatFont=="Georgia")
  StudioTypography.apply(AppSettings());precondition(StudioTypography.uiSize==13 && StudioTypography.chatSize==14.5)
  let power=PowerController();try power.set(false);precondition(!power.active)
  print("PreferencesTests: independent font settings, size bounds, persistence/defaults and inactive power controller passed; no login or power assertion enabled")
 }
}
