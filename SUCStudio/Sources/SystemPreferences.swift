import SwiftUI
import AppKit
import ServiceManagement
import IOKit.pwr_mgt

@MainActor enum StudioTypography {
    static var uiName = "System", chatName = "System"
    static var uiSize = 13.0, chatSize = 14.5
    static func apply(_ settings: AppSettings) {
        uiName=settings.uiFont ?? "System";chatName=settings.chatFont ?? "System"
        uiSize=min(18,max(10,settings.uiFontSize ?? 13));chatSize=min(30,max(12,settings.chatFontSize ?? 14.5))
    }
    static func ui(size: Double = 13, weight: Font.Weight = .regular, design: Font.Design = .default) -> Font {
        let value=size*uiSize/13
        return (uiName == "System" || design == .monospaced ? Font.system(size:value,weight:weight,design:design):Font.custom(uiName,size:value).weight(weight))
    }
    static func ui(_ style: Font.TextStyle, design: Font.Design = .default) -> Font {
        let size:Double
        switch style {case .caption2:size=10;case .caption,.footnote:size=11;case .callout:size=12;case .title3:size=15;case .title2:size=17;case .title:size=22;case .largeTitle:size=28;default:size=13}
        return ui(size:size,weight:style == .headline ? .semibold:.regular,design:design)
    }
    static func chat(scale: Double = 1, weight: Font.Weight = .regular, pointOffset:Double=0) -> Font {
        chatName == "System" ? Font.system(size:max(8,chatSize*scale+pointOffset),weight:weight):Font.custom(chatName,size:max(8,chatSize*scale+pointOffset)).weight(weight)
    }
    static func code(pointOffset:Double=0) -> Font { .system(size:max(8,chatSize*0.9+pointOffset),design:.monospaced) }
    static var nsChat:NSFont { chatName == "System" ? .systemFont(ofSize:chatSize):(NSFontManager.shared.font(withFamily:chatName,traits:[],weight:5,size:chatSize) ?? .systemFont(ofSize:chatSize)) }
}
@MainActor final class PowerController {
    private var assertion:IOPMAssertionID = 0
    var active:Bool{assertion != 0}
    func set(_ enabled:Bool) throws {
        if !enabled {if assertion != 0{IOPMAssertionRelease(assertion);assertion=0};return}
        guard assertion == 0 else{return}
        let status=IOPMAssertionCreateWithName(kIOPMAssertionTypeNoIdleSleep as CFString,IOPMAssertionLevel(kIOPMAssertionLevelOn),"SUC Studio: mantieni il Mac attivo" as CFString,&assertion)
        guard status==kIOReturnSuccess else{assertion=0;throw MCPConnection.failure("Impossibile impedire lo standby (\(status)).")}
    }
    deinit {if assertion != 0{IOPMAssertionRelease(assertion)}}
}
extension StudioStore {
    func setAwake(_ enabled:Bool) {
        do {try power.set(enabled);archive.settings.keepAwake=power.active;powerStatus=power.active ? "Standby automatico del sistema impedito":"Standby gestito da macOS";scheduleSave()}
        catch{self.error=error.localizedDescription;archive.settings.keepAwake=false}
    }
    func refreshLoginStatus() {
        let state=SMAppService.mainApp.status
        launchAtLogin=state == .enabled || state == .requiresApproval
        switch state {case .enabled:loginStatus="Avvio al login attivo";case .requiresApproval:loginStatus="Richiede approvazione in Impostazioni di Sistema";case .notRegistered:loginStatus="Avvio al login disattivato";default:loginStatus="Servizio non disponibile: installa l’app in Applicazioni"}
    }
    func setLogin(_ enabled:Bool) {
        do {if enabled{try SMAppService.mainApp.register()}else{try SMAppService.mainApp.unregister()};refreshLoginStatus()}
        catch{self.error=error.localizedDescription;refreshLoginStatus()}
    }
    func applyTypography() {StudioTypography.apply(archive.settings);scheduleSave()}
}
