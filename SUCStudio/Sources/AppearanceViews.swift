import SwiftUI
import AppKit
import ServiceManagement
struct LifecycleSettingsView:View{
    @EnvironmentObject var store:StudioStore
    var body:some View{Section(L("Avvio ed energia")){
        Toggle(L("Esegui all’avvio del Mac (al login)"),isOn:Binding(get:{store.launchAtLogin},set:{store.setLogin($0)}))
        HStack{Text(LT(store.loginStatus)).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary);Spacer();Button(L("Impostazioni macOS…")){SMAppService.openSystemSettingsLoginItems()}}
        Toggle(L("Mantieni il computer attivo ed evita lo standby"),isOn:Binding(get:{store.archive.settings.keepAwake ?? false},set:{store.setAwake($0)}))
        Text(LT(store.powerStatus)+L(". Il display può spegnersi; stop manuale e chiusura del portatile seguono macOS.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
    }.onAppear{store.refreshLoginStatus()}}
}
struct TypographySettingsView:View{
    @EnvironmentObject var store:StudioStore
    private var families:[String]{["System"]+NSFontManager.shared.availableFontFamilies.sorted()}
    var body:some View{
        Picker(L("Font interfaccia"),selection:Binding(get:{store.archive.settings.uiFont ?? "System"},set:{store.archive.settings.uiFont=$0;store.applyTypography()})){ForEach(families,id:\.self){Text($0=="System" ? L("Sistema"):$0).tag($0)}}
        HStack{Text(L("Dimensione interfaccia"));Slider(value:Binding(get:{store.archive.settings.uiFontSize ?? 13},set:{store.archive.settings.uiFontSize=$0;store.applyTypography()}),in:10...18,step:0.5);Text(String(format:"%.1f pt",store.archive.settings.uiFontSize ?? 13)).monospacedDigit()}
        Picker(L("Font chat"),selection:Binding(get:{store.archive.settings.chatFont ?? "System"},set:{store.archive.settings.chatFont=$0;store.applyTypography()})){ForEach(families,id:\.self){Text($0=="System" ? L("Sistema"):$0).tag($0)}}
        HStack{Text(L("Dimensione chat"));Slider(value:Binding(get:{store.archive.settings.chatFontSize ?? 14.5},set:{store.archive.settings.chatFontSize=$0;store.applyTypography()}),in:12...30,step:0.5);Text(String(format:"%.1f pt",store.archive.settings.chatFontSize ?? 14.5)).monospacedDigit()}
        Text(L("Anteprima chat: testo leggibile, alla tua misura.")).font(StudioTypography.chat())
        Button(L("Ripristina font di sistema")){store.archive.settings.uiFont=nil;store.archive.settings.uiFontSize=nil;store.archive.settings.chatFont=nil;store.archive.settings.chatFontSize=nil;store.applyTypography()}
    }
}
