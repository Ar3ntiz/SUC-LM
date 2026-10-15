import SwiftUI
import AppKit
import ServiceManagement
struct LifecycleSettingsView:View{
    @EnvironmentObject var store:StudioStore
    var body:some View{Section("Avvio ed energia"){
        Toggle("Esegui all’avvio del Mac (al login)",isOn:Binding(get:{store.launchAtLogin},set:{store.setLogin($0)}))
        HStack{Text(store.loginStatus).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary);Spacer();Button("Impostazioni macOS…"){SMAppService.openSystemSettingsLoginItems()}}
        Toggle("Mantieni il computer attivo ed evita lo standby",isOn:Binding(get:{store.archive.settings.keepAwake ?? false},set:{store.setAwake($0)}))
        Text(store.powerStatus+". Il display può spegnersi; stop manuale e chiusura del portatile seguono macOS.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
    }.onAppear{store.refreshLoginStatus()}}
}
struct TypographySettingsView:View{
    @EnvironmentObject var store:StudioStore
    private var families:[String]{["System"]+NSFontManager.shared.availableFontFamilies.sorted()}
    var body:some View{
        Picker("Font interfaccia",selection:Binding(get:{store.archive.settings.uiFont ?? "System"},set:{store.archive.settings.uiFont=$0;store.applyTypography()})){ForEach(families,id:\.self){Text($0=="System" ? "Sistema":$0).tag($0)}}
        HStack{Text("Dimensione interfaccia");Slider(value:Binding(get:{store.archive.settings.uiFontSize ?? 13},set:{store.archive.settings.uiFontSize=$0;store.applyTypography()}),in:10...18,step:0.5);Text(String(format:"%.1f pt",store.archive.settings.uiFontSize ?? 13)).monospacedDigit()}
        Picker("Font chat",selection:Binding(get:{store.archive.settings.chatFont ?? "System"},set:{store.archive.settings.chatFont=$0;store.applyTypography()})){ForEach(families,id:\.self){Text($0=="System" ? "Sistema":$0).tag($0)}}
        HStack{Text("Dimensione chat");Slider(value:Binding(get:{store.archive.settings.chatFontSize ?? 14.5},set:{store.archive.settings.chatFontSize=$0;store.applyTypography()}),in:12...30,step:0.5);Text(String(format:"%.1f pt",store.archive.settings.chatFontSize ?? 14.5)).monospacedDigit()}
        Text("Anteprima chat: testo leggibile, alla tua misura.").font(StudioTypography.chat())
        Button("Ripristina font di sistema"){store.archive.settings.uiFont=nil;store.archive.settings.uiFontSize=nil;store.archive.settings.chatFont=nil;store.archive.settings.chatFontSize=nil;store.applyTypography()}
    }
}
