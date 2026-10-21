import Foundation
@main struct LocalizationTests {
    static func main() throws {
        var settings=AppSettings()
        precondition(settings.effectiveLanguage=="en")
        let legacy=try JSONSerialization.jsonObject(with:JSONEncoder().encode(settings)) as! [String:Any]
        let migrated=try JSONDecoder().decode(AppSettings.self,from:JSONSerialization.data(withJSONObject:legacy.filter{$0.key != "language"}))
        precondition(migrated.effectiveLanguage=="en")
        StudioL10n.language=settings.effectiveLanguage
        precondition(L("Consenti comandi shell")=="Allow shell commands")
        precondition(L("Subtask (\(3))")=="Subtasks (3)")
        precondition(L("Rimuovi \("file {0}.txt")")=="Remove file {0}.txt")
        precondition(LT("Contesto: ≈150 / 4096 token")=="Context: ≈150 / 4096 tokens")
        precondition(LT("Server remoto · Sistema")=="Remote server · Sistema")
        precondition(LT("Untouched user content")=="Untouched user content")
        settings.language="it"
        let roundtrip=try JSONDecoder().decode(AppSettings.self,from:JSONEncoder().encode(settings))
        precondition(roundtrip.effectiveLanguage=="it")
        StudioL10n.language=roundtrip.effectiveLanguage
        precondition(L("Consenti comandi shell")=="Consenti comandi shell")
        precondition(L("Settings")=="Impostazioni")
        precondition(L("Subtask (\(3))")=="Subtask (3)")
        settings.language="unsupported";precondition(settings.effectiveLanguage=="en")
        var session=ChatSession();session.allowCommands=true
        precondition(!session.shellCommandsEnabled)
        session.workspacePath="/tmp";precondition(!session.shellCommandsEnabled)
        session.agentEnabled=true;precondition(session.shellCommandsEnabled)
        session.allowCommands=false;precondition(!session.shellCommandsEnabled)
        session.allowCommands=true;session.workspacePath="";precondition(!session.shellCommandsEnabled)
        for (key,translation) in StudioCatalog.english {
            let pattern=try NSRegularExpression(pattern:#"\{\d+\}"#)
            func tokens(_ value:String)->[String]{let ns=value as NSString;return pattern.matches(in:value,range:NSRange(location:0,length:ns.length)).map{ns.substring(with:$0.range)}.sorted()}
            precondition(tokens(key)==tokens(translation),"Placeholder mismatch: \(key)")
        }
        print("Localization: EN default/legacy migration, IT persistence, interpolation, catalog placeholders and shell prerequisites PASS")
    }
}
