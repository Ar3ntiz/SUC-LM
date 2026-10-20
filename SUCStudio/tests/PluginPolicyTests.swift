import Foundation
@main struct PluginPolicyTests {
    static func main() {
        var skill=StudioSkill(name:"Private",description:"Explicit",instructions:"EXPLICIT_BODY",enabled:true);skill.allowImplicitInvocation=false
        let manifest=PluginManifest(formatVersion:1,identifier:"policy.test",name:"Policy",version:"1",description:"Metadata only")
        var plugin=InstalledPlugin(manifest:manifest,source:"fixture",skills:[skill],servers:[],hooks:[]);plugin.enabled=true
        precondition(!PluginCatalog.load("policy.test",plugins:[plugin],skills:[skill])!.contains("EXPLICIT_BODY"))
        precondition(PluginCatalog.load("policy.test",plugins:[plugin],skills:[skill],explicit:true)!.contains("EXPLICIT_BODY"))
        plugin.enabled=false
        precondition(PluginCatalog.load("policy.test",plugins:[plugin],skills:[skill],explicit:true)==nil)
        print("Plugin policy PASS: explicit-only skills respected; disabled plugins unavailable.")
    }
}
