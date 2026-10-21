import SwiftUI

struct BackendSettingsView: View {
    @EnvironmentObject var store: StudioStore
    @State private var config = RemoteConfiguration()
    @State private var key = ""
    @State private var loaded = false
    @State private var busy = false
    @State private var models: [String] = []
    @State private var status = ""
    var body: some View {
        GroupBox(L("Motore per la chat")) {
            VStack(alignment:.leading,spacing:14) {
                Picker("Backend", selection:Binding(get:{ StudioPlatform.effectiveBackend(store.archive.settings.chatBackend) },set:{ store.archive.settings.chatBackend = StudioPlatform.effectiveBackend($0); store.scheduleSave() })) {
                    if StudioPlatform.localInferenceAvailable { Text(L("Server locale")).tag("local") }
                    Text(L("Server remoto")).tag("remote")
                }.pickerStyle(.segmented).disabled(store.hasWork)
                if !StudioPlatform.localInferenceAvailable {
                    Label(StudioPlatform.localUnavailableMessage, systemImage: "network").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                }
                if store.usesRemote {
                    TextField(L("Base API · http://192.0.2.2:8080/v1"),text:$config.endpoint).textFieldStyle(.roundedBorder)
                    HStack { TextField(L("ID modello"),text:$config.model).textFieldStyle(.roundedBorder)
                        if !models.isEmpty { Menu(L("Scegli modello")) { ForEach(models,id:\.self) { model in Button(model) { config.model = model } } } }
                    }
                    SecureField(L("API key (facoltativa se il server non la richiede)"),text:$key).textFieldStyle(.roundedBorder)
                    Picker(L("Profilo telemetria"),selection:Binding(get:{config.telemetryProfile ?? "auto"},set:{config.telemetryProfile=$0})) {
                        Text(L("Automatico / OpenAI compatibile")).tag("auto");Text("SUC Studio").tag("suc");Text("DS4 (ds4-server)").tag("ds4");Text("LM Studio").tag("lmstudio");Text("llama.cpp").tag("llama");Text("LLM Router").tag("router");Text(L("Campi JSON personalizzati")).tag("custom")
                    }
                    Toggle(L("Aggiorna le metriche durante lo streaming"),isOn:Binding(get:{config.liveTelemetry ?? true},set:{config.liveTelemetry=$0}))
                    Toggle(L("Stima ricezione live quando il server non fornisce velocità"),isOn:Binding(get:{config.estimateLiveDecode ?? false},set:{config.estimateLiveDecode=$0}))
                    Text(L("Le misure reali dipendono dal server. La stima live usa circa 4 caratteri per token, è indicata come Ricezione ≈ e non misura la velocità del motore; il prefill resta — senza tempi server. llama.cpp richiede timings_per_token e return_progress quando lo streaming metriche è attivo.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                    Text(L("Le generazioni lunghe restano attive anche durante prefill senza token. Ogni 15 secondi un health check separato aggiorna lo stato senza interrompere o ripetere la richiesta.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                    if config.telemetryProfile=="ds4" {Text(L("Il ds4-server verificato invia reasoning_content e usage finale, ma stampa le velocità nei log: non le espone nell’API.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)}
                    if config.telemetryProfile=="lmstudio" {Text(L("Usa /v1 per l’API compatibile oppure /api/v0 per le statistiche estese, se supportato dalla tua versione. I campi stats sono letti quando presenti.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)}
                    if config.telemetryProfile=="router" {Text(L("LLM Router conserva gli eventi OpenAI del backend. Per richiedere le estensioni di llama.cpp attraverso il router scegli il profilo llama.cpp.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)}
                    if config.telemetryProfile=="custom" {
                        TextField(L("Percorso JSON prefill tok/s (es. metrics.prefill_tps)"),text:Binding(get:{config.customPrefillRatePath ?? ""},set:{config.customPrefillRatePath=$0})).textFieldStyle(.roundedBorder)
                        TextField(L("Percorso JSON decode tok/s (es. metrics.decode_tps)"),text:Binding(get:{config.customDecodeRatePath ?? ""},set:{config.customDecodeRatePath=$0})).textFieldStyle(.roundedBorder)
                    }
                    Toggle(L("Invia parametri SUC/estesi (thinking, top_k, min_p)"),isOn:$config.sendExtensions)
                    Text(L("Gli allegati immagine richiedono un modello vision e un server compatibile con image_url. Se non supportati, il server può rifiutare la richiesta.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                    Text(L("Prompt, cronologia e allegati vengono inviati al server scelto. Puoi usare un server in rete oppure localhost; l’inferenza SUC integrata è disponibile solo su Apple Silicon.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                    HStack {
                        Button(L("Verifica / modelli")) {
                            busy = true; status = L("Connessione…")
                            Task { let client = RemoteInference(config,key:key); defer { client.stop(); busy = false }
                                do { models = try await client.models(); config.modelContextWindows=client.modelContextWindows; status = L("Connesso · \(models.count) modelli"); if config.model.isEmpty { config.model = models.first ?? "" } }
                                catch { status = error.localizedDescription }
                            }
                        }.disabled(busy || !loaded)
                        Button(L("Salva server remoto")) {
                            do { _ = try RemoteInference.endpoint(config.endpoint,path:"models"); try MCPSecrets.write(["key":key],id:config.id); store.archive.settings.remote = config; store.scheduleSave(); status = L("Configurazione salvata") }
                            catch { status = error.localizedDescription }
                        }.buttonStyle(.borderedProminent).disabled(!loaded || busy || store.hasWork || config.model.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty)
                        if busy { ProgressView().controlSize(.small) }
                    }
                    if !status.isEmpty { Text(LT(status)).font(StudioTypography.ui(.caption)).textSelection(.enabled) }
                    Text(L("La base URL deve includere /v1 o il prefisso API del server. In caso di mancato supporto a /models puoi inserire direttamente l’ID. Non sono seguiti redirect HTTP.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                } else { Text(L("La chat usa il modello caricato nei Settings. Qui sotto puoi esporlo anche ai client di rete.")).font(StudioTypography.ui(.caption)).foregroundStyle(.secondary) }
            }.padding(12)
        }.onAppear { config = store.archive.settings.remote ?? RemoteConfiguration(); do { key = try MCPSecrets.read(config.id)["key"] ?? ""; loaded = true } catch { status = error.localizedDescription } }
    }
}
