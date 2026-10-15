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
        GroupBox("Motore per la chat") {
            VStack(alignment:.leading,spacing:14) {
                Picker("Backend", selection:Binding(get:{ store.archive.settings.chatBackend ?? "local" },set:{ store.archive.settings.chatBackend = $0; store.scheduleSave() })) {
                    Text("Server locale").tag("local")
                    Text("Server remoto").tag("remote")
                }.pickerStyle(.segmented).disabled(store.hasWork)
                if store.usesRemote {
                    TextField("Base API · http://192.168.1.2:8080/v1",text:$config.endpoint).textFieldStyle(.roundedBorder)
                    HStack { TextField("ID modello",text:$config.model).textFieldStyle(.roundedBorder)
                        if !models.isEmpty { Menu("Scegli modello") { ForEach(models,id:\.self) { model in Button(model) { config.model = model } } } }
                    }
                    SecureField("API key (facoltativa se il server non la richiede)",text:$key).textFieldStyle(.roundedBorder)
                    Picker("Profilo telemetria",selection:Binding(get:{config.telemetryProfile ?? "auto"},set:{config.telemetryProfile=$0})) {
                        Text("Automatico / OpenAI compatibile").tag("auto");Text("SUC Studio").tag("suc");Text("DS4 (ds4-server)").tag("ds4");Text("LM Studio").tag("lmstudio");Text("llama.cpp").tag("llama");Text("LLM Router").tag("router");Text("Campi JSON personalizzati").tag("custom")
                    }
                    Toggle("Aggiorna le metriche durante lo streaming",isOn:Binding(get:{config.liveTelemetry ?? true},set:{config.liveTelemetry=$0}))
                    Toggle("Stima decode live quando il server non fornisce velocità",isOn:Binding(get:{config.estimateLiveDecode ?? false},set:{config.estimateLiveDecode=$0}))
                    Text("Le misure reali dipendono dal server. La stima live usa circa 4 caratteri per token ed è indicata con ≈; il prefill resta — senza tempi server. llama.cpp richiede timings_per_token e return_progress quando lo streaming metriche è attivo.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                    Text("Le generazioni lunghe restano attive anche durante prefill senza token. Ogni 15 secondi un health check separato aggiorna lo stato senza interrompere o ripetere la richiesta.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                    if config.telemetryProfile=="ds4" {Text("Il ds4-server verificato invia reasoning_content e usage finale, ma stampa le velocità nei log: non le espone nell’API.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)}
                    if config.telemetryProfile=="lmstudio" {Text("Usa /v1 per l’API compatibile oppure /api/v0 per le statistiche estese, se supportato dalla tua versione. I campi stats sono letti quando presenti.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)}
                    if config.telemetryProfile=="router" {Text("LLM Router conserva gli eventi OpenAI del backend. Per richiedere le estensioni di llama.cpp attraverso il router scegli il profilo llama.cpp.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)}
                    if config.telemetryProfile=="custom" {
                        TextField("Percorso JSON prefill tok/s (es. metrics.prefill_tps)",text:Binding(get:{config.customPrefillRatePath ?? ""},set:{config.customPrefillRatePath=$0})).textFieldStyle(.roundedBorder)
                        TextField("Percorso JSON decode tok/s (es. metrics.decode_tps)",text:Binding(get:{config.customDecodeRatePath ?? ""},set:{config.customDecodeRatePath=$0})).textFieldStyle(.roundedBorder)
                    }
                    Toggle("Invia parametri SUC/estesi (thinking, top_k, min_p)",isOn:$config.sendExtensions)
                    Text("Gli allegati immagine richiedono un modello vision e un server compatibile con image_url. Se non supportati, il server può rifiutare la richiesta.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                    Text("Prompt, cronologia e allegati vengono inviati al server scelto. Il motore locale e la sua API restano indipendenti.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                    HStack {
                        Button("Verifica / modelli") {
                            busy = true; status = "Connessione…"
                            Task { let client = RemoteInference(config,key:key); defer { client.stop(); busy = false }
                                do { models = try await client.models(); status = "Connesso · \(models.count) modelli"; if config.model.isEmpty { config.model = models.first ?? "" } }
                                catch { status = error.localizedDescription }
                            }
                        }.disabled(busy || !loaded)
                        Button("Salva server remoto") {
                            do { _ = try RemoteInference.endpoint(config.endpoint,path:"models"); try MCPSecrets.write(["key":key],id:config.id); store.archive.settings.remote = config; store.scheduleSave(); status = "Configurazione salvata" }
                            catch { status = error.localizedDescription }
                        }.buttonStyle(.borderedProminent).disabled(!loaded || busy || store.hasWork || config.model.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty)
                        if busy { ProgressView().controlSize(.small) }
                    }
                    if !status.isEmpty { Text(status).font(StudioTypography.ui(.caption)).textSelection(.enabled) }
                    Text("La base URL deve includere /v1 o il prefisso API del server. In caso di mancato supporto a /models puoi inserire direttamente l’ID. Non sono seguiti redirect HTTP.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary)
                } else { Text("La chat usa il modello caricato nei Settings. Qui sotto puoi esporlo anche ai client di rete.").font(StudioTypography.ui(.caption)).foregroundStyle(.secondary) }
            }.padding(12)
        }.onAppear { config = store.archive.settings.remote ?? RemoteConfiguration(); do { key = try MCPSecrets.read(config.id)["key"] ?? ""; loaded = true } catch { status = error.localizedDescription } }
    }
}
