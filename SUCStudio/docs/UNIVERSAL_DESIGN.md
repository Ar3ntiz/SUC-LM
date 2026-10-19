# Variante Universal parallela

Versione 1.4.5_Universal, derivata dalla cartella locale SUCStudio 1.4.5_Final.
I sette file Swift intermedi disallineati con la UI finale sono recuperati dalla
release registrata nel commit 33faf3db9e691d0639dd769aedc0740cbb242e5a.
Il recupero riguarda solo la variante Universal; il motore e le memory policy
restano quelli correnti sul disco. Vedere SOURCE_BASELINE.json.

| Componente | Architetture |
|---|---|
| App SwiftUI | arm64 e x86_64 |
| Helper shell e PTY | arm64 e x86_64 |
| Motore SUC C++/Metal | arm64 |

StudioPlatform centralizza la disponibilità del motore. Su x86_64 il backend
remoto è obbligatorio anche dopo il ripristino di preferenze locali. UI e metodi
loadModel/startServer/EngineClient.start applicano lo stesso controllo.
Il motore non viene avviato durante l’inizializzazione dell’app.

Bundle ID, percorso Application Support, Keychain e frame della finestra sono
separati dalla versione stabile. I profili WebKit hanno nuovi UUID. Artifact,
plugin installati e dati web seguono l’archivio separato. L’importazione delle
sessioni non importa le credenziali né le impostazioni globali.

La preview può comunque eseguire strumenti nella directory scelta dall’utente:
non è una sandbox dei comandi. Evitare prove simultanee delle due app nella
stessa directory di lavoro se gli strumenti possono modificare gli stessi file.

Il solo motore autorevole rimane ../cpp; gli script calcolano un fingerprint dei
sorgenti prima e dopo la compilazione. La prova iniziale usa i sorgenti locali,
non quelli arretrati del worktree. ENGINE_BUILD_MANIFEST.json identifica gli input.

La build arm64 può essere compilata su Intel, ma il suo runtime richiede una
verifica su Apple Silicon. I test di piattaforma x86_64 verificano il blocco del
motore; i test HTTP verificano streaming, errori, auth, cancellazione e telemetria.

Sostituire la versione stabile solo dopo il collaudo dell’interfaccia e delle
funzionalità su entrambe le piattaforme. Fino ad allora i due alberi sono distinti.

## 1.4.6 — chat, routine e skill

ChatMessage.protocolText separa il thinking dal protocollo delle azioni.
I payload operativi restano nell’archivio e nel registro, ma non sono renderizzati;
ActivityMessageBox visualizza solo i riepiloghi/avvisi compatti.
ChatSession.hasUnreadMessages è opzionale per leggere gli archivi precedenti.

StudioRoutine usa repeatEnabled/repeatCount/repeatUnit opzionali: quando assenti
mantiene la precedente ricorrenza in intervalMinutes. Le nuove routine sono una
tantum. Calendar gestisce giorni/mesi e DST; nextRun viene salvato prima dell’invio.
activeRoutines identifica la sessione prima del dispatch: l’avvio del browser
può ereditare l’approvazione automatica, mentre quella manuale resta inline.

StudioSkill.description e PluginSkill.description sono opzionali. PromptComposer
elenca titoli, ID e descrizioni; load_skill restituisce solo skill attive, senza
eseguire file/script o modificare i permessi. SkillDocument legge il frontmatter
comune di SKILL.md. Le istruzioni caricate restano nella cronologia del turno.
