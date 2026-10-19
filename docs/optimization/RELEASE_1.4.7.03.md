# SUC Studio 1.4.7.03 — build 30

Cache dei turni Qwen qualificata sul Mac Studio M1 Ultra: il turno successivo da circa 2k passa da 21,0 a 1,13 s alla prima risposta; quello da 11.679 token da 108,0 a 0,94 s. Risposte identiche fra checkpoint disattivi/attivi. Il primo prefill lungo resta completo.

Un solo checkpoint degli stati ricorrenti/convoluzioni/n-gram CPU, massimo 128 MiB; KV e chiavi QSA del prefisso restano in sede. Sul Flash-Next side8 occupa circa 110,3 MiB. Prefisso incompatibile o snapshot indisponibile comportano ricalcolo; annullamento, errore e reload invalidano lo stato. Nessuna modifica ai pesi, ai kernel, al renderer SUC o alla memory policy.

950 posizioni reali e 235,9 milioni di logits identici, KL 0 fra cache e ricalcolo; fixture QSA anche a blocco parziale, fallback del Qwen27B e continuazioni verificati. Suite Universal Intel/M1, sette smoke CLI/server/helper, chat e cancellazione sul Qwen reale, rilascio effettivo del modello dopo unload. Conteggi indipendenti HF e parser Swift confermano i tok/s sui soli token nuovi. Non è una nuova valutazione hard o KL generale contro reference esterna.

L’helper rilascia le catture dei task asincroni conclusi, evitando mappature trattenute dopo unload. I dati personali e l’identità dell’app restano compatibili.

## File e utilizzo

- `SUCStudio/dist/SUC-Studio-1.4.7.03-universal.pkg`: macOS 15+, client Intel e inferenza locale Apple Silicon.
- `SUCStudio/dist/SUC-CLI-Server-1.4.7.03-arm64.tar.gz`: CLI/server Apple Silicon; target macOS verificato nel suo BUILD.json (26.0+ per questi binari).
- `SUCStudio/dist/SHA256SUMS-1.4.7.03.txt`: verifica dalla cartella dist con `shasum -a 256 -c SHA256SUMS-1.4.7.03.txt`.

Chiudere SUC Studio prima di eseguire l’installer; non occorre convertire i modelli o i tokenizer. Distribuzione locale con firma ad hoc, senza notarizzazione. Nessuna installazione automatica.

## Profili e protocollo

App/server mantengono chunk 64; CLI conserva il profilo 1024 per Qwen. I circa 366 tok/s della model card descrivono la CLI, non il profilo dell’app da circa 100–110 tok/s. La cache riduce i token rielaborati, non moltiplica tale velocità. La qualifica del profilo veloce nell’app è separata. Il renderer SUC conserva una differenza preesistente rispetto al Jinja HF nei turni assistant storici; la verifica dei conteggi usa il prompt effettivamente elaborato e un tokenizer indipendente.

Piano, risultati e limiti: [CHAT_PREFIX_CHECKPOINTS.md](CHAT_PREFIX_CHECKPOINTS.md). I risultati baseline 1.4.7 di model card e reference esterna rimangono invariati.
