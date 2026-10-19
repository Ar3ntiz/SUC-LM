# SUC Studio 1.4.7.04

Versione unica **SUC Studio**, Universal per Intel e Apple Silicon, macOS 15 o superiore.

- Apple Silicon: motore SUC C++/Metal locale oppure server remoto.
- Intel: solo client OpenAI compatibile; nessun porting del motore SUC.
- Sessioni/dati: `~/Library/Application Support/SUC Studio Universal Preview/`.
- Bundle ID: `local.sucstudio.universal.app`. Identità e credenziali della versione Universal conservate.
- Installazione: aprire `dist/SUC-Studio-1.4.7.04-universal.pkg`.
  Installa `/Applications/SUC Studio.app` come unica versione del prodotto.
- Il pacchetto usa firma ad hoc, senza Developer ID/notarizzazione.
- Non importa automaticamente dati, plugin o credenziali dalla precedente versione non Universal.
  Esportare/importare le sessioni dall’interfaccia; reinserire le API key e riconfigurare gli strumenti.

## Aggiornamento alla 1.4.7.03

Chiudere le copie dell’app prima dell’installazione. Il pacchetto installa `SUC Studio.app` e rimuove la precedente `SUC Studio Universal Preview.app` solo se ha l’identificativo previsto. Le chat, impostazioni, credenziali e risorse della Universal restano disponibili: il percorso dati storico e il bundle ID sono mantenuti intenzionalmente. La cartella sorgenti unica è ora `SUCStudio/`; il motore rimane quello di `../cpp`.

## Setup e server remoto

Aprire Server, scegliere Server remoto (obbligatorio su Intel), impostare URL base
con prefisso `/v1`, API key se necessaria e ID modello. Usare Verifica / modelli,
poi Salva server remoto. Aprire Chat e creare una sessione.
Esempi: `http://192.0.2.20:8080/v1` oppure `http://127.0.0.1:8080/v1` per un
motore esterno in esecuzione sullo stesso Mac. Modelli, vision, strumenti e metriche
dipendono dalle capacità del server. Nessun modello/tokenizer locale è richiesto su Intel.

## Build e verifica

Mantenere `SUCStudio` dentro SUC-LM accanto a `cpp`.
Il motore viene compilato dai sorgenti del padre, incluse modifiche non committate.
Non usare il motore del vecchio worktree. App e helper shell/terminale sono
compilati per arm64 e x86_64 e uniti con lipo; il motore integrato rimane arm64
ed è bloccato su Intel. Per compilare entrambe le architetture eseguire:

```sh
./scripts/build.sh
./scripts/test-universal.sh
./scripts/package.sh
```

La build seleziona SDK 26.5 se presente (oppure SDKROOT esplicito).
Un singolo installer contiene entrambe le architetture. Le build per entrambe le architetture e i test automatici non sostituiscono
il collaudo dell’interfaccia su Mac Intel e Apple Silicon.
Collaudo utente completato su Intel, M3 e M1 Ultra.

Vedere [USAGE.md](USAGE.md), [PROGRESS.md](PROGRESS.md) e [docs/QA_REPORT.md](docs/QA_REPORT.md).

Novità della versione: [Changelog](dist/CHANGELOG.md).
