# SUC Studio 1.4.9.03 · avanzamento

- [x] Diagnosi in sola lettura: promesse senza azioni, metriche con conteggi incompleti e JSON `write_file` senza chiusura finale.
- [x] Recupero IT/EN e correzione del formato, con massimo due tentativi condivisi per turno; niente esecuzioni dedotte.
- [x] Indicatore streaming senza ritorni a prefill dopo l’inizio del decode/ragionamento.
- [x] Regressioni streaming, file/artifact reali fixture, limiti, approvazioni/Stop, coda e risposte finali.
- [x] Build Universal 40, installer estratto, QA e checksum.
- [x] Commit e sincronizzazione sorgenti, Mac Studio e repository pubblico.

Nessuna modifica all’archivio live o esecuzione dei comandi dell’utente. Motore, memory policy, Android e misurazioni dei modelli invariati.

Collaudo completo superato su Intel e M1 Ultra. Installer estratto, firme, architetture, guide IT/EN e identità verificati. Checksum nel report QA; motore byte per byte identico alla release precedente.

## Correzioni build 41

- [x] Diagnosi in sola lettura dei marcatori visibili e dei documenti ripetuti con sola parentesi esterna mancante.
- [x] Involucri tecnici nascosti senza alterare archivio/commenti/esempi; validazione azione conservata.
- [x] Soglia 99% senza riserva output anticipata, con conteggi coerenti all’indicatore.
- [x] Tolleranza strutturale limitata alla sola chiusura esterna dopo stop normale; contenuto sempre integro.
- [x] Regressioni complete UI/protocollo, file/artifact, contesto/compattazione e collaudo nativo M1.
- [x] Installer Universal build 41 e sincronizzazione sorgenti/Mac Studio/pubblico.

## Interfaccia build 42

- [x] Rimozione del pannello fisso dal fondo della chat, preservando processi e registro azioni.
- [x] Toggle del comando su tutta la riga con area minima di 36 punti e stato accessibile.
- [x] Avvisi di errore indirizzati al registro azioni e guida IT/EN aggiornata.
- [x] Verifica UI di dimensioni, pulsante accessibile e toggle ripetuto nella finestra fixture; registro consultabile con configurazione bloccata.
- [x] Build Universal 42, installer verificato e sincronizzazione sorgenti/Mac Studio/pubblico.

- [x] Registro azioni a schede leggibili, più alto, con richiesta/risultato separati, copia/apertura integrale e preview limitate.
- [x] Consultazione del registro separata dal blocco delle impostazioni durante l’esecuzione.

## Compattazione build 43

- [x] Diagnosi della sessione reale in sola lettura e riproduzione del budget negativo.
- [x] Budget coerenti in token, blocchi Unicode senza perdita, retry entro finestra.
- [x] Riduzione limitata della memoria precedente quando la finestra viene diminuita.
- [x] Risultato operativo conservato una sola volta dopo errore o Stop, senza riesecuzione.
- [x] Suite contesto completa Intel e M1 Ultra, inclusi i nuovi casi HTTP.
- [x] Build Universal 43, installer, checksum, commit e sincronizzazione sorgenti/Mac Studio/pubblico.

## Contesto/thinking build 44

- [x] Diagnosi in sola lettura della sessione e del router; verificata la build 43 installata.
- [x] Reasoning non reinviato escluso dal contesto futuro, conteggi server opzionali e stima legacy.
- [x] Controllo thinking per profilo e override persistente IT/EN; nessun retry HTTP generico.
- [x] Sintesi breve, memoria precedente ridotta, metà Unicode e recuperi limitati senza commit parziali.
- [x] Suite contesto/telemetria/remota complete su Intel e M1 Ultra; prova reale neutra del Router.
- [x] Localizzazione, build Universal 44, installer verificato e sincronizzazione sorgenti/Mac Studio/pubblico.
