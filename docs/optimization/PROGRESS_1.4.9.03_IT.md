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
