# SUC Studio 1.4.9.05 · build 47

- [x] Riprodurre i formati GLM non canonici senza eseguire contenuti della sessione.
- [x] Riconoscere marcatori inline e prefissi streaming, preservando codice, esempi tra virgolette e testo originale.
- [x] Richiedere JSON canonico con recupero limitato, senza interpretare chiamate posizionali o inventare argomenti.
- [x] Bloccare la quarta operazione consecutiva identica dopo tre esiti terminali uguali, escludendo stato/attesa/annullamento dei job.
- [x] Collaudare parsing, streaming, recupero, approvazioni, Stop, correzioni in coda e controllo dei cicli su Intel/M1.
- [x] Compilare e verificare installer Universal build 47, guide e changelog.
- [x] Committare e sincronizzare sorgenti, Mac Studio e PUBLIC.

Il controllo dei cicli è un’euristica conservativa: non prova l’assenza di effetti. Il conteggio vale per un singolo ciclo e si azzera con un nuovo turno utente, non con una correzione in corso. Tre risultati diversi, un’operazione interposta diversa e i job ancora attivi non equivalgono a tre esiti terminali identici. Nessun intervento sulla sessione live, sui provider REA o sui binari esaminati dall’utente.
