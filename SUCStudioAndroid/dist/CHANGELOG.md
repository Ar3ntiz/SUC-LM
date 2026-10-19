# SUC Studio Android — changelog

## 0.1.3

Corretto il prefill con cache: dai tempi SUC si contano solo i token nuovi, rispettando il conteggio già netto di llama.cpp e le velocità esplicite dei server. Visualizzati i token riusati; con cache completa nessuna velocità artificiale. Gestiti frame SSE separati e server senza dati cache. Sessioni e impostazioni conservate.

## 0.1.2

Corretto lo scorrimento della chat durante lo streaming: risalendo si sospende il seguito automatico e si conserva la posizione di lettura anche a risposta conclusa. Tornando in fondo o usando la freccia si riattiva il seguito. Eliminati i salti causati da frame accodati e aggiornamenti del contenuto.

## 0.1.1

Corretta la gestione CAPTCHA: browser integrato a schermo intero e riapribile dalla chat, controlli periodici anche senza navigazione, attesa senza timeout e annullamento esplicito. Le pagine di verifica non vengono inviate al modello. Aggiunti ricarica, apertura nel browser esterno e incolla testo per riprendere manualmente quando il sito rifiuta la WebView.

## 0.1.0

Prima versione Android per smartphone e tablet: chat e sessioni locali, server OpenAI compatibili, streaming, thinking, Markdown, immagini e documenti. Prompt, skill, MCP HTTP, plugin compatibili, ricerca web con conferme inline e artifact salvabili. Metriche reali o stimate, temi e font. Esclusi motore locale, stdio, hook shell, terminale, routine e notifiche.
