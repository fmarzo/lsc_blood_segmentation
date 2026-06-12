# Documentazione: Semantic Segmentation, Instance Segmentation, U-Net e UNet++

## 1. Introduzione alla segmentazione nelle immagini

Nella **computer vision** esistono diversi task per comprendere il contenuto di un'immagine. La semplice **classificazione** assegna una sola classe all'intera immagine, per esempio "gatto" o "cane". Tuttavia, nel mondo reale un'immagine contiene spesso più oggetti, sfondi complessi e oggetti che possono sovrapporsi o toccarsi. Per questo motivo sono stati sviluppati task più dettagliati, come **object detection**, **semantic segmentation** e **instance segmentation**.

La **semantic segmentation** consiste nel classificare **ogni pixel** dell'immagine in una delle classi disponibili. In altre parole, la rete produce una mappa in cui ogni pixel riceve un'etichetta, per esempio "strada", "persona", "auto", "erba" o "sfondo". Il risultato finale è una mappa di segmentazione che rappresenta semanticamente l'intera immagine.

La **instance segmentation** è più dettagliata: non si limita a dire che un pixel appartiene alla classe "persona", ma distingue anche tra persone diverse. Se nell'immagine ci sono tre persone, la rete deve produrre tre maschere separate, una per ogni istanza. Questo è utile quando non basta sapere dove sono i pixel di una certa classe, ma serve anche **contare** e **separare** i singoli oggetti.

La differenza principale è quindi questa:

- nella **semantic segmentation**, tutti gli oggetti della stessa classe vengono fusi in un'unica categoria;
- nella **instance segmentation**, ogni oggetto viene separato dagli altri anche se appartiene alla stessa classe.

Per esempio, se due mucche sono vicine e si toccano, la semantic segmentation può assegnare a entrambe la classe "mucca", ma non distingue necessariamente dove finisce una e dove inizia l'altra. L'instance segmentation, invece, produce due maschere diverse.

---

## 2. Semantic segmentation vs instance segmentation

| Aspetto | Semantic Segmentation | Instance Segmentation |
|---|---|---|
| Obiettivo | Classificare ogni pixel | Classificare e separare ogni oggetto |
| Output | Una classe per ogni pixel | Una maschera per ogni istanza |
| Oggetti della stessa classe | Possono essere fusi | Sono separati |
| Esempio | Tutti i pixel delle persone hanno classe "persona" | Ogni persona ha una maschera diversa |
| Utilità | Comprensione globale della scena | Conteggio e separazione degli oggetti |

La **semantic segmentation** è quindi adatta quando si vuole capire la struttura complessiva della scena. L'**instance segmentation** è più utile quando è necessario distinguere oggetti individuali, per esempio in immagini mediche, guida autonoma, robotica o analisi di oggetti sovrapposti.

---

## 3. Perché servono reti encoder-decoder

Per fare segmentazione non basta classificare l'immagine intera. Serve produrre una predizione densa, cioè una predizione per ogni pixel.

Un approccio ingenuo sarebbe usare una **sliding window**: si prende una piccola porzione dell'immagine, la si passa a una CNN e si classifica il pixel centrale. Questo metodo però è molto inefficiente, perché molte finestre si sovrappongono e la rete ricalcola più volte feature simili.

Un approccio più moderno è usare una rete **fully convolutional**, cioè una rete composta principalmente da convoluzioni, capace di produrre una mappa spaziale di output. Tuttavia, durante l'elaborazione dell'immagine è utile ridurre progressivamente la risoluzione per catturare informazione più astratta e aumentare il **receptive field**. Alla fine, però, la segmentazione deve tornare alla risoluzione originale. Per questo motivo si usano architetture **encoder-decoder**.

---

## 4. Che cos'è un encoder

L'**encoder** è la parte della rete che comprime l'immagine in una rappresentazione più piccola ma più ricca semanticamente.

In una rete CNN classica, l'encoder è formato da:

- blocchi convoluzionali;
- funzioni di attivazione, per esempio ReLU;
- operazioni di downsampling, come max pooling o convoluzioni con stride maggiore di 1.

Durante il percorso dell'encoder:

1. la dimensione spaziale della feature map diminuisce;
2. il numero di canali aumenta;
3. le feature diventano più astratte;
4. il receptive field cresce, quindi ogni neurone "vede" una porzione più ampia dell'immagine.

Per esempio, partendo da un'immagine di dimensione `H x W`, dopo alcuni livelli di downsampling si può ottenere una feature map `H/2 x W/2`, poi `H/4 x W/4`, poi `H/8 x W/8`, e così via. Allo stesso tempo, i canali possono passare da 64 a 128, poi 256, poi 512.

L'encoder serve quindi a capire **che cosa** è presente nell'immagine e a estrarre informazioni globali e contestuali.

---

## 5. Che cos'è un decoder

Il **decoder** è la parte della rete che ricostruisce progressivamente la risoluzione spaziale dell'immagine a partire dalle feature compresse prodotte dall'encoder.

Il decoder usa operazioni di **upsampling**, cioè operazioni che aumentano la dimensione spaziale delle feature map. Alcuni metodi comuni sono:

- nearest neighbor upsampling;
- unpooling;
- max unpooling;
- transpose convolution, detta anche informalmente deconvolution;
- upsampling seguito da convoluzione.

La **transpose convolution** è un'operazione learnable: a differenza di un semplice ridimensionamento, ha pesi addestrabili e può imparare come ricostruire meglio la mappa spaziale.

Il decoder serve quindi a recuperare il **dove**, cioè la posizione precisa dei pixel da classificare.

---

## 6. Encoder e decoder sono CNN o Transformer?

Encoder e decoder non indicano necessariamente un tipo specifico di rete. Sono **ruoli architetturali**.

Nella **U-Net originale**, encoder e decoder sono basati su **CNN**, cioè reti convoluzionali. La U-Net nasce infatti come architettura fully convolutional per segmentazione biomedica.

Tuttavia, in architetture più recenti, l'encoder o il decoder possono essere anche basati su **Transformer**. Nei Transformer l'immagine viene spesso divisa in patch o token, e la rete usa meccanismi di **self-attention** per modellare relazioni anche molto distanti nell'immagine. Alcuni modelli moderni usano un encoder Transformer e un decoder convoluzionale, oppure combinano CNN e Transformer.

Quindi:

- **U-Net classica**: encoder-decoder CNN;
- **UNet++ classica**: encoder-decoder CNN con skip connections dense e annidate;
- **modelli moderni tipo TransUNet o Swin-Unet**: architetture ispirate a U-Net ma con componenti Transformer;
- **DETR per object detection**: usa una CNN per estrarre feature e poi un Transformer encoder-decoder per predire oggetti.

---

## 7. U-Net: idea generale

La **U-Net** è un'architettura progettata originariamente per la segmentazione di immagini biomediche. Il nome deriva dalla sua forma a **U**: a sinistra c'è il percorso di contrazione, cioè l'encoder; a destra c'è il percorso di espansione, cioè il decoder.

La U-Net è molto usata perché combina due esigenze:

1. catturare contesto globale tramite downsampling;
2. mantenere dettagli spaziali tramite skip connections.

Nella segmentazione medica, per esempio, è fondamentale localizzare con precisione bordi, cellule, tessuti o lesioni. Se si perde troppa informazione spaziale durante il downsampling, il risultato può diventare impreciso. Le skip connections servono proprio a ridurre questo problema.

---

## 8. Struttura della U-Net

La U-Net è composta da tre parti principali:

1. **Contracting path / encoder**;
2. **bottleneck**;
3. **expanding path / decoder**.

### 8.1 Contracting path / Encoder

È il ramo discendente della U. In questa parte la rete applica blocchi convoluzionali e downsampling.

Un blocco tipico contiene:

1. convoluzione `3x3`;
2. funzione di attivazione, per esempio ReLU;
3. seconda convoluzione `3x3`;
4. altra ReLU;
5. max pooling o strided convolution per dimezzare la risoluzione.

A ogni livello, la risoluzione si riduce e il numero di canali aumenta. Questo permette alla rete di imparare feature sempre più astratte.

### 8.2 Bottleneck

Il **bottleneck** è la parte più profonda della rete. Qui la risoluzione è minima, ma il numero di canali è massimo. Questa parte contiene l'informazione più astratta e contestuale dell'immagine.

Il bottleneck rappresenta il punto di passaggio tra encoder e decoder.

### 8.3 Expanding path / Decoder

È il ramo ascendente della U. In questa parte la rete aumenta progressivamente la risoluzione delle feature map.

Ogni livello del decoder solitamente fa:

1. upsampling o transpose convolution;
2. concatenazione con la feature map corrispondente dell'encoder;
3. convoluzioni per raffinare le feature.

La concatenazione con l'encoder è la parte fondamentale: permette al decoder di usare sia informazione semantica profonda sia dettagli spaziali fini.

---

## 9. Skip connections nella U-Net

Le **skip connections** collegano ogni livello dell'encoder con il livello corrispondente del decoder.

Il motivo è semplice: durante il downsampling l'encoder perde parte della precisione spaziale. Le feature profonde sono molto utili per capire il contenuto dell'immagine, ma sono meno precise nei dettagli locali. Le feature dei primi livelli, invece, contengono bordi, texture e dettagli spaziali.

Con le skip connections, il decoder riceve entrambe le informazioni:

- feature profonde, utili per il significato semantico;
- feature superficiali, utili per localizzare meglio i bordi.

In U-Net, le skip connections sono spesso implementate tramite **concatenazione** lungo la dimensione dei canali.

---

## 10. Output della U-Net

L'output della U-Net è una mappa di segmentazione.

Per una segmentazione binaria, per esempio oggetto/sfondo, l'ultimo layer può produrre una sola mappa `H x W`, seguita da una **sigmoid**. Ogni pixel avrà un valore tra 0 e 1, interpretabile come probabilità di appartenere all'oggetto.

Per una segmentazione multi-classe, l'ultimo layer produce una mappa `H x W x C`, dove `C` è il numero di classi. Per ogni pixel si ottiene un vettore di probabilità sulle classi. Di solito si usa una **softmax** e poi un **argmax** per scegliere la classe finale di ogni pixel.

Le loss più comuni sono:

- binary cross entropy, per segmentazione binaria;
- categorical cross entropy, per segmentazione multi-classe;
- Dice loss, molto usata in ambito medico;
- IoU loss o combinazioni di più loss.

---

## 11. U-Net e semantic segmentation

La U-Net è principalmente usata per **semantic segmentation**. La rete classifica ogni pixel dell'immagine in una classe.

Nel caso di immagini mediche, la classe può essere "tumore", "organo", "cellula", "lesione" o "sfondo". Nel caso di immagini stradali, le classi possono essere "strada", "macchina", "persona", "cielo", "marciapiede".

La U-Net non nasce come modello di instance segmentation. Tuttavia, in alcuni casi può essere usata per segmentare oggetti singoli o oggetti separabili. Per ottenere una vera instance segmentation, spesso servono passaggi aggiuntivi di post-processing oppure architetture specifiche come **Mask R-CNN**.

---

## 12. Limiti della U-Net

La U-Net è efficace, ma ha alcuni limiti.

Il primo limite riguarda il cosiddetto **semantic gap**. Le feature dell'encoder e quelle del decoder possono avere significati molto diversi: le feature dell'encoder nei primi livelli sono più locali e meno semantiche, mentre quelle del decoder sono più astratte. Collegarle direttamente tramite skip connection può non essere sempre ottimale.

Il secondo limite è che la U-Net classica usa skip connections semplici. Queste aiutano a recuperare dettagli, ma non sempre fondono in modo graduale le informazioni tra livelli diversi.

Il terzo limite riguarda la capacità del modello. Per problemi molto complessi, la U-Net base può non essere abbastanza potente, oppure può richiedere modifiche come residual blocks, attention, dilated convolutions o backbone più avanzati.

---

## 13. UNet++: idea generale

**UNet++** è una variante della U-Net progettata per migliorare la qualità della segmentazione, soprattutto in ambito medico.

L'idea principale di UNet++ è modificare le skip connections. Invece di collegare direttamente encoder e decoder con una singola connessione, UNet++ usa **skip pathways annidati e densi**.

Questi percorsi intermedi servono a ridurre il semantic gap tra encoder e decoder. In altre parole, prima di fondere le feature dell'encoder con quelle del decoder, UNet++ le trasforma gradualmente tramite blocchi convoluzionali intermedi.

---

## 14. Differenza tra U-Net e UNet++

Nella U-Net classica, le connessioni sono dirette:

```text
encoder livello 1 -> decoder livello 1
encoder livello 2 -> decoder livello 2
encoder livello 3 -> decoder livello 3
```

In UNet++, invece, tra encoder e decoder ci sono nodi intermedi. Le feature vengono elaborate più volte e combinate in modo denso. Questo crea una struttura a griglia, in cui ogni nodo riceve informazioni da livelli precedenti e da livelli più profondi.

La differenza principale è quindi:

- **U-Net**: skip connections semplici e dirette;
- **UNet++**: skip connections dense, annidate e progressive.

---

## 15. Come funziona UNet++

UNet++ può essere vista come una U-Net con percorsi di skip più complessi.

Ogni livello dell'encoder produce feature map a una certa risoluzione. Invece di passarle direttamente al decoder, UNet++ le fa attraversare una serie di blocchi convoluzionali intermedi. Questi blocchi rendono le feature dell'encoder più simili semanticamente a quelle del decoder.

Questo aiuta perché il decoder riceve feature più compatibili con il proprio livello di astrazione.

UNet++ introduce spesso anche la **deep supervision**. Ciò significa che la rete può produrre output di segmentazione a diverse profondità del decoder. Durante il training, ogni output può contribuire alla loss. Questo rende l'addestramento più stabile e permette anche di usare versioni più leggere della rete in fase di inferenza.

---

## 16. Vantaggi di UNet++

I vantaggi principali di UNet++ sono:

1. riduce il semantic gap tra encoder e decoder;
2. migliora la fusione tra feature locali e globali;
3. può produrre segmentazioni più precise;
4. usa deep supervision per facilitare il training;
5. è particolarmente utile in segmentazione medica, dove i bordi e i dettagli sono molto importanti.

---

## 17. Svantaggi di UNet++

UNet++ è più complessa della U-Net classica.

Poiché aggiunge molti nodi intermedi e connessioni dense, richiede più memoria e più tempo di calcolo. Inoltre, può essere più difficile da implementare e da ottimizzare.

In generale:

- U-Net è più semplice, leggera e facile da usare;
- UNet++ è più potente, ma più costosa computazionalmente.

---

## 18. Confronto sintetico tra U-Net e UNet++

| Aspetto | U-Net | UNet++ |
|---|---|---|
| Tipo di rete | CNN encoder-decoder | CNN encoder-decoder annidata |
| Skip connections | Dirette | Dense e annidate |
| Obiettivo | Recuperare dettagli spaziali | Ridurre il semantic gap |
| Complessità | Minore | Maggiore |
| Memoria richiesta | Più bassa | Più alta |
| Accuratezza | Buona | Spesso migliore |
| Uso tipico | Semantic segmentation | Segmentazione medica avanzata |

---

## 19. Collegamento con CNN e Transformer

La U-Net originale e UNet++ sono architetture basate su **CNN**. Le convoluzioni sono adatte alle immagini perché lavorano localmente, condividono i pesi e mantengono la struttura spaziale.

I **Transformer**, invece, usano self-attention e sono molto efficaci nel modellare relazioni globali. Possono essere utili quando è importante collegare parti lontane dell'immagine. Tuttavia, spesso richiedono più dati e più risorse computazionali.

Oggi esistono molte architetture ibride: alcune mantengono la struttura encoder-decoder della U-Net, ma sostituiscono l'encoder CNN con un Transformer, oppure aggiungono moduli di attention nel decoder.

Quindi, U-Net e UNet++ non sono Transformer nella loro forma originale, ma il concetto di architettura a U può essere combinato con componenti Transformer.

---

## 20. Conclusione

La **semantic segmentation** e l'**instance segmentation** sono due task fondamentali della computer vision. La prima assegna una classe a ogni pixel, mentre la seconda distingue anche le singole istanze degli oggetti.

La **U-Net** è una delle architetture più importanti per la semantic segmentation. Usa una struttura encoder-decoder basata su CNN, con skip connections che permettono di recuperare dettagli spaziali persi durante il downsampling.

**UNet++** migliora la U-Net introducendo skip connections dense e annidate. Queste connessioni riducono il semantic gap tra encoder e decoder e possono migliorare la precisione della segmentazione, soprattutto in ambito medico.

In sintesi, la U-Net classica è semplice, efficace e molto usata; UNet++ è una versione più sofisticata, pensata per ottenere segmentazioni più accurate quando i dettagli sono particolarmente importanti.

---

## Riferimenti

- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*.
- Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2018). *UNet++: A Nested U-Net Architecture for Medical Image Segmentation*.
- Appunti e materiale del PDF fornito: sezioni su semantic segmentation, instance segmentation, upsampling, transpose convolution, skip connections, U-Net, SegNet, dilated convolution e Mask R-CNN.
