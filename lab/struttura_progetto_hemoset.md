# Struttura del progetto HemoSet

Questa struttura separa:

- il **codice**, gli script SLURM e i file CSV degli split in `/homes`;
- il **dataset**, i checkpoint e i risultati degli esperimenti in `/work`.

## Directory in `/homes`

```text
/homes/<username>/hemoset_project/
├── scripts/
│   ├── create_split.py
│   ├── train_unet.py
│   └── train_unetplusplus.py
│
├── splits/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
├── configs/
│
└── jobs/
    ├── train_unet.slurm
    └── train_unetplusplus.slurm
```

### Contenuto delle cartelle

- `scripts/`: contiene gli script Python per creare lo split e allenare i modelli.
- `splits/`: contiene i tre file CSV condivisi da U-Net e U-Net++.
- `configs/`: contiene eventuali file di configurazione, ad esempio percorsi e iperparametri.
- `jobs/`: contiene gli script SLURM usati per lanciare i job sul cluster.

## Directory in `/work`

```text
/work/<project>/<username>/
├── datasets/
│   └── HemoSet/
│       ├── pig_01/
│       ├── pig_02/
│       ├── pig_03/
│       └── ...
│
└── experiments/
    ├── unet/
    │   ├── checkpoints/
    │   ├── logs/
    │   └── predictions/
    │
    └── unetplusplus/
        ├── checkpoints/
        ├── logs/
        └── predictions/
```

### Contenuto delle cartelle

- `datasets/HemoSet/`: contiene le 11 cartelle `pig` e i relativi frame e maschere.
- `experiments/unet/`: contiene i risultati degli esperimenti eseguiti con U-Net.
- `experiments/unetplusplus/`: contiene i risultati degli esperimenti eseguiti con U-Net++.
- `checkpoints/`: contiene i pesi salvati durante o dopo il training.
- `logs/`: contiene le loss, le metriche e gli eventuali log di TensorBoard.
- `predictions/`: contiene le maschere predette sul validation set o sul test set.

## Collegamento tra `/homes` e `/work`

Gli script presenti in `/homes` leggono:

- i CSV degli split da `/homes/<username>/hemoset_project/splits/`;
- immagini e maschere da `/work/<project>/<username>/datasets/HemoSet/`;
- salvano checkpoint, log e predizioni in `/work/<project>/<username>/experiments/`.

U-Net e U-Net++ devono usare gli stessi file:

```text
train.csv
validation.csv
test.csv
```

In questo modo entrambi i modelli vengono allenati, validati e testati sugli stessi identici dati.
