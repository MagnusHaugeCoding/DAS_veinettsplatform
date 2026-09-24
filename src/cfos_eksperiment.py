"""
Overfører bil-deteksjonen til EKTE biler? Eksperiment på CFOS-data (se src/cfos_data.py).

Tre spørsmål, i denne rekkefølgen:
  1. NULLSKUDD: klarer modellene vi trente KUN på syntetiske data å finne ekte biler?
     Dette er den ærlige testen av generatoren vår — samme test som SAFOD var for
     jordskjelv.
  2. TRENT PÅ EKTE: hva klarer samme CNN-arkitektur når den får trene på ekte biler?
  3. FINJUSTERT: hjelper det å starte fra de syntetiske vektene?

Referansen (energi + koherens) kjøres ved siden av i alle ledd, av samme grunn som i
classifier.py: uten den er CNN-tallene ikke tolkbare.

GEOMETRI. Utsnittene gjøres like som i treningen: 2 s, re-samplet 30 Hz -> 100 Hz, så
et utsnitt er 200 tidssteg. Kanalene beholdes som de er (40 stk, antatt 2 m). Å ta hver
5. kanal for å få 10 m som i generatoren ville etterlatt bare 8 kanaler. I piksler er
helningen uansett sammenlignbar: en syntetisk bil i 80 km/t flytter seg ca. 2 kanaler/s,
en CFOS-bil ca. 4 kanaler/s (se fartssjekken under). CNN-et har global pooling og tåler
dermed 40 kanaler i stedet for 64 uten endring i vektene.

DELING ETTER TID, som ellers i prosjektet: de første 65 % av dagen er trening, de siste
25 % er test, og mellomrommet er karantenesone. Det betyr trening om morgenen/formiddagen
og test på ettermiddagen — slik et driftssystem også ville møtt nye data.

Kjør med:  python3 -m src.cfos_eksperiment
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy import ndimage
from scipy import signal as sig

from . import cfos_data
from .classifier import bygg_cnn, evaluer, klargjor_for_keras
from .features import (kanalstatistikk, normaliser_med_statistikk,
                       referanse_klassifiser, tilpass_referanse)
from .synthetic_das import BAKGRUNN, BIL, N_KLASSER

FS_MODELL = 100.0
VARIGHET_UTSNITT_S = float(os.environ.get("CFOS_UTSNITT_S", "2.0"))
ANDEL_TERSKEL = 0.12            # samme regel som lag_datasett() i classifier.py
TRENING_TIL, TEST_FRA = 0.65, 0.75


def fartssjekk(fasit: np.ndarray) -> np.ndarray:
    """
    Sjekker den ANTATTE kanalavstanden på 2 m: gir den fornuftige bilfarter?

    Vi finner hvert kjøretøy i fasitkartet som en sammenhengende flekk, og tilpasser en
    rett linje for tid mot kanal. Stigningen i kanaler per sekund ganget med antatt
    kanalavstand gir farten. Med 1 m ville en bygate gitt urimelig lave farter.
    """
    farter = []
    for kart in fasit[::4]:
        flekker, n = ndimage.label(kart > 0)
        for i in range(1, n + 1):
            kanal, tid = np.nonzero(flekker == i)
            if np.ptp(kanal) < 25:          # for kort spor til å gi en sikker stigning
                continue
            tid_per_kanal = np.polyfit(kanal, tid / cfos_data.SAMPLINGSRATE_HZ, 1)[0]
            if abs(tid_per_kanal) > 1e-3:
                farter.append(abs(cfos_data.KANALAVSTAND_M / tid_per_kanal) * 3.6)
    return np.array(farter)


def lag_utsnitt(X: np.ndarray, fasit: np.ndarray, start: np.ndarray):
    """
    Klipper 30-sekunders vinduer i 2-sekunders utsnitt i modellens geometri.

    Normaliseringen følger samme prinsipp som ellers: hver kanals median og spredning
    hentes fra en langtidsbakgrunn, her alle piksler som videoen sier er STØY i
    treningsperioden. Testperioden brukes ikke, så ingenting lekker.
    """
    # 30 Hz -> 100 Hz.
    Xr = sig.resample_poly(X, up=10, down=3, axis=2).astype(np.float32)
    fasit_r = np.repeat(fasit, 10, axis=2)[:, :, ::3][:, :, :Xr.shape[2]]

    rekkefolge = np.argsort(start)
    n = len(start)
    tren_idx = rekkefolge[: int(TRENING_TIL * n)]
    test_idx = rekkefolge[int(TEST_FRA * n):]

    # Kanalstatistikk fra STØY-piksler i treningsperioden.
    median = np.zeros((X.shape[1], 1), np.float32)
    skala = np.ones((X.shape[1], 1), np.float32)
    for k in range(X.shape[1]):
        stoy = Xr[tren_idx, k][fasit_r[tren_idx, k] == cfos_data.STOY]
        m, s = kanalstatistikk(stoy[None, :])
        median[k], skala[k] = m[0, 0], s[0, 0]

    n_t = int(VARIGHET_UTSNITT_S * FS_MODELL)

    def klipp(indekser):
        ut_X, ut_y = [], []
        for i in indekser:
            for t0 in range(0, Xr.shape[2] - n_t + 1, n_t):
                bit = Xr[i, :, t0:t0 + n_t]
                andel_bil = float((fasit_r[i, :, t0:t0 + n_t] > 0).mean())
                ut_X.append(normaliser_med_statistikk(bit, median, skala))
                ut_y.append(BIL if andel_bil > ANDEL_TERSKEL else BAKGRUNN)
        return np.array(ut_X), np.array(ut_y, dtype=np.int64)

    return (*klipp(tren_idx), *klipp(test_idx))


def som_bil_eller_ikke(pred: np.ndarray) -> np.ndarray:
    """Alt modellen ikke kaller bil, teller som bakgrunn — CFOS har bare de to."""
    return np.where(pred == BIL, BIL, BAKGRUNN)


def fleksibel_cnn(vekter_fra=None):
    """
    Samme arkitektur som i classifier.py, men med fri inndatastørrelse.

    Konvolusjonsvektene avhenger ikke av bildets størrelse, og global pooling
    oppsummerer uansett hvor mange kanaler som kommer inn. Dermed kan de syntetisk
    trente vektene brukes direkte på 40 kanaler.
    """
    modell = bygg_cnn((None, None, 1), N_KLASSER)
    if vekter_fra is not None:
        modell.set_weights(vekter_fra.get_weights())
    return modell


def tren(modell, X, y, laeringsrate):
    from tensorflow import keras

    modell.compile(
        optimizer=keras.optimizers.Adam(laeringsrate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    antall = np.bincount(y, minlength=N_KLASSER).astype(float)
    tilstede = antall > 0
    vekter = {k: float(len(y) / (tilstede.sum() * antall[k])) if antall[k] else 1.0
              for k in range(N_KLASSER)}
    # validation_split tar de SISTE utsnittene, og de er sortert etter tid. Valideringen
    # er dermed også senere i tid enn treningen, som i resten av prosjektet.
    modell.fit(
        klargjor_for_keras(X), y,
        validation_split=0.15, epochs=60, batch_size=64, class_weight=vekter,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                                 restore_best_weights=True)],
        verbose=2,
    )
    return modell


def prediker(modell, X):
    return np.argmax(modell.predict(klargjor_for_keras(X), verbose=0), axis=1)


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(0)
    np.random.seed(0)

    X, fasit, start = cfos_data.les()

    farter = fartssjekk(fasit)
    print(f"Fartssjekk ({len(farter)} spor, antatt {cfos_data.KANALAVSTAND_M:.0f} m kanalavstand): "
          f"median {np.median(farter):.0f} km/t, 10-90 %: "
          f"{np.percentile(farter, 10):.0f}-{np.percentile(farter, 90):.0f} km/t")

    X_tren, y_tren, X_test, y_test = lag_utsnitt(X, fasit, start)
    print(f"Trening: {len(X_tren)} utsnitt ({np.mean(y_tren == BIL):.0%} bil), "
          f"test: {len(X_test)} utsnitt ({np.mean(y_test == BIL):.0%} bil)")

    resultater = {}

    # --- 1. Nullskudd: trent kun på syntetisk -------------------------------------
    print("\n=== 1. NULLSKUDD: modellene har ALDRI sett en ekte bil ===")
    lagret = json.load(open("data/processed/resultater_fase23.json"))["terskler"]
    syn_terskler = (lagret["energi"], lagret["koherens"])
    y_ref0 = np.array([referanse_klassifiser(x, *syn_terskler) for x in X_test])
    resultater["nullskudd_referanse"] = evaluer(y_test, som_bil_eller_ikke(y_ref0),
                                                "Referanse, syntetiske terskler")

    syntetisk_cnn = keras.models.load_model("data/processed/cnn_klassifikator.keras")
    y_cnn0 = prediker(fleksibel_cnn(syntetisk_cnn), X_test)
    print("\nHva det syntetiske CNN-et svarte (alle testutsnitt):",
          {navn: int(np.sum(y_cnn0 == k)) for k, navn in
           enumerate(["bakgrunn", "bil", "jordskjelv", "noe annet"])})
    resultater["nullskudd_cnn"] = evaluer(y_test, som_bil_eller_ikke(y_cnn0),
                                          "CNN trent kun pa syntetisk")

    # --- 2. Trent på ekte biler ---------------------------------------------------
    print("\n=== 2. TRENT PA EKTE BILER (samme arkitektur, tilfeldige startvekter) ===")
    terskler = tilpass_referanse(X_tren, y_tren)
    y_ref = np.array([referanse_klassifiser(x, *terskler) for x in X_test])
    resultater["ekte_referanse"] = evaluer(y_test, som_bil_eller_ikke(y_ref),
                                           "Referanse, terskler tilpasset CFOS-trening")

    ny_cnn = tren(fleksibel_cnn(), X_tren, y_tren, 1e-3)
    resultater["ekte_cnn"] = evaluer(y_test, prediker(ny_cnn, X_test),
                                     "CNN trent pa CFOS fra bunnen")

    # --- 3. Finjustert fra syntetiske vekter ------------------------------------
    print("\n=== 3. FINJUSTERT: start fra syntetiske vekter, tren videre pa CFOS ===")
    fin_cnn = tren(fleksibel_cnn(syntetisk_cnn), X_tren, y_tren, 3e-4)
    resultater["finjustert_cnn"] = evaluer(y_test, prediker(fin_cnn, X_test),
                                           "CNN finjustert fra syntetisk")

    # --- Oppsummering -------------------------------------------------------------
    print("\n=== Oppsummering: bil-F1 pa EKTE testdata (ettermiddag) ===")
    for navn, r in resultater.items():
        print(f"{navn:<22} bil F1 {r['bil']['f1']:.3f}   presisjon {r['bil']['presisjon']:.3f}"
              f"   gjenkalling {r['bil']['gjenkalling']:.3f}")

    resultater["meta"] = dict(
        n_tren=int(len(X_tren)), n_test=int(len(X_test)),
        andel_bil_test=float(np.mean(y_test == BIL)),
        fart_median_kmt=float(np.median(farter)),
    )
    with open(f"data/processed/resultater_cfos_{VARIGHET_UTSNITT_S:.0f}s.json", "w") as f:
        json.dump(resultater, f, indent=2)
    fin_cnn.save(f"data/processed/cnn_cfos_finjustert_{VARIGHET_UTSNITT_S:.0f}s.keras")
    print(f"\nLagret: resultater_cfos_{VARIGHET_UTSNITT_S:.0f}s.json og cnn_cfos_finjustert_{VARIGHET_UTSNITT_S:.0f}s.keras")
