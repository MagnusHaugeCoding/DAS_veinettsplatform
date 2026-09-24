"""
Klassifikator for tre klasser: bakgrunn, bil, jordskjelv. (Fase 2+3)

Vi bygger to modeller og sammenligner dem:
  1. En referanse uten maskinlæring (to terskler, se src/features.py)
  2. Et lite CNN i Keras

Poenget med å ha begge er at tall fra et nevralt nett er meningsløse uten noe å måle dem
mot. Klarer referansen nesten det samme, bør vi bruke referansen — den er raskere,
forklarbar, og lar seg feilsøke når den tar feil.

TO METODEKRAV som er lette å bryte og som ødelegger resultatene hvis man gjør det:

  Splitt etter TID, ikke tilfeldig. Naboutsnitt i tid overlapper fysisk: den samme bilen
  opptrer i flere utsnitt etter hverandre. Med tilfeldig splitt havner samme bil både i
  trening og test, modellen kan gjenkjenne den i stedet for å generalisere, og vi får en
  glimrende testscore som ikke betyr noe. Vi legger i tillegg inn en KARANTENESONE
  mellom trenings- og testdelen, slik at en hendelse som krysser grensen ikke havner i
  begge.

  Normaliser hver KANAL for seg. Følsomheten varierer langs fiberen — i de ekte
  SAFOD-dataene så vi nesten døde kanaler som loddrette striper. Uten normalisering
  lærer modellen hvilke kanalnumre som er støyende i stedet for hvordan hendelser ser ut.
"""

from __future__ import annotations

import numpy as np

from .features import (kanalstatistikk, normaliser_med_statistikk,
                       referanse_klassifiser, tilpass_referanse)
from .synthetic_das import (BAKGRUNN, BIL, JORDSKJELV, NOE_ANNET, N_KLASSER,
                            KLASSENAVN, DASKonfig, generer_scene, tilfeldig_domene)

# Utsnittets størrelse: 64 kanaler (640 m vei) x 2 sekunder.
N_KANALER_UTSNITT = 64
VARIGHET_UTSNITT_S = 2.0


def lag_datasett(
    n_scener: int = 36,
    seed: int = 0,
    andel_terskel: float = 0.12,
    trening_til: float = 0.65,
    test_fra: float = 0.75,
):
    """
    Genererer scener og klipper dem opp i merkede utsnitt.

    `andel_terskel` styrer hvor stor del av et utsnitt som må være merket som en
    hendelse for at HELE utsnittet skal få den etiketten. Settes den for lavt, blir
    utsnitt der bilen så vidt berører hjørnet merket som "bil", og modellen får
    motstridende eksempler. Settes den for høyt, mister vi de tidlige delene av en
    hendelse — og for varsling er det nettopp de tidlige delene som er verdifulle.

    Tidsdelingen: utsnitt som starter før `trening_til` av opptaket går til trening,
    utsnitt som starter etter `test_fra` går til test. Gapet mellom er karantenesonen.
    """
    rng = np.random.default_rng(seed)
    konfig_mal = DASKonfig()
    n_tid_utsnitt = int(VARIGHET_UTSNITT_S * konfig_mal.samplingsrate_hz)

    X_tren, y_tren, X_test, y_test = [], [], [], []

    for i in range(n_scener):
        # Scenetypene fordeles fast, ikke tilfeldig, så vi er sikre på å få nok
        # eksempler av hver klasse. Med tilfeldig trekning kan en sjelden klasse
        # bli underrepresentert ved uflaks, og det oppdager man sent.
        med_skjelv = (i % 3 == 1)
        med_flom = (i % 3 == 2)

        # DOMENERANDOMISERING: hver scene får sine egne støyegenskaper. Dette er den
        # viktigste endringen i denne runden, og den kom av en målt feil — modellen
        # overtilpasset den ene støymodellen vår og feilet på ekte data.
        konfig = tilfeldig_domene(seed * 1000 + i)

        flom_kanaler = None
        if med_flom:
            # Flommen dekker en sammenhengende strekning av tilfeldig lengde og sted.
            bredde = int(rng.integers(40, 110))
            start = int(rng.integers(0, konfig.n_kanaler - bredde))
            flom_kanaler = (start, start + bredde)

        scene = generer_scene(
            konfig=konfig,
            n_biler=int(rng.integers(3, 9)),
            med_jordskjelv=med_skjelv,
            # Litt regn i noen scener, så modellen ser forhøyet støygulv som IKKE
            # er en hendelse i seg selv. Ellers lærer den at "mye energi = hendelse",
            # og regn blir en falsk alarm-maskin.
            med_regn=bool(rng.random() < 0.25),
            flom_kanaler=flom_kanaler,
            flom_styrke=float(rng.uniform(1.5, 4.0)),
        )

        n_kanaler, n_tid = scene.waterfall.shape

        # Normaliseringsgrunnlag per kanal, hentet fra en UAVHENGIG NORMALPERIODE på
        # samme fiber — ikke fra opptaket selv. Se Scene.kalibrering for hvorfor:
        # normaliserer vi mot et opptak som inneholder flommen, fjerner vi delvis det
        # vi skal oppdage. Dette speiler drift, der hver kanals normale nivå er kjent
        # fra historikk, og gir samtidig ingen lekkasje fra testperioden.
        kanal_median, kanal_skala = kanalstatistikk(scene.kalibrering)
        kanal_startpunkter = range(0, n_kanaler - N_KANALER_UTSNITT + 1, N_KANALER_UTSNITT)
        tid_startpunkter = range(0, n_tid - n_tid_utsnitt + 1, n_tid_utsnitt)

        for t0 in tid_startpunkter:
            andel_av_opptak = t0 / n_tid
            if andel_av_opptak < trening_til:
                maal_X, maal_y = X_tren, y_tren
            elif andel_av_opptak >= test_fra:
                maal_X, maal_y = X_test, y_test
            else:
                continue                      # karantenesone: kastes bevisst

            for k0 in kanal_startpunkter:
                bit = scene.waterfall[k0:k0 + N_KANALER_UTSNITT, t0:t0 + n_tid_utsnitt]
                merker = scene.etiketter[k0:k0 + N_KANALER_UTSNITT, t0:t0 + n_tid_utsnitt]

                # Prioritering når flere ting skjer i samme utsnitt. Rekkefølgen er
                # et bevisst valg etter hvor alvorlig hendelsen er for en trafikant:
                # et jordskjelv betyr mest, deretter noe unormalt på strekningen, og
                # sist en bil — som jo er det normale på en vei.
                andel_skjelv = float((merker == JORDSKJELV).mean())
                andel_annet = float((merker == NOE_ANNET).mean())
                andel_bil = float((merker == BIL).mean())
                if andel_skjelv > andel_terskel:
                    etikett = JORDSKJELV
                elif andel_annet > andel_terskel:
                    etikett = NOE_ANNET
                elif andel_bil > andel_terskel:
                    etikett = BIL
                else:
                    etikett = BAKGRUNN

                # Normaliser med kanalens langtidsstatistikk, ikke med utsnittets egen.
                bit_norm = normaliser_med_statistikk(
                    bit,
                    kanal_median[k0:k0 + N_KANALER_UTSNITT],
                    kanal_skala[k0:k0 + N_KANALER_UTSNITT],
                )
                maal_X.append(bit_norm)
                maal_y.append(etikett)

    return (np.array(X_tren), np.array(y_tren, dtype=np.int64),
            np.array(X_test), np.array(y_test, dtype=np.int64))


def klargjor_for_keras(X: np.ndarray) -> np.ndarray:
    """
    Siste steg før nettet: klipp ekstremverdier og legg til kanaldimensjon.

    Utsnittene er allerede normaliserte fra lag_datasett(). Klippingen er der fordi en
    enkelt instrumentglitch ellers kan dominere gradientene fullstendig — nøyaktig det
    problemet vi støtte på i de ekte dataene i fase 1, der én artefakt var 8000 ganger
    sterkere enn hendelsen vi var ute etter.
    """
    return np.clip(X, -20.0, 20.0)[..., None]


def bygg_cnn(inn_form: tuple[int, int, int], n_klasser: int = N_KLASSER):
    """
    Et lite konvolusjonsnett. Med rundt 20 000 parametre trener det på CPU i minutter.

    Hvorfor konvolusjon passer her: et CNN leter etter LOKALE MØNSTRE uavhengig av hvor
    i bildet de er. Det er akkurat det vi trenger — en bil ser lik ut enten den er ved
    kanal 10 eller kanal 50, og et jordskjelv ser likt ut uansett når i vinduet det
    kommer. Nettet kan dermed lære formene (skrå linje mot vannrett stripe) i stedet for
    posisjoner.

    Vi bruker steg (strides) i stedet for pooling-lag for å krympe bildet. Det gir færre
    lag og raskere trening, uten at vi mister noe vesentlig ved denne størrelsen.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    return keras.Sequential([
        layers.Input(shape=inn_form),
        # Første lag har et bredt filter i tidsretningen (5x7), fordi signaturene er
        # utstrakte i tid. Et 3x3-filter ville sett for lite om gangen.
        layers.Conv2D(16, (5, 7), strides=(2, 3), padding="same", use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),

        layers.Conv2D(32, (3, 5), strides=(2, 2), padding="same", use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),

        layers.Conv2D(48, (3, 3), strides=(2, 2), padding="same", use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),

        # Global pooling i stedet for Flatten: gjør at nettet oppsummerer "finnes dette
        # mønsteret noe sted i utsnittet" i stedet for "hvor i utsnittet er det".
        # Det gir langt færre parametre og mindre overtilpasning.
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(n_klasser),
    ])


def evaluer(y_sann: np.ndarray, y_pred: np.ndarray, navn: str) -> dict:
    """
    Regner ut presisjon og gjenkalling PER KLASSE, pluss forvirringsmatrise.

    Hvorfor ikke bare samlet nøyaktighet: klassene er svært ubalanserte — det meste av
    et opptak er bakgrunn. En modell som alltid svarer "bakgrunn" kan få 70 % nøyaktighet
    og likevel være fullstendig ubrukelig, siden den aldri finner en eneste hendelse.
    Samlet nøyaktighet skjuler dette. Presisjon og gjenkalling per klasse gjør det synlig.

      Presisjon  = av alt modellen kalte X, hvor mye VAR X?   (få falske alarmer)
      Gjenkalling = av alt som VAR X, hvor mye fant modellen?  (få tapte hendelser)

    For varsling er gjenkalling på hendelser viktigst — en tapt flom er verre enn en
    unødig sjekk — men presisjon avgjør om folk fortsetter å stole på systemet.
    """
    klasser = [BAKGRUNN, BIL, JORDSKJELV, NOE_ANNET]
    print(f"\n--- {navn} ---")
    print(f"{'Klasse':<12}{'Presisjon':>11}{'Gjenkalling':>13}{'F1':>8}{'Antall':>9}")

    resultat = {}
    for k in klasser:
        tp = int(np.sum((y_pred == k) & (y_sann == k)))
        fp = int(np.sum((y_pred == k) & (y_sann != k)))
        fn = int(np.sum((y_pred != k) & (y_sann == k)))
        presisjon = tp / (tp + fp) if tp + fp else 0.0
        gjenkalling = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * presisjon * gjenkalling / (presisjon + gjenkalling) if presisjon + gjenkalling else 0.0
        antall = int(np.sum(y_sann == k))
        resultat[KLASSENAVN[k]] = dict(presisjon=presisjon, gjenkalling=gjenkalling, f1=f1, antall=antall)
        print(f"{KLASSENAVN[k]:<12}{presisjon:>11.3f}{gjenkalling:>13.3f}{f1:>8.3f}{antall:>9d}")

    noyaktighet = float(np.mean(y_pred == y_sann))
    print(f"\nSamlet noyaktighet: {noyaktighet:.3f}  "
          f"(til sammenligning: alltid 'bakgrunn' gir {np.mean(y_sann == BAKGRUNN):.3f})")

    print("\nForvirringsmatrise (rad = fasit, kolonne = modellens svar):")
    print(f"{'':>12}" + "".join(f"{KLASSENAVN[k]:>12}" for k in klasser))
    for kr in klasser:
        rad = "".join(f"{int(np.sum((y_sann == kr) & (y_pred == kc))):>12d}" for kc in klasser)
        print(f"{KLASSENAVN[kr]:>12}" + rad)

    resultat["noyaktighet"] = noyaktighet
    return resultat



def test_pa_ekte_data(modell, terskler: tuple[float, float] | None = None):
    """
    Tester modellen på EKTE SAFOD-data, etter at den bare har sett syntetiske eksempler.

    Dette er den hardeste og mest interessante prøven i hele fase 2+3: overfører det
    nettet har lært fra vår egen modell av virkeligheten til virkeligheten selv?
    Begge utfall er verdifulle å vite, og begge rapporteres som de er.

    GEOMETRIEN MÅ TILPASSES FØRST, ellers er testen meningsløs. De ekte dataene har
    250 Hz og 1 m kanalavstand; de syntetiske har 100 Hz og 10 m. Et utsnitt på 64
    kanaler dekker dermed 64 m i ekte data mot 640 m i syntetiske, og 2 sekunder er 500
    sampler mot 200. Et nett som har lært hvordan en bølgefront HELLER i (kanal, tid)-
    planet ville sett en helt annen helning. Vi løser det ved å:
      - ta hver 10. kanal, så kanalavstanden blir 10 m som i treningen
      - nedsample tiden 250 Hz -> 100 Hz
    Etter dette har et utsnitt samme fysiske utstrekning i meter og sekunder som
    treningsdataene, og helningene er sammenlignbare.
    """
    from scipy import signal as sig
    from . import real_data

    print("\n=== Test pa EKTE data (SAFOD) ===")
    print("Modellen har KUN sett syntetiske eksempler under trening.")

    data = real_data.les()
    rent = real_data.fjern_glitcher(data)
    filtrert = real_data.bandpass(rent, 2.0, 40.0)

    # Tilpass geometrien til treningsdataene.
    kanal_steg = 10                     # 1 m -> 10 m kanalavstand
    redusert = filtrert[::kanal_steg]
    # 250 Hz -> 100 Hz: gang opp med 2, del ned med 5.
    nedsamplet = sig.resample_poly(redusert, up=2, down=5, axis=1).astype(np.float32)
    fs_ny = 100.0
    print(f"Tilpasset geometri: {nedsamplet.shape[0]} kanaler a 10 m, "
          f"{nedsamplet.shape[1]/fs_ny:.0f} s ved {fs_ny:.0f} Hz")

    # Normaliseringsgrunnlag fra en ROLIG periode (t = 35-55 s), altså etter skjelvet.
    # Dette speiler drift: kanalenes normale nivå er kjent fra historikk.
    i0, i1 = int(35 * fs_ny), int(55 * fs_ny)
    kanal_median, kanal_skala = kanalstatistikk(nedsamplet[:, i0:i1])

    n_tid_utsnitt = int(VARIGHET_UTSNITT_S * fs_ny)
    ankomst = real_data.STANDARD_ANKOMST_S

    utsnitt, fasit, tider = [], [], []
    for t0 in range(0, nedsamplet.shape[1] - n_tid_utsnitt + 1, n_tid_utsnitt):
        t_start = t0 / fs_ny
        # Hopp over randen, der filtertransienter forstyrrer.
        if t_start < 3.0 or t_start > 56.0:
            continue
        for k0 in range(0, nedsamplet.shape[0] - N_KANALER_UTSNITT + 1, 8):
            bit = nedsamplet[k0:k0 + N_KANALER_UTSNITT, t0:t0 + n_tid_utsnitt]
            bit_norm = normaliser_med_statistikk(
                bit, kanal_median[k0:k0 + N_KANALER_UTSNITT],
                kanal_skala[k0:k0 + N_KANALER_UTSNITT])
            utsnitt.append(bit_norm)
            # Jordskjelvet varer ca. 2 s fra ankomst. Utsnitt som overlapper regnes
            # som jordskjelv; alt annet er bakgrunn (SAFOD har ingen veitrafikk).
            overlapper = (t_start < ankomst + 2.0) and (t_start + VARIGHET_UTSNITT_S > ankomst)
            fasit.append(JORDSKJELV if overlapper else BAKGRUNN)
            tider.append(t_start)

    X = np.array(utsnitt)
    y = np.array(fasit)
    tider = np.array(tider)
    print(f"Utsnitt fra ekte data: {len(X)} "
          f"({int(np.sum(y == JORDSKJELV))} jordskjelv, {int(np.sum(y == BAKGRUNN))} bakgrunn)")

    y_cnn = np.argmax(modell.predict(klargjor_for_keras(X), verbose=0), axis=1)
    res_cnn = evaluer(y, y_cnn, "CNN pa EKTE data (trent kun pa syntetisk)")

    res_ref = None
    if terskler is not None:
        y_ref = np.array([referanse_klassifiser(x, *terskler) for x in X])
        res_ref = evaluer(y, y_ref, "Referanse pa EKTE data")

    # Hva svarte modellen nøyaktig i vinduene rundt skjelvet?
    naer = np.abs(tider - ankomst) < 4.0
    print("\nModellens svar i vinduene rundt jordskjelvet:")
    for t in np.unique(tider[naer]):
        m = tider == t
        svar = np.bincount(y_cnn[m], minlength=3)
        fasit_t = KLASSENAVN[int(y[m][0])]
        print(f"  t={t:5.1f}s  fasit={fasit_t:<10} "
              f"modell: bakgrunn={svar[0]}, bil={svar[1]}, jordskjelv={svar[2]}")

    return res_ref, res_cnn


if __name__ == "__main__":
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(0)
    np.random.seed(0)

    print("=== Bygger datasett (syntetisk) ===")
    X_tren, y_tren, X_test, y_test = lag_datasett(n_scener=60, seed=1)
    print(f"Trening: {X_tren.shape[0]} utsnitt, form {X_tren.shape[1:]}")
    print(f"Test:    {X_test.shape[0]} utsnitt  (delt etter TID, med karantenesone)")
    for navn, y in [("trening", y_tren), ("test", y_test)]:
        fordeling = {KLASSENAVN[k]: int(np.sum(y == k)) for k in range(N_KLASSER)}
        print(f"  Klassefordeling {navn}: {fordeling}")

    # --- Referanse uten maskinlæring ---------------------------------------
    print("\n=== Referanse: to terskler, ingen maskinlaering ===")
    # Tersklene tilpasses pa TRENINGSDATA, aldri pa testdata — samme regler som nettet.
    e_terskel, k_terskel = tilpass_referanse(X_tren, y_tren)
    y_ref = np.array([referanse_klassifiser(x, e_terskel, k_terskel) for x in X_test])
    res_ref = evaluer(y_test, y_ref, "Referanse (energi + koherens)")

    # --- CNN ---------------------------------------------------------------
    print("\n=== CNN ===")
    Xn_tren = klargjor_for_keras(X_tren)
    Xn_test = klargjor_for_keras(X_test)

    modell = bygg_cnn(Xn_tren.shape[1:])
    modell.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    print(f"Antall parametre: {modell.count_params():,}")

    # Klassevekter: siden bakgrunn dominerer, ville nettet ellers lære at det lønner seg
    # å gjette bakgrunn hele tiden. Vektene gjør sjeldne klasser dyrere å bomme på.
    antall = np.bincount(y_tren, minlength=N_KLASSER).astype(float)
    vekter = {k: float(len(y_tren) / (N_KLASSER * max(antall[k], 1))) for k in range(N_KLASSER)}
    print(f"Klassevekter: { {KLASSENAVN[k]: round(v, 2) for k, v in vekter.items()} }")

    # Tidlig stopp med tilbakestilling til beste vekter. Uten dette risikerer vi å
    # ende på en tilfeldig dårlig epoke — forrige runde svingte valideringstapet
    # kraftig mot slutten, og siste epoke er ikke nødvendigvis den beste modellen.
    # Tålmodigheten er satt høyt (18) med vilje. Med domenerandomisering er oppgaven
    # vesentlig vanskeligere, og valideringstapet forbedrer seg langsomt og ujevnt —
    # med tålmodighet 8 stoppet treningen på epoke 4 og modellen ble klart dårligere.
    stopp = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=18, restore_best_weights=True, verbose=1)
    senk_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5, verbose=0)

    modell.fit(
        Xn_tren, y_tren,
        validation_split=0.15,
        epochs=120, batch_size=64,
        class_weight=vekter,
        callbacks=[stopp, senk_lr],
        verbose=2,
    )

    y_cnn = np.argmax(modell.predict(Xn_test, verbose=0), axis=1)
    res_cnn = evaluer(y_test, y_cnn, "CNN")

    # --- Sammenligning -----------------------------------------------------
    print("\n=== Lonner CNN-et seg? ===")
    print(f"{'Klasse':<12}{'Referanse F1':>15}{'CNN F1':>10}{'Endring':>10}")
    for navn in ("bakgrunn", "bil", "jordskjelv", "noe annet"):
        a, b = res_ref[navn]["f1"], res_cnn[navn]["f1"]
        print(f"{navn:<12}{a:>15.3f}{b:>10.3f}{b - a:>+10.3f}")
    print(f"{'noyaktighet':<12}{res_ref['noyaktighet']:>15.3f}"
          f"{res_cnn['noyaktighet']:>10.3f}{res_cnn['noyaktighet'] - res_ref['noyaktighet']:>+10.3f}")

    # Den hardeste prøven: ekte data, uten å ha sett ett eneste ekte eksempel.
    ekte_ref, ekte_cnn = test_pa_ekte_data(modell, terskler=(e_terskel, k_terskel))

    # Lagre resultatene så figurkoden kan lese dem uten å trene på nytt.
    import json
    with open("data/processed/resultater_fase23.json", "w") as f:
        json.dump({
            "syntetisk": {"referanse": res_ref, "cnn": res_cnn},
            "ekte": {"referanse": ekte_ref, "cnn": ekte_cnn},
            "terskler": {"energi": e_terskel, "koherens": k_terskel},
            "n_parametre": int(modell.count_params()),
        }, f, indent=2)
    print("\nResultater lagret: data/processed/resultater_fase23.json")

    modell.save("data/processed/cnn_klassifikator.keras")
    print("\nModell lagret: data/processed/cnn_klassifikator.keras")
