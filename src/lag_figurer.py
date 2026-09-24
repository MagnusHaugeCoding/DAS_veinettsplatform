"""
Lager pitch-figurene. Kjøres med:  python3 -m src.lag_figurer

Hver figur er nummerert etter hvilken fase den hører til, og navngitt slik at
rekkefølgen i figures/ er den samme som rekkefølgen i pitchen.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import plotting, real_data
from .synthetic_das import DASKonfig, generer_scene


def fig1_waterfall_sammenligning():
    """
    Figur 1: syntetiske og ekte DAS-data side om side.

    Poenget med figuren er å vise at de to signaturene vi bryr oss om ser ulike ut, og
    at de er lette å kjenne igjen med øyet:
      - Biler blir SKRÅ linjer. Stigningen er farten.
      - Jordskjelv blir en nesten VANNRETT stripe, fordi bølgen går tusen ganger
        raskere enn en bil og treffer hele fiberen nærmest samtidig.
    Panel 3 viser at den vannrette stripen også finnes i ekte, offentlige data.
    """
    fig, akser = plt.subplots(1, 3, figsize=(15, 6.0))

    # --- Panel 1: syntetisk normal trafikk ---------------------------------
    konfig = DASKonfig(seed=7)
    trafikk = generer_scene(konfig=konfig, n_biler=6, beskrivelse="normal trafikk")
    plotting.waterfall(
        akser[0], trafikk.waterfall,
        konfig.samplingsrate_hz, konfig.kanalavstand_m,
        tittel="Normal trafikk\nBiler = skrå linjer (stigning = fart)",
    )

    # --- Panel 2: syntetisk jordskjelv -------------------------------------
    konfig2 = DASKonfig(seed=11)
    skjelv = generer_scene(konfig=konfig2, n_biler=3, med_jordskjelv=True,
                           beskrivelse="jordskjelv")
    plotting.waterfall(
        akser[1], skjelv.waterfall,
        konfig2.samplingsrate_hz, konfig2.kanalavstand_m,
        tittel="Jordskjelv + trafikk\nJordskjelv = nesten vannrett stripe",
        vis_ylabel=False,
    )

    # --- Panel 3: EKTE jordskjelv fra SAFOD --------------------------------
    data = real_data.les()
    # To steg, i denne rekkefølgen, og begge er nødvendige:
    #   1. fjern instrumentglitcher — uten dette dominerer en artefakt ved 28,8 s
    #      fullstendig, og man plotter feil hendelse (se fjern_glitcher()).
    #   2. båndpass — fjerner lavfrekvent drift så bølgetoget blir lesbart.
    rent = real_data.fjern_glitcher(data)
    filtrert = real_data.bandpass(rent, 2.0, 40.0)

    # Fargeskalaen settes fra et ROLIG parti (t = 40-50 s), godt etter skjelvet.
    # Regnet vi persentilen på vinduet med skjelvet i, ville skalaen blitt satt av
    # hendelsen selv og all bakgrunnstekstur forsvunnet i hvitt.
    # Faktoren 0.35 metter fargeskalaen bevisst, slik at det svake bølgetoget (bare
    # 3,2x støygulvet) blir synlig på en projektor. Et rent visuelt valg — dataene
    # er uendret.
    rolig = real_data.hent_utsnitt(filtrert, start_s=40.0, varighet_s=10.0)
    grense = float(np.percentile(np.abs(rolig), 99.0)) * 0.35

    start_s = real_data.STANDARD_ANKOMST_S - 2.0
    utsnitt = real_data.hent_utsnitt(filtrert, start_s=start_s, varighet_s=6.0)

    plotting.waterfall(
        akser[2], utsnitt,
        real_data.SAMPLINGSRATE_HZ, real_data.KANALAVSTAND_M,
        tittel="Ekte jordskjelv, SAFOD (M2.46, 11,7 km)\nSamme vannrette signatur",
        ekte_data=True,
        grense=grense,
        start_s=start_s,
        vis_ylabel=False,
    )

    fig.suptitle(
        "DAS-signaturer: hva vi leter etter i veinettet",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.04,
        "Panel 1-2: syntetiske data fra egen generator (src/synthetic_das.py).  "
        "Panel 3: ekte offentlige data fra SAFOD-arrayet (DASDetection), båndpassfiltrert 2-40 Hz.\n"
        "Aksene har ulik skala: de syntetiske panelene viser 3 km vei over 60 s, det ekte panelet 800 m fiber over 6 s. "
        "Ekte data er despiket (instrumentglitch ved 28,8 s fjernet) og båndpassfiltrert.",
        ha="center", fontsize=8.5, color="#444444",
    )

    plotting.lagre(fig, "fig1_waterfall_sammenligning.png")



def fig1b_glitch_vs_skjelv():
    """
    Figur 1b: det sterkeste utslaget er ikke hendelsen.

    Denne figuren dokumenterer et funn vi gjorde underveis, og den er tatt med fordi
    den sier noe vesentlig om hva et varslingssystem må kunne:

    Det kraftigste utslaget i opptaket (t = 28,80 s) er 25 000 ganger støygulvet — og
    det er en INSTRUMENTGLITCH, ikke et jordskjelv. Det ekte jordskjelvet (t = 9,56 s)
    er bare 3,2 ganger støygulvet, altså nær 8000 ganger svakere.

    Det som skiller dem er ikke styrke, men KOHERENS langs fiberen:
      - Glitchen treffer alle 800 kanaler på nøyaktig samme sample. Uendelig hastighet.
      - Skjelvet brer seg med ca. 4000 m/s, som er fysisk riktig for berggrunn.

    Konsekvensen for systemet vi foreslår: en ren amplitudeterskel ville slått ut på
    glitchen og oversett skjelvet. Det er en direkte begrunnelse for at vi bygger på
    mønsteret i (kanal, tid)-planet — ikke på hvor kraftig utslaget er.
    """
    from scipy import signal as sig

    data = real_data.les()
    rent = real_data.fjern_glitcher(data)
    fs = real_data.SAMPLINGSRATE_HZ

    fig, akser = plt.subplots(1, 2, figsize=(13, 5.5))

    # Venstre: glitchen, i ufiltrerte data.
    rolig_raa = real_data.hent_utsnitt(data, start_s=40.0, varighet_s=10.0)
    plotting.waterfall(
        akser[0], real_data.hent_utsnitt(data, start_s=28.3, varighet_s=1.2),
        fs, real_data.KANALAVSTAND_M,
        tittel="ARTEFAKT  t=28,80 s  —  25 000x støygulvet\nAlle 800 kanaler samtidig: uendelig hastighet",
        ekte_data=True,
        grense=float(np.percentile(np.abs(rolig_raa), 99.0)),
        start_s=28.3,
    )

    # Høyre: det ekte skjelvet, despiket og filtrert.
    filtrert = real_data.bandpass(rent, 2.0, 40.0)
    rolig = real_data.hent_utsnitt(filtrert, start_s=40.0, varighet_s=10.0)
    plotting.waterfall(
        akser[1], real_data.hent_utsnitt(filtrert, start_s=7.5, varighet_s=5.0),
        fs, real_data.KANALAVSTAND_M,
        tittel="EKTE JORDSKJELV  t=9,56 s  —  3,2x støygulvet\nBrer seg med ca. 4000 m/s: fysisk riktig",
        ekte_data=True,
        grense=float(np.percentile(np.abs(rolig), 99.0)) * 0.35,
        start_s=7.5,
        vis_ylabel=False,
    )

    fig.suptitle(
        "Det sterkeste utslaget er ikke hendelsen",
        fontsize=14, fontweight="bold", y=1.0,
    )
    fig.text(
        0.5, -0.05,
        "Samme opptak, SAFOD M2.46. Artefaktet er ca. 8000 ganger sterkere enn jordskjelvet, men er ikke fysikk.\n"
        "Det som skiller dem er koherens langs fiberen, ikke amplitude — derfor kan ikke et varslingssystem "
        "bygge på en ren styrketerskel.",
        ha="center", fontsize=8.5, color="#444444",
    )
    fig.tight_layout()
    plotting.lagre(fig, "fig1b_glitch_vs_skjelv.png")



def fig2_klassifikator():
    """
    Figur 2: resultatene fra fase 2+3.

    Historien figuren forteller, i to trinn:

    FØRSTE RUNDE feilet CNN-et fullstendig på ekte data (jordskjelv F1 0,00 — det kalte
    alle ekte jordskjelvutsnitt "bil"), mens den enkle toterskel-regelen traff perfekt.
    Nettet hadde lært teksturen i vår egen støymodell, ikke fysikken.

    ANDRE RUNDE, etter at vi målte hva som var galt og rettet det, overfører nettet:
      - vi fant og rettet en feil der kanalfølsomhet bare gjaldt støyen og ikke
        hendelsene, slik at "døde" kanaler fikk 100x for sterke hendelser
      - vi innførte domenerandomisering: spektralhelning, kanalfølsomhet, døde kanaler
        og jordskjelvfrekvens varierer nå bredt mellom scener, så nettet ikke KAN lene
        seg på hvordan støyen ser ut
      - vi normaliserer mot en uavhengig normalperiode i stedet for mot opptaket selv

    Lærdommen er generell nok til å ta med i pitchen: når en modell ikke overfører, er
    det som regel treningsdataene som må fikses, ikke modellen.

    Krever at `python3 -m src.classifier` er kjørt først (lagrer resultat-JSON).
    """
    import json
    from .features import estimer_fart
    from .synthetic_das import DASKonfig, farget_stoy, legg_til_bil

    sti = plotting.FIGURMAPPE.parent / "data" / "processed" / "resultater_fase23.json"
    if not sti.exists():
        print(f"Mangler {sti} — kjor 'python3 -m src.classifier' forst.")
        return
    res = json.loads(sti.read_text())

    klasser = ["bakgrunn", "bil", "jordskjelv", "noe annet"]
    fig, akser = plt.subplots(1, 3, figsize=(16, 5.0))

    def stolper(ax, data_ref, data_cnn, tittel, undertekst, legend_loc="upper left"):
        x = np.arange(len(klasser))
        b = 0.36
        f_ref = [data_ref[k]["f1"] if data_ref else 0.0 for k in klasser]
        f_cnn = [data_cnn[k]["f1"] for k in klasser]
        ax.bar(x - b/2, f_ref, b, label="Referanse (2 terskler)", color="#0369a1")
        ax.bar(x + b/2, f_cnn, b, label="CNN (22 595 parametre)", color="#b45309")
        for xi, (a, c) in enumerate(zip(f_ref, f_cnn)):
            ax.text(xi - b/2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
            ax.text(xi + b/2, c + 0.02, f"{c:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(klasser)
        ax.set_ylim(0, 1.15); ax.set_ylabel("F1-score")
        ax.set_title(tittel, fontsize=11)
        ax.text(0.5, -0.19, undertekst, transform=ax.transAxes, ha="center",
                fontsize=8.5, color="#444444")
        ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc=legend_loc)

        # Marker klasser som ikke har noen eksempler i fasiten. Uten dette ser en
        # nullstolpe ut som om modellene feiler, mens den i virkeligheten betyr at
        # klassen ikke forekommer i datasettet i det hele tatt.
        for xi, k in enumerate(klasser):
            antall = data_cnn[k]["antall"]
            if antall == 0:
                ax.text(xi, 0.06, "ingen\neksempler", ha="center", fontsize=7.5,
                        color="#6b7280", style="italic")

    stolper(akser[0], res["syntetisk"]["referanse"], res["syntetisk"]["cnn"],
            "SYNTETISKE testdata\nCNN vinner, saerlig pa jordskjelv og 'noe annet'",
            "1680 utsnitt, delt etter tid med karantenesone.\n"
            "Referansen har ingen regel for 'noe annet' og far derfor 0 der.")
    stolper(akser[1], res["ekte"]["referanse"], res["ekte"]["cnn"],
            "EKTE data (SAFOD)\nBegge treffer etter at generatoren ble rettet",
            "81 utsnitt fra ett jordskjelv. SAFOD er ikke en vei, sa klassene 'bil' og\n"
            "'noe annet' finnes ikke her. Lite utvalg (6 jordskjelvutsnitt): tolk med forsiktighet.",
            legend_loc="center left")

    # Panel 3: fartsestimat mot fasit.
    sanne, estimerte, skarpheter = [], [], []
    for i, fart in enumerate([30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]):
        for retning in (1, -1):
            konfig = DASKonfig(seed=300 + i * 2 + (retning > 0))
            rng = np.random.default_rng(konfig.seed)
            wf = farget_stoy(konfig, rng)
            et = np.zeros(wf.shape, dtype=np.int8)
            legg_til_bil(wf, et, konfig, rng, fart_kmt=fart,
                         start_m=0.0 if retning > 0 else konfig.veilengde_m,
                         start_s=0.0, tyngde=1.0, retning=retning)
            est, skarp = estimer_fart(wf, konfig.kanalavstand_m, konfig.samplingsrate_hz)
            sanne.append(fart * retning); estimerte.append(est); skarpheter.append(skarp)

    sanne = np.array(sanne); estimerte = np.array(estimerte); skarpheter = np.array(skarpheter)
    palitelig = skarpheter > 0.3

    ax = akser[2]
    ax.plot([-140, 140], [-140, 140], "k--", lw=1, alpha=0.5, label="perfekt estimat")
    ax.scatter(sanne[palitelig], estimerte[palitelig], s=30, color="#15803d",
               label=f"skarphet > 0.3  (n={palitelig.sum()})", zorder=3)
    ax.scatter(sanne[~palitelig], estimerte[~palitelig], s=30, color="#dc2626",
               marker="x", label=f"lav skarphet, forkastes  (n={(~palitelig).sum()})", zorder=3)
    ax.set_xlabel("Sann fart (km/t)"); ax.set_ylabel("Estimert fart (km/t)")
    avvik = np.abs(estimerte[palitelig] - sanne[palitelig])
    ax.set_title(f"Fartsestimat fra stigning\nMedianavvik {np.median(avvik):.1f} km/t", fontsize=11)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25)
    ax.text(0.5, -0.19, "Negativ fart = motsatt kjoreretning. Skarphet flagger upalitelige estimater selv.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#444444")

    for ax in akser[:2]:
        ax.text(0.985, 0.975, "SYNTETISK" if ax is akser[0] else "EKTE DATA",
                transform=ax.transAxes, fontsize=8, fontweight="bold", color="white",
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor=plotting.FARGE_SYNTETISK if ax is akser[0] else plotting.FARGE_EKTE,
                          edgecolor="none"))
    akser[2].text(0.985, 0.03, "SYNTETISK", transform=akser[2].transAxes, fontsize=8,
                  fontweight="bold", color="white", va="bottom", ha="right",
                  bbox=dict(boxstyle="round,pad=0.35", facecolor=plotting.FARGE_SYNTETISK,
                            edgecolor="none"))

    fig.suptitle("Fase 2+3: bil, jordskjelv og \"noe annet\"",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    plotting.lagre(fig, "fig2_klassifikator.png")



def fig3_deteksjonsdemo():
    """
    Figur 3: modellen i arbeid på hele scenarier.

    Dette er demonstrasjonsfiguren. Hvert panel viser et helt opptak som et
    waterfall-plott, med modellens vurdering lagt oppå som fargede ruter — én rute per
    utsnitt på 64 kanaler x 2 sekunder. Man ser altså BÅDE hva som skjer på strekningen
    OG hva systemet mener om det, i samme bilde.

    Poenget er ikke en perfekt modell, men å vise konseptet: fiberen deles i ruter, hver
    rute får en vurdering, og vurderingene kan fargelegges på et kart over veien.
    """
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from matplotlib.patches import Rectangle
    from tensorflow import keras
    from scipy import signal as sig

    from .classifier import (N_KANALER_UTSNITT, VARIGHET_UTSNITT_S, klargjor_for_keras)
    from .features import kanalstatistikk, normaliser_med_statistikk
    from .synthetic_das import (BAKGRUNN, BIL, JORDSKJELV, NOE_ANNET, KLASSENAVN,
                                generer_scene, tilfeldig_domene)

    modellsti = plotting.FIGURMAPPE.parent / "data" / "processed" / "cnn_klassifikator.keras"
    if not modellsti.exists():
        print("Mangler modell — kjor 'python3 -m src.classifier' forst.")
        return
    modell = keras.models.load_model(modellsti)

    # Farger per klasse. Bakgrunn tegnes ikke — bare det som skjer noe med.
    FARGE = {BIL: "#2563eb", JORDSKJELV: "#ea580c", NOE_ANNET: "#dc2626"}

    def vurder_scene(ax, data, kalibrering, fs, dx, start_s=0.0):
        """Deler opptaket i ruter, klassifiserer hver, og tegner resultatet oppa."""
        med, skala = kanalstatistikk(kalibrering)
        n_t = int(VARIGHET_UTSNITT_S * fs)
        n_kan, n_tid = data.shape

        bunke, bokser = [], []
        for k0 in range(0, n_kan - N_KANALER_UTSNITT + 1, N_KANALER_UTSNITT):
            for t0 in range(0, n_tid - n_t + 1, n_t):
                bit = data[k0:k0 + N_KANALER_UTSNITT, t0:t0 + n_t]
                bunke.append(normaliser_med_statistikk(
                    bit, med[k0:k0 + N_KANALER_UTSNITT], skala[k0:k0 + N_KANALER_UTSNITT]))
                bokser.append((k0, t0))

        logits = modell.predict(klargjor_for_keras(np.array(bunke)), verbose=0)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        sannsyn = e / e.sum(axis=1, keepdims=True)
        spadd = np.argmax(sannsyn, axis=1)

        tellere = {}
        for (k0, t0), kl, p in zip(bokser, spadd, sannsyn.max(axis=1)):
            if kl == BAKGRUNN:
                continue
            x = k0 * dx
            y = start_s + t0 / fs
            ax.add_patch(Rectangle(
                (x, y), N_KANALER_UTSNITT * dx, VARIGHET_UTSNITT_S,
                facecolor=FARGE[kl], alpha=0.30, edgecolor=FARGE[kl], linewidth=1.6,
                zorder=5))
            tellere[kl] = tellere.get(kl, 0) + 1
        return tellere, len(bokser)

    paneler = []

    # --- Scenario 1: normal trafikk ---
    konfig = tilfeldig_domene(4242)
    sc = generer_scene(konfig=konfig, n_biler=6, beskrivelse="trafikk")
    paneler.append(("Normal trafikk", sc.waterfall, sc.kalibrering,
                    konfig.samplingsrate_hz, konfig.kanalavstand_m, 0.0, False))

    # --- Scenario 2: jordskjelv under trafikk ---
    konfig = tilfeldig_domene(777)
    sc = generer_scene(konfig=konfig, n_biler=4, med_jordskjelv=True, beskrivelse="skjelv")
    paneler.append(("Jordskjelv + trafikk", sc.waterfall, sc.kalibrering,
                    konfig.samplingsrate_hz, konfig.kanalavstand_m, 0.0, False))

    # --- Scenario 3: noe unormalt pa en strekning ---
    konfig = tilfeldig_domene(31337)
    sc = generer_scene(konfig=konfig, n_biler=4, flom_kanaler=(110, 200),
                       flom_styrke=3.0, beskrivelse="flom")
    paneler.append(("Vann over veien\n(strekning 1100-2000 m)", sc.waterfall, sc.kalibrering,
                    konfig.samplingsrate_hz, konfig.kanalavstand_m, 0.0, False))

    # --- Scenario 4: EKTE data ---
    from . import real_data
    d = real_data.bandpass(real_data.fjern_glitcher(real_data.les()), 2.0, 40.0)
    red = sig.resample_poly(d[::10], up=2, down=5, axis=1).astype(np.float32)
    fs_e = 100.0
    kal = red[:, int(35 * fs_e):int(55 * fs_e)]
    utsnitt_e = red[:, int(4 * fs_e):int(20 * fs_e)]
    paneler.append(("EKTE jordskjelv, SAFOD\n(M2.46, 11,7 km)", utsnitt_e, kal,
                    fs_e, 10.0, 4.0, True))

    fig, akser = plt.subplots(1, 4, figsize=(19, 6.6))

    for ax, (tittel, data, kal, fs, dx, t0, er_ekte) in zip(akser, paneler):
        plotting.waterfall(
            ax, data, fs, dx, tittel="", ekte_data=er_ekte,
            # Fargeskalaen settes fra NORMALPERIODEN, ikke fra opptaket selv. Da ser
            # man umiddelbart hva som stikker seg ut i forhold til det vanlige.
            grense=float(np.percentile(np.abs(kal), 99.5)) * (0.45 if er_ekte else 0.9),
            start_s=t0, vis_ylabel=(ax is akser[0]))
        tellere, n_ruter = vurder_scene(ax, data, kal, fs, dx, start_s=t0)

        sammendrag = "  ".join(
            f"{KLASSENAVN[k]}: {v}" for k, v in sorted(tellere.items())) or "kun bakgrunn"
        # Sammendraget legges i TITTELEN, ikke under plottet, der det ellers kolliderer
        # med aksetittelen.
        ax.set_title(f"{tittel}\n{n_ruter} ruter vurdert  |  {sammendrag}",
                     fontsize=11, pad=8)

    # Felles forklaring pa fargene.
    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(facecolor=FARGE[k], alpha=0.45, edgecolor=FARGE[k], label=KLASSENAVN[k])
                 for k in (BIL, JORDSKJELV, NOE_ANNET)] +
                [Patch(facecolor="white", edgecolor="#9ca3af", label="bakgrunn (ingen markering)")],
        loc="lower center", ncol=4, fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle("Modellen i arbeid: hver rute er 64 kanaler x 2 sekunder, fargelagt etter modellens vurdering",
                 fontsize=14, fontweight="bold", y=0.985)
    fig.text(0.5, 0.075,
             "Panel 1-3 er syntetiske scenarier. Panel 4 er EKTE data fra SAFOD — modellen har aldri sett et ekte treningseksempel under trening.\n"
             "Klassen \"noe annet\" er bevisst vagt definert: systemet pastar ikke a vite at det er en flom, bare at strekningen avviker fra sin normaltilstand.",
             ha="center", fontsize=9.5, color="#4b5563")
    fig.tight_layout(rect=[0, 0.135, 1, 0.955])
    plotting.lagre(fig, "fig3_deteksjonsdemo.png")


if __name__ == "__main__":
    fig1_waterfall_sammenligning()
    fig1b_glitch_vs_skjelv()
    fig2_klassifikator()
    fig3_deteksjonsdemo()
