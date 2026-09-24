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


if __name__ == "__main__":
    fig1_waterfall_sammenligning()
    fig1b_glitch_vs_skjelv()
