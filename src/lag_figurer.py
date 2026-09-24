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
    Figur 2: resultatene fra fase 2+3, og den viktigste innsikten i dem.

    Figuren er bygget rundt et funn som snudde konklusjonen vår: CNN-et vinner klart på
    syntetiske data, men FEILER på ekte data, mens den enkle toterskel-regelen overfører
    perfekt. Forklaringen er at referansen koder fysikk — hvor stor del av fiberen som
    rister samtidig — mens nettet har lært teksturen i vår egen støymodell.

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

    klasser = ["bakgrunn", "bil", "jordskjelv"]
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
            "SYNTETISKE testdata\nCNN vinner",
            "1344 utsnitt, delt etter tid med karantenesone")
    stolper(akser[1], res["ekte"]["referanse"], res["ekte"]["cnn"],
            "EKTE data (SAFOD)\nReferansen vinner — CNN overforer ikke",
            "81 utsnitt fra ett jordskjelv. SAFOD er ikke en vei, sa klassen 'bil' finnes ikke her.\n"
            "Lite utvalg (6 jordskjelvutsnitt): tolk med forsiktighet.",
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

    fig.suptitle("Fase 2+3: bil og jordskjelv — og hvorfor den enkle modellen vant til slutt",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    plotting.lagre(fig, "fig2_klassifikator.png")



if __name__ == "__main__":
    fig1_waterfall_sammenligning()
    fig1b_glitch_vs_skjelv()
    fig2_klassifikator()
