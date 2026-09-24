"""
Felles plottefunksjoner for waterfall-figurer. Alle tekster på norsk.

Et sentralt prinsipp i hele prosjektet: hver figur merkes tydelig [SYNTETISK] eller
[EKTE DATA]. Merkingen er lagt inn i selve plottefunksjonen, ikke overlatt til den som
lager figuren, slik at det ikke er mulig å glemme den.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # tegner til fil uten å åpne vindu — trengs for skript
import matplotlib.pyplot as plt
import numpy as np

FIGURMAPPE = Path(__file__).resolve().parent.parent / "figures"

# Fargekoder for merkingen, så skillet er synlig på én tiendedels sekund fra salen.
FARGE_SYNTETISK = "#b45309"     # rustoransje
FARGE_EKTE = "#15803d"          # grønn


def waterfall(
    ax,
    data: np.ndarray,
    samplingsrate_hz: float,
    kanalavstand_m: float,
    tittel: str = "",
    ekte_data: bool = False,
    persentil: float = 99.0,
    grense: float | None = None,
    start_s: float = 0.0,
    vis_ylabel: bool = True,
):
    """
    Tegner et waterfall-plott.

    ORIENTERING — dette er viktig å få riktig, og lett å snuble i:
      Posisjon langs fiberen går BORTOVER (x-aksen).
      Tid går NEDOVER (y-aksen), slik at nyere data kommer lenger ned.

    Det er denne konvensjonen som gir signaturene sine kjente former, og som gjør navnet
    "waterfall" meningsfullt — dataene renner nedover skjermen etter hvert som tiden går:
      - En bil er ett sted om gangen og flytter seg. Det gir en SKRÅ LINJE, og
        stigningen tilsvarer farten.
      - Et jordskjelv treffer hele fiberen nærmest samtidig, fordi bølgen går rundt
        4000 m/s mot bilens 22 m/s. Det gir en nesten VANNRETT STRIPE.
      - En flom rammer et avgrenset kanalintervall over tid. Det gir en LODDRETT SØYLE
        av forhøyet støy på en bestemt del av veien.
    Tre fenomener, tre helt ulike geometriske former. Det er derfor vi kan skille dem.

    Siden matrisene våre er lagret som (kanal, tid), transponerer vi her ved plotting.

    Om fargeskalaen: DAS-verdier spenner over mange størrelsesordener. Skalerte vi etter
    minimum og maksimum, ville noen få ekstremverdier gjort hele bildet blekt. Vi klipper
    derfor ved en persentil. For data med et kraftig innslag (som et ekte jordskjelv,
    der toppen ligger sju størrelsesordener over støygulvet) havner selv 99-persentilen
    inne i selve hendelsen, og bakgrunnen blir helt hvit. Da må `grense` settes manuelt
    ut fra et rolig parti av opptaket. Dette er et VISUELT valg og endrer ikke dataene.

    Fargekartet er divergerende (rødt-hvitt-blått) fordi DAS måler tøyning, som har
    fortegn. Hvitt betyr "ingen bevegelse", rødt og blått er bevegelse hver sin vei.
    """
    if grense is None:
        grense = float(np.percentile(np.abs(data), persentil))
    if grense <= 0:
        grense = float(np.abs(data).max()) or 1.0

    varighet_s = data.shape[1] / samplingsrate_hz
    lengde_m = data.shape[0] * kanalavstand_m

    bilde = ax.imshow(
        data.T,                       # (kanal, tid) -> (tid, kanal)
        aspect="auto",
        cmap="RdBu_r",
        vmin=-grense,
        vmax=grense,
        # extent: [venstre, hoyre, bunn, topp]. Tiden oker nedover, derfor
        # bunn = slutt-tid og topp = start-tid.
        extent=[0, lengde_m, start_s + varighet_s, start_s],
        interpolation="nearest",
    )

    ax.set_xlabel("Posisjon langs fiber (m)", fontsize=10)
    if vis_ylabel:
        ax.set_ylabel("Tid (s)  \u2192 nedover", fontsize=10)

    if tittel:
        ax.set_title(tittel, fontsize=11, pad=8)

    # Merkingen: ekte eller syntetisk. Plasseres inne i plottet sa den folger med
    # uansett hvordan figuren beskjaeres eller limes inn i en slide.
    merke = "EKTE DATA" if ekte_data else "SYNTETISK"
    farge = FARGE_EKTE if ekte_data else FARGE_SYNTETISK
    ax.text(
        0.015, 0.975, merke,
        transform=ax.transAxes,
        fontsize=8, fontweight="bold", color="white",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=farge, edgecolor="none", alpha=0.95),
    )
    return bilde


def lagre(fig, filnavn: str) -> Path:
    """Lagrer figuren til figures/ med høy nok oppløsning til en projektor."""
    FIGURMAPPE.mkdir(parents=True, exist_ok=True)
    sti = FIGURMAPPE / filnavn
    fig.savefig(sti, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Lagret: {sti}")
    return sti
