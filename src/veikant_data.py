"""
EKTE DAS-data fra en VEIKANT: Fairbanks Farmers Loop Road, Alaska.

Hvorfor dette datasettet er viktig for prosjektet:
  SAFOD-dataene vi ellers bruker kommer fra et forsknings-/borehullsarray. De gir ekte
  jordskjelvsignatur, men de sier ingenting om hvordan fiber LANGS EN VEI oppfører seg —
  og det er jo nettopp det systemet vårt skal stå i.

  Farmers Loop Road er en faktisk vei. Datasettet ligger i
  github.com/eileenrmartin/FiberOpticEarthquakes, og er så vidt vi kan se det eneste
  lett tilgjengelige veikant-DAS i hele awesome-das-listen. Det er publisert som del av
  en jordskjelvartikkel, ikke som et trafikkdatasett, og derfor er det svært begrenset.

BEGRENSNINGENE, som må sies klart:
  - Bare TO kanaler (nr. 27 og 75). Vi kan altså ikke lage waterfall-plott og ikke se
    den skrå linjen som er bilens signatur. Til det trengs mange nabokanaler.
  - 120 sekunder totalt.
  - Opptaket er 12:23 UTC = ca. 03:23 lokal tid i Alaska. Midt på natten på en landevei,
    så det er nesten ingen trafikk å se.

  Datasettet duger derfor IKKE til å trene eller teste en trafikkdetektor. Det vi bruker
  det til er noe annet og likevel verdifullt: å måle hvordan STØYEN i ekte veikantfiber
  faktisk ser ut, slik at vi kan vurdere hvor troverdig vår syntetiske støymodell er.

PubDAS inneholder FORESEE og andre veikant-array med rikelig trafikk, men ligger bak
Globus, som krever egen konto og klient. Det er en naturlig neste kilde hvis prosjektet
skal videre.
"""

from __future__ import annotations

import struct
import urllib.request
from pathlib import Path

import numpy as np

SAMPLINGSRATE_HZ = 1000.0
_BASIS_URL = ("https://raw.githubusercontent.com/eileenrmartin/"
              "FiberOpticEarthquakes/master/data/")
_DATA_MAPPE = Path(__file__).resolve().parent.parent / "data" / "raw"

FILER = [
    "DAS.2018.08.26.12.23.42.ch01.SAC",     # kanal 75
    "DAS.2018.08.26.12.23.42.ch02.SAC",     # kanal 27
]


def _les_sac(sti: Path) -> tuple[float, np.ndarray]:
    """
    Minimal SAC-leser. Vi skriver den selv i stedet for å installere obspy, siden
    formatet er enkelt og vi bare trenger to felt.

    SAC-headeren er 632 byte: 70 flyttall (byte 0-279), 40 heltall (280-439) og
    192 byte tekst (440-631). Deretter kommer måledataene som float32.
      DELTA (tid mellom sampler) er flyttall nr. 0, altså byte 0-3.
      NPTS  (antall sampler)     er heltall nr. 9, altså byte 316-319.
    """
    raw = sti.read_bytes()
    delta = struct.unpack("<f", raw[0:4])[0]
    npts = struct.unpack("<i", raw[316:320])[0]
    data = np.frombuffer(raw[632:632 + npts * 4], dtype="<f4").astype(np.float32)
    return float(delta), data


def last_ned() -> list[Path]:
    """Laster ned de to sporene (480 kB hver) hvis de ikke allerede finnes."""
    _DATA_MAPPE.mkdir(parents=True, exist_ok=True)
    stier = []
    for filnavn in FILER:
        maal = _DATA_MAPPE / filnavn
        if not maal.exists():
            print(f"Laster ned {filnavn} (0,5 MB) ...")
            urllib.request.urlretrieve(_BASIS_URL + filnavn, maal)
        stier.append(maal)
    return stier


def les(nedsample_til_hz: float | None = 100.0) -> np.ndarray:
    """
    Returnerer de to sporene som en (kanal, tid)-matrise.

    Standard nedsampling til 100 Hz gjør dataene direkte sammenlignbare med
    generatoren vår, som også kjører 100 Hz.
    """
    from scipy import signal as sig

    spor = []
    for sti in last_ned():
        _, x = _les_sac(sti)
        if nedsample_til_hz is not None:
            faktor = int(round(SAMPLINGSRATE_HZ / nedsample_til_hz))
            x = sig.resample_poly(x, up=1, down=faktor)
        spor.append(x.astype(np.float32))
    return np.stack(spor)


if __name__ == "__main__":
    # Kjør med:  python3 -m src.veikant_data
    from scipy import stats

    data = les(nedsample_til_hz=100.0)
    print("--- EKTE VEIKANTFIBER: Fairbanks Farmers Loop Road ---")
    print(f"Form (kanal, tid): {data.shape}  ->  {data.shape[1] / 100.0:.0f} s ved 100 Hz")
    print(f"Kun {data.shape[0]} kanaler — for fa til waterfall-plott. Se modulens docstring.")

    for i, kanalnr in enumerate([75, 27]):
        x = data[i]
        n = 100
        e = np.array([x[j * n:(j + 1) * n].std() for j in range(len(x) // n)])
        med = np.median(e)
        forhoyet = np.where(e / med > 1.3)[0]
        print(f"\nKanal {kanalnr}: sekunder med forhoyet energi (>1.3x median): "
              f"{len(forhoyet)} av {len(e)}")
        if len(forhoyet):
            print(f"  ved t = {forhoyet.tolist()}")

    print("\nMerk: opptaket er 12:23 UTC = ca. 03:23 lokal tid. Nesten ingen trafikk.")
