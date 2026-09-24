"""
EKTE DAS-data med EKTE BILER og fasit: CFOS, Klausner St., Tel Aviv.

Hvorfor dette datasettet er et vendepunkt for prosjektet:
  Til nå har vi ikke hatt én eneste ekte bil å trene eller teste på. SAFOD er et
  borehull, og Farmers Loop Road har bare 2 kanaler tatt opp kl. 03 om natta. Klassen
  "bil" var derfor bare vist på syntetiske data.

  CFOS er fiber under en bygate, med fasit laget av video: et kamera filmet samme
  strekning, YOLO fant kjøretøyene i videoen, og posisjonene ble overført til
  (kanal, tid)-planet. Fasiten er altså uavhengig av DAS-signalet selv.

Kilde: Zenodo 10.5281/zenodo.15869300 (CC-BY 4.0), artikkel: Scientific Reports 2025,
"A fiber-optic traffic monitoring network trained with video inputs".

DET SOM SKILLER DETTE FRA VÅR GENERATOR, og som må sies klart:
  - README-en i datasettet sier at signalet er nedsamplet fra 1 kHz til 30 Hz med
    LAVPASS VED 1 Hz. Det som er igjen er den kvasistatiske TØYNINGEN fra bilens vekt,
    ikke vibrasjonen. Vi målte det: nesten all energi ligger under 1 Hz.
    Generatoren vår modellerer derimot bilen som bredbåndet vibrasjon (støy formet av
    en innhylling). Det er to ulike fysiske signaturer av samme bil.
  - 40 kanaler. Artikkelen sier 1 m kanalavstand og at kameraet dekker ca. 80 m, så vi
    antar 2 m mellom kanalene i de publiserte filene. Det er en SLUTNING, ikke oppgitt.
  - Alt er fra én dag (07:57-17:00), ett sted. Forfatterne delte trening/test
    tilfeldig. Vi slår sammen alle filene og deler ETTER TID, som ellers i prosjektet.
  - Hver treningsfil finnes også speilet (lr/ud/both). Vi bruker bare originalene,
    ellers havner speilvendte kopier av samme vindu i både trening og test.

Fasiten er RGB-bilder: svart = støy, grønn = personbil, rød = buss/lastebil.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np

SAMPLINGSRATE_HZ = 30.0
KANALAVSTAND_M = 2.0            # antatt, se docstring
LAVPASS_HZ = 1.0

# Klasser i fasitbildene.
STOY, PERSONBIL, TUNGTRANSPORT = 0, 1, 2

_URL = ("https://zenodo.org/api/records/15869300/files/"
        "cfos-full-supervised-without-video.zip/content")
_DATA_MAPPE = Path(__file__).resolve().parent.parent / "data" / "raw" / "cfos"


def last_ned() -> Path:
    """Laster ned og pakker ut datasettet (1 GB) hvis det ikke allerede finnes."""
    rot = _DATA_MAPPE / "cfos-data"
    if rot.exists():
        return rot
    _DATA_MAPPE.mkdir(parents=True, exist_ok=True)
    zip_sti = _DATA_MAPPE / "full.zip"
    if not zip_sti.exists():
        print("Laster ned CFOS (1 GB) fra Zenodo ...")
        urllib.request.urlretrieve(_URL, zip_sti)
    with zipfile.ZipFile(zip_sti) as z:
        z.extractall(_DATA_MAPPE)
    return rot


def _les_fasit(sti: Path) -> np.ndarray:
    """RGB-fasit -> klassekart. Gul (både rød og grønn) regnes som tungtransport."""
    from PIL import Image

    rgb = np.asarray(Image.open(sti)).astype(np.int16)
    kart = np.full(rgb.shape[:2], STOY, dtype=np.int8)
    kart[rgb[..., 1] > 127] = PERSONBIL
    kart[rgb[..., 0] > 127] = TUNGTRANSPORT
    return kart


def _klokkeslett_s(filnavn: str) -> int:
    """'das_075732-075802_.npy' -> sekunder etter midnatt for starten (07:57:32)."""
    hhmmss = filnavn.split("_")[1].split("-")[0]
    return int(hhmmss[:2]) * 3600 + int(hhmmss[2:4]) * 60 + int(hhmmss[4:])


def les() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returnerer alle ORIGINALE 30-sekunders vinduer, sortert etter klokkeslett:
      X      (n, 40, 900)  DAS-signal, (kanal, tid) ved 30 Hz
      fasit  (n, 40, 900)  klassekart: STOY / PERSONBIL / TUNGTRANSPORT
      start  (n,)          starttid i sekunder etter midnatt
    """
    rot = last_ned()
    stier = sorted(rot.glob("*/das/das_*_.npy"), key=lambda p: _klokkeslett_s(p.name))

    X, fasit, start = [], [], []
    for sti in stier:
        fasit_sti = sti.parent.parent / "labels" / sti.name.replace("das_", "label_").replace(".npy", ".png")
        X.append(np.load(sti).astype(np.float32))
        fasit.append(_les_fasit(fasit_sti))
        start.append(_klokkeslett_s(sti.name))
    return np.stack(X), np.stack(fasit), np.array(start)


if __name__ == "__main__":
    # Kjør med:  python3 -m src.cfos_data
    X, fasit, start = les()
    timer = len(X) * 30 / 3600
    print("--- EKTE BYGATE MED VIDEOFASIT: CFOS, Tel Aviv ---")
    print(f"{len(X)} vinduer a 30 s = {timer:.1f} timer, form {X.shape[1:]} (kanal, tid) ved 30 Hz")
    print(f"Klokkeslett: {start.min() // 3600:02d}:{start.min() % 3600 // 60:02d} - "
          f"{start.max() // 3600:02d}:{start.max() % 3600 // 60:02d}")
    andel = np.bincount(fasit.ravel(), minlength=3) / fasit.size
    print(f"Andel piksler: stoy {andel[0]:.3f}, personbil {andel[1]:.3f}, tungtransport {andel[2]:.3f}")

    # Hvor ligger energien i frekvens? Bekrefter lavpasset ved 1 Hz.
    spekter = np.mean(np.abs(np.fft.rfft(X[::20], axis=2)) ** 2, axis=(0, 1))
    frekv = np.fft.rfftfreq(X.shape[2], d=1 / SAMPLINGSRATE_HZ)
    print(f"Andel energi under 1 Hz: {spekter[frekv < 1.0].sum() / spekter.sum():.3f}")
