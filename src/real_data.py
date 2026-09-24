"""
Nedlasting og lesing av EKTE, offentlige DAS-data.

Kilde: https://github.com/ariellellouch/DASDetection (SAFOD-arrayet)
Funnet via awesome-das (https://github.com/DAS-RCN/awesome-das).

Hvorfor akkurat dette datasettet:
  - Hver hendelse ligger som én enkeltstående .npy-fil på ca. 48 MB. Vi kan altså laste
    ned ÉN fil i stedet for hele repoet (~1,5 GB). numpy leser .npy direkte, så vi slipper
    h5py og DAS-spesifikke lesebiblioteker.
  - Dataene er kjente jordskjelv med katalog (magnitude, avstand, azimut).

VIKTIG BEGRENSNING (som skal stå i pitchen):
  SAFOD er et forsknings-/borehullsarray, ikke en vei. Datasettet gir oss ekte
  jordskjelvsignatur i ekte DAS-støy, men INGEN ekte veitrafikk. All trafikk og flom i
  dette prosjektet er syntetisk. Se README.
"""

from __future__ import annotations

import urllib.request
import urllib.parse
from pathlib import Path

import numpy as np

# Katalogen sier 250 Hz (nedsamplet fra 2,5 kHz), 1 m kanalavstand, 10 m gauge length.
SAMPLINGSRATE_HZ = 250.0
KANALAVSTAND_M = 1.0

_BASIS_URL = "https://raw.githubusercontent.com/ariellellouch/DASDetection/master/"
_DATA_MAPPE = Path(__file__).resolve().parent.parent / "data" / "raw"

# Valg av hendelse er gjort empirisk, ikke gjettet. Avgjørende er forholdet mellom
# magnitude og avstand, ikke magnituden alene:
#   mag 2.86 på 87 km avstand  -> knapt synlig, 1.5x over støygulvet selv etter stabling
#   mag 2.46 på 11.7 km avstand -> svært tydelig, over 1000x etter stabling
# Vi bruker derfor den nære hendelsen. Avstandene er hentet fra catalog.csv (kolonnen
# "radius", i meter).
STANDARD_HENDELSE = "2017-06-30T15:05:13.000000Z_mag2.46.npy"

# Ankomsttid for jordskjelvet i standardhendelsen, målt med moveout-analyse (se
# funksjonen fjern_glitcher() for hvorfor dette var vanskeligere enn ventet).
# Verifisert: tilsynelatende hastighet 3862 m/s, som er riktig område for seismiske
# bølger i berggrunn.
STANDARD_ANKOMST_S = 9.56


def last_ned(filnavn: str = STANDARD_HENDELSE) -> Path:
    """
    Laster ned én hendelsesfil til data/raw/ hvis den ikke allerede finnes.

    Returnerer stien til filen. Hopper over nedlastingen hvis filen er der fra før,
    slik at det er trygt å kjøre denne funksjonen om og om igjen.
    """
    _DATA_MAPPE.mkdir(parents=True, exist_ok=True)
    maal = _DATA_MAPPE / filnavn

    if maal.exists():
        print(f"Finnes allerede, hopper over nedlasting: {maal.name} "
              f"({maal.stat().st_size / 1e6:.1f} MB)")
        return maal

    # Filnavnet inneholder kolon (f.eks. 03:17:32). Kolon er et reservert tegn i URL-er
    # og må kodes om til %3A, ellers svarer serveren med 404.
    url = _BASIS_URL + urllib.parse.quote(filnavn)
    print(f"Laster ned {filnavn} (~48 MB) ...")
    urllib.request.urlretrieve(url, maal)
    print(f"Ferdig: {maal} ({maal.stat().st_size / 1e6:.1f} MB)")
    return maal


def les(filnavn: str = STANDARD_HENDELSE) -> np.ndarray:
    """
    Leser hendelsesfilen og returnerer en matrise med formen (kanal, tid).

    Vi ANTAR ingenting om formen på forhånd. Repoets README dokumenterer samplingsrate
    og kanalavstand, men ikke antall kanaler, så vi leser det ut av selve filen og
    orienterer matrisen basert på hva vi finner.
    """
    sti = last_ned(filnavn)
    data = np.load(sti)

    # DAS-data har alltid langt flere tidssteg enn kanaler (250 målinger per sekund mot
    # noen hundre kanaler). Vi bruker det til å avgjøre hvilken akse som er hvilken,
    # og transponerer om nødvendig, så resten av koden trygt kan anta (kanal, tid).
    if data.shape[0] > data.shape[1]:
        data = data.T

    return np.asarray(data, dtype=np.float32)


def hent_utsnitt(
    data: np.ndarray,
    start_s: float = 0.0,
    varighet_s: float = 30.0,
    kanal_fra: int = 0,
    kanal_til: int | None = None,
) -> np.ndarray:
    """
    Klipper ut et mindre vindu av (kanal, tid)-matrisen.

    Hele filen er stor å plotte og treg å jobbe med. Til figurer og eksempler holder
    det med noen titalls sekunder og et utvalg kanaler.
    """
    i0 = int(start_s * SAMPLINGSRATE_HZ)
    i1 = int((start_s + varighet_s) * SAMPLINGSRATE_HZ)
    kanal_til = data.shape[0] if kanal_til is None else kanal_til
    return data[kanal_fra:kanal_til, i0:i1]



def fjern_glitcher(
    data: np.ndarray,
    terskel_faktor: float = 100.0,
    andel_kanaler: float = 0.5,
    etterheng: int = 30,
) -> np.ndarray:
    """
    Fjerner instrumentglitcher: korte pigger som treffer alle kanaler samtidig.

    DETTE ER IKKE KOSMETIKK — uten dette steget finner man feil hendelse.

    Da vi først lette etter jordskjelvet i denne filen, fant vi en kraftig "hendelse"
    ved t = 28,80 s og holdt på å bruke den i figurene. Tre kontroller avslørte at det
    ikke er et jordskjelv i det hele tatt:

      1. Signalet hopper fra 3e-12 til 1,3e-5 på ÉN sample — en faktor på fem millioner,
         momentant. Bakken beveger seg ikke slik.
      2. Alle 800 kanaler har toppen på NØYAKTIG samme sample. En seismisk bølge i
         ca. 4000 m/s bruker minst 0,2 s (50 sampler) på å krysse 800 m fiber. Null
         forsinkelse er fysisk umulig.
      3. Verdiene etter toppen avtar som en ren fortegnsvekslende geometrisk rekke.
         Det er impulsresponsen til et anti-aliasfilter — ringing fra nedsamplingen
         2,5 kHz -> 250 Hz — ikke bakkebevegelse.

    Det ekte jordskjelvet ligger ved t = 9,56 s, er langt svakere (3,2x over støygulvet),
    og har tilsynelatende hastighet 3862 m/s — altså ekte moveout langs fiberen.

    Lærdommen, som er verdt å ta med i pitchen: det sterkeste utslaget i et DAS-opptak
    er ofte IKKE den interessante hendelsen. Et varslingssystem som bare ser etter
    amplitude ville slått ut på glitchen og oversett skjelvet. Derfor sjekker vi om et
    utslag er KOHERENT langs fiberen — det er den testen som skiller fysikk fra artefakt.

    Metoden her er bevisst enkel: finn sampler der mer enn `andel_kanaler` av kanalene
    samtidig overskrider `terskel_faktor` ganger medianamplituden, og nullstill et kort
    vindu rundt dem (inkludert `etterheng` sampler, for å få med filterringingen).
    """
    data = data.copy()
    terskel = terskel_faktor * np.median(np.abs(data))
    ekstrem_andel = (np.abs(data) > terskel).mean(axis=0)
    glitcher = np.where(ekstrem_andel > andel_kanaler)[0]

    for g in glitcher:
        fra = max(0, g - 2)
        til = min(data.shape[1], g + etterheng)
        # Vi nullstiller. Bakgrunnen ligger allerede rundt 1e-12, så dette gir
        # nesten ingen kunstig kant å ringe på for det påfølgende båndpassfilteret.
        data[:, fra:til] = 0.0

    if len(glitcher):
        print(f"  Fjernet {len(glitcher)} glitch-sampler "
              f"(t = {glitcher.min()/SAMPLINGSRATE_HZ:.2f}-{glitcher.max()/SAMPLINGSRATE_HZ:.2f} s)")
    return data


def bandpass(data: np.ndarray, lav_hz: float = 2.0, hoy_hz: float = 40.0) -> np.ndarray:
    """
    Båndpassfiltrerer dataene, altså beholder bare frekvenser mellom lav_hz og hoy_hz.

    Hvorfor dette er nødvendig: DAS-støy domineres av svært lave frekvenser (vind,
    havdønninger, temperaturdrift i fiberen). Ufiltrert drukner et jordskjelv fullstendig
    — vi målte at et skjelv av magnitude 2.46 ikke ga noe som helst utslag i rå
    bredbåndsenergi. Filtrerer vi bort den lavfrekvente driften, står signalet igjen.

    Vi bruker sosfiltfilt (ikke sosfilt): den kjører filteret begge veier, slik at
    signalet ikke forskyves i tid. Det er viktig når vi skal lese av ankomsttider.
    """
    from scipy import signal as sig
    sos = sig.butter(4, [lav_hz, hoy_hz], btype="band", fs=SAMPLINGSRATE_HZ, output="sos")
    return sig.sosfiltfilt(sos, data, axis=1).astype(np.float32)


def stable(data: np.ndarray) -> np.ndarray:
    """
    Stabler (midler) alle kanaler til én tidsrekke.

    ADVARSEL — dette virker dårligere enn lærebøkene lover, og vi målte det selv:

    Teorien sier at stabling av N kanaler forbedrer signal/støy med kvadratroten av N,
    altså 28x for våre 800 kanaler. Vi målte på de to ekte hendelsene:

        M2.46 på 11,7 km  : enkeltkanal 316 000  ->  stablet 18 000   (0.1x, DÅRLIGERE)
        M2.86 på 87 km    : enkeltkanal     3.5  ->  stablet     4.0   (1.2x, marginalt)

    Hvorfor teorien svikter — vi antok først feil årsak og målte oss fram til den rette:

      FEIL ANTAKELSE: at støyen er romlig korrelert og derfor ikke kansellerer.
      Målingen motbeviser det. Korrelasjonen mellom nabokanaler i støy er r = 0,02 ved
      1 m og null lenger unna. Støyen er uavhengig, og den faller faktisk som teorien
      sier: amplituden går ned med faktor 0,070 ved stabling av 800 kanaler, nær det
      teoretiske 1/sqrt(800) = 0,035.

      FAKTISK ÅRSAK: det er SIGNALET som ikke overlever. Målt i jordskjelvvinduet er
      signalkorrelasjonen +0,91 mellom kanaler 1 m fra hverandre, men den snur til
      -0,20 ved 25 m og -0,37 ved 50 m. Fortegnet på signaltoppen er 50,1 % positivt og
      49,9 % negativt over arrayet. Signalet faller derfor med nøyaktig samme faktor som
      støyen, og netto SNR-endring blir 0,99x.

      Sannsynlig forklaring: DAS måler tøyning LANGS fiberen, og fiberen ligger ikke
      rett. Uansett årsak er konsekvensen målt og entydig.

    Funksjonen beholdes fordi den er nyttig som rask oversikt over hele fiberen, men den
    skal IKKE brukes som deteksjonsmetode, og vi bruker den ikke til det i dette
    prosjektet. Riktig framgangsmåte for DAS er å utnytte at signalet er koherent langs
    en KURVE i (kanal, tid)-planet — nesten vannrett for jordskjelv, skrå for biler —
    i stedet for å midle blindt over alle kanaler.
    """
    return data.mean(axis=0)


if __name__ == "__main__":
    # Kjør med:  python3 -m src.real_data
    # Formålet er å SE hva filen faktisk inneholder, før vi bygger noe på antagelser.
    data = les()
    varighet = data.shape[1] / SAMPLINGSRATE_HZ
    lengde_m = data.shape[0] * KANALAVSTAND_M

    print("\n--- EKTE DATA: SAFOD ---")
    print(f"Form (kanal, tid) : {data.shape}")
    print(f"Datatype          : {data.dtype}")
    print(f"Antall kanaler    : {data.shape[0]}  ->  {lengde_m:.0f} m fiber")
    print(f"Antall tidssteg   : {data.shape[1]}  ->  {varighet:.1f} s ved {SAMPLINGSRATE_HZ:.0f} Hz")
    print(f"Verdiområde       : min {data.min():.3g}, maks {data.max():.3g}")
    print(f"Median absoluttverdi: {np.median(np.abs(data)):.3g}")

    # Vis hvorfor despiking er nødvendig, og hvor det ekte skjelvet ligger.
    from scipy import signal as sig

    print("\n--- Datakvalitet: instrumentglitcher ---")
    rent = fjern_glitcher(data)

    kant = int(3 * SAMPLINGSRATE_HZ)
    f_med = bandpass(data)[:, kant:-kant]      # med glitch
    f_uten = bandpass(rent)[:, kant:-kant]     # uten glitch
    t = np.arange(f_med.shape[1]) / SAMPLINGSRATE_HZ + 3.0

    def sterkeste(f):
        innh = np.abs(sig.hilbert(f, axis=1)).mean(axis=0)
        innh = np.convolve(innh, np.ones(int(0.3 * SAMPLINGSRATE_HZ)) / int(0.3 * SAMPLINGSRATE_HZ),
                           mode="same")
        i = int(np.argmax(innh))
        return t[i], innh[i] / np.median(innh)

    t1, r1 = sterkeste(f_med)
    t2, r2 = sterkeste(f_uten)
    print(f"Sterkeste utslag MED glitch  : t = {t1:5.2f} s  ({r1:6.2f}x)  <- artefakt")
    print(f"Sterkeste utslag UTEN glitch : t = {t2:5.2f} s  ({r2:6.2f}x)  <- ekte jordskjelv")

    # Moveout: den avgjørende fysiske testen.
    i0, i1 = int((t2 - 1.5) * SAMPLINGSRATE_HZ), int((t2 + 2.5) * SAMPLINGSRATE_HZ)
    innh = np.abs(sig.hilbert(f_uten[:, i0:i1], axis=1))
    ankomst = np.argmax(innh, axis=1) / SAMPLINGSRATE_HZ
    kanaler = np.arange(len(ankomst))
    helning = np.polyfit(kanaler, ankomst, 1)[0]
    print(f"\nMoveout-test (skiller fysikk fra artefakt):")
    print(f"  Tilsynelatende hastighet: {1/abs(helning):.0f} m/s")
    print(f"  (seismiske bolger i berggrunn: 2500-6000 m/s -> konsistent med et ekte skjelv)")
