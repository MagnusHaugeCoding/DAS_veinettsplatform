"""
Syntetisk DAS-generator for en veistrekning.  [SYNTETISK]

Hvorfor vi trenger denne:
  Det finnes ikke lett tilgjengelige offentlige DAS-datasett med veitrafikk. For å kunne
  vise at metoden virker, lager vi derfor data selv — med signaturer som etterligner de
  fysiske fenomenene vi vet DAS fanger opp. Fordelen er at vi kjenner fasiten nøyaktig
  (hvilken bil, hvilken fart, når flommen startet), noe vi aldri ville hatt med ekte data.
  Ulempen er at resultatene viser at metoden virker på signaler med de RETTE EGENSKAPENE
  — ikke at den er validert på ekte veidata. Det skillet skal stå tydelig overalt.

Datamodellen:
  All DAS-data er en matrise med to akser: kanal (posisjon langs fiberen) og tid.
  Vi holder oss konsekvent til formen (kanal, tid) i hele prosjektet.

Signaturene vi modellerer:
  Bil        - skrå linje. Stigningen tilsvarer farten.
  Jordskjelv - nesten vannrett stripe, med svak P-ankomst fulgt av sterkere S-ankomst.
  Vann/flom  - bredbåndet støy på et kanalintervall, som vokser over tid.
  Regn       - bredbåndet støy over HELE fiberen (skiller seg fra flom ved utstrekningen).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Klasseetiketter. Brukes av klassifikatoren i fase 2+3.
BAKGRUNN, BIL, JORDSKJELV = 0, 1, 2
KLASSENAVN = {BAKGRUNN: "bakgrunn", BIL: "bil", JORDSKJELV: "jordskjelv"}


@dataclass
class DASKonfig:
    """
    Oppsettet for en simulert veistrekning.

    Standardverdiene beskriver 3 km vei med 10 m mellom kanalene, målt i 60 sekunder.
    100 Hz er rikelig for trafikk (som holder seg under ~50 Hz) og gjør matrisene små
    nok til å trene på CPU. Ekte DAS-systemer kjører ofte 250-2500 Hz, men å beholde
    all den oppløsningen ville bare kostet oss regnetid uten å gi mer informasjon.
    """
    n_kanaler: int = 300
    kanalavstand_m: float = 10.0
    varighet_s: float = 60.0
    samplingsrate_hz: float = 100.0
    stoynivaa: float = 1.0
    seed: int | None = 42

    @property
    def n_tid(self) -> int:
        return int(self.varighet_s * self.samplingsrate_hz)

    @property
    def veilengde_m(self) -> float:
        return self.n_kanaler * self.kanalavstand_m

    @property
    def tidsakse(self) -> np.ndarray:
        return np.arange(self.n_tid) / self.samplingsrate_hz

    @property
    def kanalakse_m(self) -> np.ndarray:
        return np.arange(self.n_kanaler) * self.kanalavstand_m


# ---------------------------------------------------------------------------
# Bakgrunnsstøy
# ---------------------------------------------------------------------------

def farget_stoy(konfig: DASKonfig, rng: np.random.Generator, eksponent: float = 1.0) -> np.ndarray:
    """
    Lager bakgrunnsstøy med 1/f-karakter ("farget støy"), ikke hvit støy.

    Hvorfor dette er viktig: hvit støy har like mye energi på alle frekvenser. Ekte
    seismisk støy og DAS-støy har derimot klart mest energi på lave frekvenser (vind,
    havdønninger, fjern trafikk, temperaturdrift). Bruker vi hvit støy blir oppgaven
    kunstig lett for modellene våre, fordi ekte signaler også er lavfrekvente og dermed
    ville skilt seg ut altfor tydelig. Da ville resultatene vært verdiløse.

    Framgangsmåte: vi lager hvit støy, går til frekvensdomenet med FFT, demper høye
    frekvenser med 1/f, og går tilbake. I tillegg gir vi hver kanal sin egen følsomhet,
    fordi ekte fiber varierer langs kabelen (kobling mot bakken, skjøter, bøyer).
    """
    hvit = rng.standard_normal((konfig.n_kanaler, konfig.n_tid))

    # rfftfreq gir frekvensene FFT-en jobber med. Første element er 0 Hz, som vi må
    # unngå å dele på — vi setter den til den nest laveste frekvensen.
    frekvenser = np.fft.rfftfreq(konfig.n_tid, d=1.0 / konfig.samplingsrate_hz)
    frekvenser[0] = frekvenser[1] if len(frekvenser) > 1 else 1.0

    spektrum = np.fft.rfft(hvit, axis=1)
    spektrum *= 1.0 / frekvenser**eksponent          # demper høye frekvenser
    stoy = np.fft.irfft(spektrum, n=konfig.n_tid, axis=1)

    # Normaliser til standardavvik 1, så stoynivaa betyr det samme uansett eksponent.
    stoy /= stoy.std() + 1e-12

    # Per-kanal følsomhet: noen kanaler er systematisk mer støyende enn andre.
    kanalfolsomhet = rng.lognormal(mean=0.0, sigma=0.25, size=(konfig.n_kanaler, 1))
    return (stoy * kanalfolsomhet * konfig.stoynivaa).astype(np.float32)


# ---------------------------------------------------------------------------
# Hendelser
# ---------------------------------------------------------------------------

def legg_til_bil(
    waterfall: np.ndarray,
    etiketter: np.ndarray,
    konfig: DASKonfig,
    rng: np.random.Generator,
    fart_kmt: float,
    start_m: float,
    start_s: float,
    tyngde: float = 1.0,
    retning: int = 1,
) -> dict:
    """
    Legger inn én bil som passerer langs veien.

    Fysikken: bilen befinner seg i posisjon x(t) = start + fart * t. Vibrasjonen sprer
    seg noen titalls meter i bakken rundt kjøretøyet. I (kanal, tid)-matrisen blir det
    en SKRÅ LINJE, og stigningen er nettopp farten — det er dette som gjør at vi kan
    lese av fart direkte fra bildet.

    `tyngde` skalerer både amplituden og hvor langt vibrasjonen sprer seg: en lastebil
    rister bakken kraftigere og over et større område enn en personbil.
    """
    t = konfig.tidsakse
    fart_ms = fart_kmt / 3.6

    # Posisjonen til bilen ved hvert tidssteg, omregnet til kanalnummer.
    posisjon_m = start_m + retning * fart_ms * (t - start_s)
    senterkanal = posisjon_m / konfig.kanalavstand_m

    # Hvor bredt vibrasjonen merkes, i antall kanaler.
    spredning_m = 20.0 * (0.7 + 0.5 * tyngde)
    spredning_kanaler = spredning_m / konfig.kanalavstand_m

    kanaler = np.arange(konfig.n_kanaler)[:, None]      # (kanal, 1)
    avstand = kanaler - senterkanal[None, :]            # (kanal, tid)
    innhylling = np.exp(-0.5 * (avstand / spredning_kanaler) ** 2)

    # Bilen finnes bare fra det tidspunktet den kjører inn i utsnittet.
    innhylling[:, t < start_s] = 0.0

    # Selve vibrasjonen er støyaktig, ikke en ren tone. Vi former derfor støy med
    # innhyllingen: strukturen kommer fra innhyllingen, teksturen fra støyen.
    amplitude = 3.0 * tyngde
    signal = amplitude * innhylling * rng.standard_normal(waterfall.shape)

    waterfall += signal.astype(np.float32)

    # Merk hvor bilen faktisk er sterk nok til å regnes som en deteksjon.
    etiketter[innhylling > 0.4] = BIL

    return {
        "type": "bil",
        "fart_kmt": fart_kmt,
        "tyngde": tyngde,
        "retning": retning,
        "start_m": start_m,
        "start_s": start_s,
    }


def legg_til_jordskjelv(
    waterfall: np.ndarray,
    etiketter: np.ndarray,
    konfig: DASKonfig,
    rng: np.random.Generator,
    ankomst_s: float,
    amplitude: float = 8.0,
    tilsynelatende_hastighet_ms: float = 4000.0,
    sp_tid_s: float = 3.0,
) -> dict:
    """
    Legger inn et jordskjelv med P-ankomst fulgt av S-ankomst.

    Hvorfor det ser nesten VANNRETT ut: seismiske bølger går i ca. 4000 m/s gjennom
    berggrunn, mens en bil går i ca. 22 m/s (80 km/t). Over 3 km fiber bruker
    jordskjelvbølgen under ett sekund på å treffe alle kanalene, mens bilen bruker over
    to minutter. Det er nettopp denne enorme forskjellen i stigning som gjør at et
    jordskjelv og en bil er lette å skille i et waterfall-plott.

    P-bølgen kommer først, men er svak. S-bølgen kommer noen sekunder senere og er
    klart kraftigere. Tiden mellom dem (`sp_tid_s`) forteller hvor langt unna skjelvet er.
    """
    t = konfig.tidsakse
    posisjon_m = konfig.kanalakse_m

    # Bølgefronten treffer fjerne kanaler litt senere -> en svak helning.
    p_ankomst = ankomst_s + posisjon_m / tilsynelatende_hastighet_ms
    s_ankomst = p_ankomst + sp_tid_s

    def bolgepakke(ankomst: np.ndarray, frekvens: float, henfall_s: float, styrke: float) -> np.ndarray:
        """En dempet svingning som starter ved ankomsttiden for hver kanal."""
        dt = t[None, :] - ankomst[:, None]          # tid siden ankomst, per kanal
        aktiv = dt >= 0
        # Eksponentielt henfall ganget med en svingning: slik ser en seismisk fase ut.
        pakke = np.where(aktiv, np.exp(-np.maximum(dt, 0) / henfall_s) * np.sin(2 * np.pi * frekvens * dt), 0.0)
        return styrke * pakke

    # P er høyfrekvent og svak, S er lavfrekvent og sterk. Dette er den typiske ordenen.
    p_bolge = bolgepakke(p_ankomst, frekvens=8.0, henfall_s=0.6, styrke=0.3 * amplitude)
    s_bolge = bolgepakke(s_ankomst, frekvens=3.5, henfall_s=2.5, styrke=1.0 * amplitude)

    # Litt tilfeldig variasjon mellom kanaler, siden koblingen mot bakken varierer.
    variasjon = rng.normal(1.0, 0.15, size=(konfig.n_kanaler, 1))
    signal = (p_bolge + s_bolge) * variasjon

    waterfall += signal.astype(np.float32)

    # Vi merker fra P-ankomst og gjennom S-koden, siden det er der signalet er synlig.
    for i in range(konfig.n_kanaler):
        fra = np.searchsorted(t, p_ankomst[i])
        til = np.searchsorted(t, s_ankomst[i] + 4.0)
        etiketter[i, fra:til] = JORDSKJELV

    return {
        "type": "jordskjelv",
        "ankomst_s": ankomst_s,
        "amplitude": amplitude,
        "sp_tid_s": sp_tid_s,
    }


def legg_til_vannstoy(
    waterfall: np.ndarray,
    konfig: DASKonfig,
    rng: np.random.Generator,
    kanal_fra: int,
    kanal_til: int,
    styrke: float = 2.0,
    opptrapping: bool = True,
) -> dict:
    """
    Legger inn bredbåndet støy på et AVGRENSET kanalintervall — signaturen til
    rennende vann eller en oversvømmelse som brer seg over veibanen.

    Det som skiller dette fra regn er utstrekningen: flom rammer en strekning, regn
    rammer hele fiberen. Denne forskjellen er hele grunnen til at DAS kan lokalisere
    hvor problemet er, og ikke bare at det finnes et problem.

    `opptrapping` lar støyen vokse gradvis gjennom vinduet, slik en oversvømmelse faktisk
    utvikler seg. Det er dette som gjør at vi kan varsle FØR veien er ufremkommelig.
    """
    n_kanaler_rammet = kanal_til - kanal_fra
    stoy = rng.standard_normal((n_kanaler_rammet, konfig.n_tid))

    if opptrapping:
        # Lineær opptrapping fra 0 til full styrke gjennom vinduet.
        rampe = np.linspace(0.0, 1.0, konfig.n_tid)[None, :]
        stoy = stoy * rampe

    waterfall[kanal_fra:kanal_til] += (styrke * stoy).astype(np.float32)

    return {
        "type": "vannstoy",
        "kanal_fra": kanal_fra,
        "kanal_til": kanal_til,
        "styrke": styrke,
        "opptrapping": opptrapping,
    }


def legg_til_regn(
    waterfall: np.ndarray,
    konfig: DASKonfig,
    rng: np.random.Generator,
    styrke: float = 0.8,
) -> dict:
    """
    Bredbåndet støy over HELE fiberen — regn.

    Denne er med av en grunn som er lett å overse: regn er den viktigste kilden til
    FALSKE ALARMER. Det løfter støygulvet akkurat slik en flom gjør, bare overalt
    samtidig. Et system som ikke kan skille regn fra flom vil varsle rødt hver gang det
    høljer ned, og da slutter folk å bruke appen. Vi tar den med i datasettet nettopp
    for å kunne vise at vi håndterer den.
    """
    stoy = rng.standard_normal((konfig.n_kanaler, konfig.n_tid))
    waterfall += (styrke * stoy).astype(np.float32)
    return {"type": "regn", "styrke": styrke}


# ---------------------------------------------------------------------------
# Scener
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    """Et ferdig generert utsnitt med fasit."""
    waterfall: np.ndarray                  # (kanal, tid), selve DAS-dataene
    etiketter: np.ndarray                  # (kanal, tid), klasse per punkt
    konfig: DASKonfig
    hendelser: list[dict] = field(default_factory=list)
    beskrivelse: str = ""


def generer_scene(
    konfig: DASKonfig | None = None,
    n_biler: int = 6,
    med_jordskjelv: bool = False,
    med_regn: bool = False,
    flom_kanaler: tuple[int, int] | None = None,
    flom_styrke: float = 2.0,
    beskrivelse: str = "",
) -> Scene:
    """
    Setter sammen en komplett scene: bakgrunnsstøy pluss de hendelsene vi ber om.

    Dette er hovedinngangen til generatoren. Alt annet i prosjektet — treningsdata til
    klassifikatoren, "normale" data til autoencoderen, og flomscenariet i demoen —
    bygges av kall hit med ulike argumenter.
    """
    konfig = konfig or DASKonfig()
    rng = np.random.default_rng(konfig.seed)

    waterfall = farget_stoy(konfig, rng)
    etiketter = np.zeros(waterfall.shape, dtype=np.int8)
    hendelser: list[dict] = []

    # --- Biler -------------------------------------------------------------
    for _ in range(n_biler):
        retning = int(rng.choice([-1, 1]))
        # Biler som kjører bakover starter i motsatt ende av veien.
        start_m = konfig.veilengde_m if retning < 0 else 0.0
        # Litt slingringsmonn, så ikke alle starter i nøyaktig samme punkt.
        start_m += rng.uniform(-200, 200)

        hendelser.append(legg_til_bil(
            waterfall, etiketter, konfig, rng,
            # Typiske veifarter. Spredningen gir modellen noe å generalisere over.
            fart_kmt=float(rng.uniform(50, 100)),
            start_m=float(start_m),
            # MERK det negative området: start_s kan ligge FØR opptaket begynner.
            # Ekte trafikk synkroniserer seg ikke med når vi slår på måleutstyret — når
            # vi starter opptaket er det allerede biler midt på strekningen. Uten dette
            # ville bare den enden av veien der bilene kjører inn hatt trafikk, siden en
            # bil i 80 km/t bare rekker 1333 m på 60 s. Da ville modellene våre lært en
            # kunstig sammenheng mellom posisjon og trafikkmengde som ikke finnes i
            # virkeligheten.
            start_s=float(rng.uniform(-konfig.varighet_s * 0.9, konfig.varighet_s * 0.7)),
            # Mest personbiler, av og til noe tyngre.
            tyngde=float(rng.choice([0.6, 1.0, 1.0, 2.2])),
            retning=retning,
        ))

    # --- Jordskjelv --------------------------------------------------------
    if med_jordskjelv:
        # Ankomsttiden spres over HELE opptaket, ikke bare den første halvdelen.
        # Dette er ikke kosmetikk: når vi senere deler trenings- og testdata etter tid,
        # ville et skjelv som alltid kommer tidlig havnet utelukkende i treningsdelen,
        # og testsettet ville ikke inneholdt et eneste jordskjelv å måle på.
        # Jordskjelv inntreffer uansett ikke fortrinnsvis tidlig i et opptak.
        hendelser.append(legg_til_jordskjelv(
            waterfall, etiketter, konfig, rng,
            ankomst_s=float(rng.uniform(0.08, 0.85) * konfig.varighet_s),
        ))

    # --- Avvik -------------------------------------------------------------
    if med_regn:
        hendelser.append(legg_til_regn(waterfall, konfig, rng))

    if flom_kanaler is not None:
        hendelser.append(legg_til_vannstoy(
            waterfall, konfig, rng,
            kanal_fra=flom_kanaler[0], kanal_til=flom_kanaler[1],
            styrke=flom_styrke,
        ))

    return Scene(waterfall, etiketter, konfig, hendelser, beskrivelse)


def generer_normal_scene(seed: int, n_biler: int | None = None) -> Scene:
    """
    En scene som representerer NORMALTILSTANDEN: bakgrunnsstøy og vanlig trafikk,
    ingen avvik.

    Dette er byggesteinen for fase 4. Autoencoderen trenes utelukkende på slike scener,
    slik at den lærer hva som er vanlig på strekningen — og dermed feiler når noe uvant
    skjer. Poenget med den tilnærmingen er at vi ALDRI trenger flom-eksempler for å
    kunne oppdage en flom.
    """
    rng = np.random.default_rng(seed)
    if n_biler is None:
        # Normal trafikkvariasjon. Autoencoderen må tåle at det er litt ulikt fra
        # vindu til vindu, ellers slår den ut på helt vanlige rolige perioder.
        n_biler = int(rng.integers(4, 9))
    return generer_scene(
        konfig=DASKonfig(seed=seed),
        n_biler=n_biler,
        beskrivelse="normaltilstand",
    )


if __name__ == "__main__":
    # Kjør med:  python3 -m src.synthetic_das
    konfig = DASKonfig()
    print("--- SYNTETISK DAS-GENERATOR ---")
    print(f"Vei          : {konfig.veilengde_m:.0f} m ({konfig.n_kanaler} kanaler "
          f"a {konfig.kanalavstand_m:.0f} m)")
    print(f"Tid          : {konfig.varighet_s:.0f} s ved {konfig.samplingsrate_hz:.0f} Hz "
          f"({konfig.n_tid} tidssteg)")

    for navn, scene in [
        ("normal trafikk", generer_scene(n_biler=6, beskrivelse="normal trafikk")),
        ("jordskjelv", generer_scene(n_biler=4, med_jordskjelv=True, beskrivelse="jordskjelv")),
        ("flom pa strekning", generer_scene(n_biler=3, flom_kanaler=(120, 180),
                                            beskrivelse="flom")),
    ]:
        wf = scene.waterfall
        andel = {k: float((scene.etiketter == k).mean()) for k in (BAKGRUNN, BIL, JORDSKJELV)}
        print(f"\n{navn:20s} form={wf.shape} std={wf.std():.2f} "
              f"maks={np.abs(wf).max():.1f}")
        print(f"{'':20s} andel bakgrunn={andel[BAKGRUNN]:.2%} "
              f"bil={andel[BIL]:.2%} jordskjelv={andel[JORDSKJELV]:.2%}")
        print(f"{'':20s} hendelser: {len(scene.hendelser)}")
