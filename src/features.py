"""
Klassiske, forklarbare funksjoner: normalisering, energimål, referanseklassifikator
og fartsestimat.

Hvorfor vi bygger dette FØR et nevralt nett:
  Et CNN uten referanse er umulig å vurdere. Får vi 92 % nøyaktighet, er det bra? Det
  kommer helt an på hva en enkel regel klarer. Hvis energiterskelen også treffer 90 %,
  har nettet knapt tilført noe, og da bør vi bruke den enkle regelen — den er raskere,
  forklarbar og lar seg feilsøke. Referansen er altså ikke en formalitet, den er det som
  gjør tallene fra nettet tolkbare.

Designet her bygger direkte på funnet fra fase 1: det sterkeste utslaget i ekte DAS-data
var en instrumentglitch, ikke jordskjelvet. Ren amplitude er derfor et dårlig kriterium.
Referanseklassifikatoren vår bruker i stedet KOHERENS — hvor stor del av fiberen som
rister samtidig — og det er samme prinsipp CNN-et skal lære på egen hånd.
"""

from __future__ import annotations

import numpy as np

from .synthetic_das import BAKGRUNN, BIL, JORDSKJELV


def kanalstatistikk(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Regner ut median og robust spredning per kanal, til bruk som normaliseringsgrunnlag.

    Skal kalles på en LANG periode (typisk et helt opptak), ikke på et kort utsnitt.
    Se forklaringen i normaliser_per_kanal() for hvorfor det skillet er avgjørende.
    """
    median = np.median(data, axis=1, keepdims=True)
    mad = np.median(np.abs(data - median), axis=1, keepdims=True)
    skala = np.maximum(1.4826 * mad, 1e-12)
    return median, skala


def normaliser_med_statistikk(
    utsnitt: np.ndarray, median: np.ndarray, skala: np.ndarray
) -> np.ndarray:
    """Normaliserer et utsnitt med statistikk regnet ut på forhånd fra en lengre periode."""
    return ((utsnitt - median) / skala).astype(np.float32)


def normaliser_per_kanal(utsnitt: np.ndarray) -> np.ndarray:
    """
    Normaliserer hver kanal for seg: trekk fra medianen, del på robust spredning.

    Hvorfor per kanal og ikke over hele utsnittet: følsomheten varierer langs fiberen.
    Hvordan kabelen ligger mot bakken, skjøter og bøyer gjør at nabokanaler kan ha
    systematisk ulikt støynivå. I de ekte SAFOD-dataene så vi dette tydelig som
    loddrette striper i waterfall-plottet — enkelte kanaler er nærmest døde, andre langt
    mer støyende enn resten. Uten normalisering ville modellen bruke kapasitet på å lære
    hvilke kanalnumre som er støyende, i stedet for å lære hvordan hendelser ser ut.

    Hvorfor median og MAD, ikke gjennomsnitt og standardavvik: vi normaliserer utsnitt
    som KAN inneholde en hendelse. Gjennomsnitt og standardavvik trekkes kraftig av
    selve hendelsen, slik at et utsnitt med en sterk bil ville blitt skalert ned til å
    ligne bakgrunn. Median og MAD (median absolute deviation) påvirkes langt mindre.

    ADVARSEL — denne funksjonen skal IKKE brukes på korte utsnitt med en bil i.
    Vi målte konsekvensen: med 2-sekunders utsnitt falt gjenkallingen for bil til 0,11.
    Grunnen er at en bil er til stede nesten HELE vinduet i de kanalene den berører.
    Da er den ikke lenger en uteligger som medianen tåler — den ER fordelingen, blåser
    opp MAD-en, og blir delt på seg selv. Selv et robust mål hjelper ikke når hendelsen
    fyller vinduet.

    Riktig framgangsmåte for korte utsnitt er å hente median og spredning fra kanalens
    LANGTIDSBAKGRUNN med kanalstatistikk() og bruke normaliser_med_statistikk(). Det er
    også slik et driftssystem ville fungert: hver kanals normale støynivå er kjent fra
    en kalibreringsperiode, ikke regnet ut på nytt for hvert to-sekunders vindu.
    """
    median = np.median(utsnitt, axis=1, keepdims=True)
    mad = np.median(np.abs(utsnitt - median), axis=1, keepdims=True)
    # 1.4826 gjør MAD sammenliknbar med standardavviket for normalfordelt støy.
    skala = 1.4826 * mad
    # Døde kanaler har MAD = 0. Uten gulvet ville vi delt på null og fått uendelig.
    skala = np.maximum(skala, 1e-12)
    return ((utsnitt - median) / skala).astype(np.float32)


def energi_per_kanal(utsnitt: np.ndarray) -> np.ndarray:
    """Gjennomsnittlig energi (kvadrert amplitude) i hver kanal."""
    return np.mean(utsnitt ** 2, axis=1)


def koherens(utsnitt_normalisert: np.ndarray, terskel: float = 3.0) -> float:
    """
    Andelen av kanalene som er aktive samtidig — vårt mål på romlig koherens.

    Dette er den enkeltstørrelsen som best skiller de to hendelsestypene, og grunnen er
    geometrisk:
      - En BIL er på ett sted om gangen. I et 640 m bredt utsnitt berører den kanskje
        5-15 % av kanalene i løpet av vinduet.
      - Et JORDSKJELV treffer nesten alle kanalene samtidig, fordi bølgen går rundt
        4000 m/s. Koherensen nærmer seg 1.

    Merk at dette er nøyaktig samme test vi brukte manuelt i fase 1 for å avsløre
    instrumentglitchen — forskjellen er at glitchen hadde koherens 1,0 OG uendelig
    hastighet, mens et jordskjelv har høy koherens og endelig hastighet.
    """
    aktiv = np.abs(utsnitt_normalisert) > terskel
    # Andel kanaler som er aktive i minst ett tidssteg innenfor vinduet.
    return float(np.mean(aktiv.any(axis=1)))


def referanse_klassifiser(
    utsnitt_normalisert: np.ndarray,
    energiterskel: float = 6.0,
    koherensterskel: float = 0.55,
) -> int:
    """
    Referanseklassifikator uten maskinlæring. To enkle regler i rekkefølge:

    Forventer et utsnitt som ALLEREDE er normalisert per kanal mot langtidsbakgrunnen.

      1. Er det nok energi til at noe i det hele tatt skjer? Hvis ikke: BAKGRUNN.
      2. Rister mesteparten av fiberen samtidig? Hvis ja: JORDSKJELV. Hvis nei: BIL.

    Hele modellen er altså to terskler. Det gjør den fullstendig gjennomsiktig: når den
    tar feil, kan vi se nøyaktig hvorfor. Det er denne egenskapen CNN-et må være
    vesentlig bedre for å fortjene plassen sin.
    """
    norm = utsnitt_normalisert

    # Styrkemål: 90-persentilen over kanaler av maksimal absoluttverdi.
    #
    # Valget av persentil er ikke likegyldig, og vi lærte det på den harde måten. Først
    # brukte vi MEDIANEN over kanaler. Det ga gjenkalling 0,00 for bil — referansen fant
    # ikke en eneste bil. Grunnen er at en bil bare berører 5-10 av 64 kanaler, så
    # medianen over kanaler ser utelukkende bakgrunn og faller under terskelen.
    # Medianen er riktig for noe som rammer hele fiberen, men blind for alt lokalt.
    #
    # 90-persentilen fanger opp lokale hendelser og er fortsatt robust mot enkeltkanaler
    # som spretter. Dette er verdt å være nøye med: en referanse som er kunstig svak
    # får et nevralt nett til å se bedre ut enn det er.
    styrke = float(np.percentile(np.max(np.abs(norm), axis=1), 90))
    if styrke < energiterskel:
        return BAKGRUNN

    return JORDSKJELV if koherens(norm) > koherensterskel else BIL


def tilpass_referanse(
    X_normalisert: np.ndarray,
    y: np.ndarray,
    energi_kandidater: np.ndarray | None = None,
    koherens_kandidater: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Finner de to tersklene til referanseklassifikatoren ved søk på TRENINGSDATA.

    Dette er et spørsmål om rettferdighet i sammenligningen. CNN-et får tilpasse 22 595
    parametre til treningsdataene. Gir vi referansen faste terskler vi har gjettet, måler
    vi ikke "nevralt nett mot enkel regel" — vi måler "tilpasset modell mot utilpasset
    modell", og konklusjonen blir verdiløs. Referansen får derfor tilpasse sine to
    parametre på nøyaktig samme data.

    Vi optimerer makro-F1 (gjennomsnittet av F1 over de tre klassene) framfor nøyaktighet,
    fordi nøyaktighet ville premiert å svare "bakgrunn" på alt.
    """
    if energi_kandidater is None:
        energi_kandidater = np.linspace(2.0, 14.0, 25)
    if koherens_kandidater is None:
        koherens_kandidater = np.linspace(0.1, 0.9, 17)

    # Regn ut styrke og koherens én gang per utsnitt, så slipper vi å gjøre det på nytt
    # for hver kombinasjon av terskler.
    styrker = np.array([np.percentile(np.max(np.abs(x), axis=1), 90) for x in X_normalisert])
    koherenser = np.array([koherens(x) for x in X_normalisert])

    def makro_f1(pred: np.ndarray) -> float:
        f1er = []
        for k in (BAKGRUNN, BIL, JORDSKJELV):
            tp = np.sum((pred == k) & (y == k))
            fp = np.sum((pred == k) & (y != k))
            fn = np.sum((pred != k) & (y == k))
            pres = tp / (tp + fp) if tp + fp else 0.0
            gjen = tp / (tp + fn) if tp + fn else 0.0
            f1er.append(2 * pres * gjen / (pres + gjen) if pres + gjen else 0.0)
        return float(np.mean(f1er))

    beste, beste_score = (6.0, 0.55), -1.0
    for e in energi_kandidater:
        er_hendelse = styrker >= e
        for c in koherens_kandidater:
            pred = np.where(~er_hendelse, BAKGRUNN,
                            np.where(koherenser > c, JORDSKJELV, BIL))
            score = makro_f1(pred)
            if score > beste_score:
                beste_score, beste = score, (float(e), float(c))

    print(f"  Tilpassede terskler: energi={beste[0]:.2f}, koherens={beste[1]:.2f} "
          f"(makro-F1 pa treningsdata: {beste_score:.3f})")
    return beste


def estimer_fart(
    utsnitt: np.ndarray,
    kanalavstand_m: float,
    samplingsrate_hz: float,
    aktiv_terskel: float = 4.0,
    min_kanaler: int = 12,
) -> tuple[float, float]:
    """
    Estimerer farten til en bil ut fra stigningen på sporet i waterfall-plottet.

    METODEVALG — og hvorfor den åpenbare metoden ikke virket:
      Første forsøk var slant stack (tau-p): gjett en fart, skyv hver kanal i tid
      tilsvarende, og se om sporet blir vannrett. Den feilet fullstendig, og grunnen er
      geometrisk. En bil i 80 km/t flytter seg 44 m på 2 sekunder — altså bare 4-5
      kanaler ved 10 m kanalavstand. Tidsforskyvningen som trengs over 64 kanaler blir
      da 2880 sampler, i et vindu som bare er 200 sampler langt. Åpningen strekker ikke
      til, og optimeringen låser seg på den høyeste farten i søkeområdet, der
      forskyvningene så vidt får plass.

      Konklusjonen er verdt å merke seg: vinduet som passer til KLASSIFISERING (64
      kanaler x 2 s) er altfor lite til FARTSMÅLING. Fart krever et langt vindu i tid,
      typisk 20-60 sekunder, slik at bilen rekker å krysse mange kanaler.

    Metoden vi bruker i stedet er den samme moveout-tilpasningen som fungerte på det
    ekte jordskjelvet i fase 1: finn når hver kanal har sin energitopp, og tilpass en
    rett linje til ankomsttid mot kanalnummer. Stigningen på den linjen gir farten
    direkte, siden fart = kanalavstand / (tid per kanal).

    Vi bruker Theil-Sen-tilpasning (medianen av stigningen mellom alle kanalpar) i
    stedet for minste kvadraters metode. Grunnen er at døde og støyende kanaler gir
    helt feil topptidspunkt, og minste kvadrater lar seg trekke kraftig av slike
    uteliggere. Medianen bryr seg ikke om dem.

    BEGRENSNING: metoden forutsetter at ÉN bil dominerer vinduet. Med flere biler
    samtidig blandes sporene, og estimatet blir upålitelig. Skarpheten som returneres
    fanger delvis opp dette.

    Returnerer (fart i km/t, skarphet). Skarpheten er andelen kanaler som lar seg
    forklare av den tilpassede linjen, og sier hvor mye estimatet er verdt. Verdier
    nær 1 betyr et rent spor; lave verdier betyr at det ikke er noen tydelig linje.
    Fortegnet på farten angir kjøreretningen.
    """
    norm = normaliser_per_kanal(utsnitt)
    n_kanaler, n_tid = norm.shape

    # Glatt energien i hver kanal, så vi finner sporet og ikke en tilfeldig støypigg.
    vindu = max(3, int(0.3 * samplingsrate_hz))
    kjerne = np.ones(vindu) / vindu
    innhylling = np.abs(norm)
    glattet = np.apply_along_axis(lambda r: np.convolve(r, kjerne, mode="same"), 1, innhylling)

    topp_tid = np.argmax(glattet, axis=1) / samplingsrate_hz
    topp_verdi = np.max(glattet, axis=1)

    # Behold bare kanaler der det faktisk skjer noe. Resten har en meningslos topptid.
    aktive = np.where(topp_verdi > aktiv_terskel)[0]
    if len(aktive) < min_kanaler:
        return 0.0, 0.0

    k = aktive.astype(float)
    t = topp_tid[aktive]

    # Theil-Sen: medianen av stigningen mellom alle par av punkter.
    # Med mange kanaler blir alle par dyrt, så vi tar et tilfeldig utvalg par.
    rng = np.random.default_rng(0)
    n_par = min(4000, len(k) * (len(k) - 1) // 2)
    i1 = rng.integers(0, len(k), n_par)
    i2 = rng.integers(0, len(k), n_par)
    gyldig = i1 != i2
    i1, i2 = i1[gyldig], i2[gyldig]
    if len(i1) == 0:
        return 0.0, 0.0

    stigninger = (t[i2] - t[i1]) / (k[i2] - k[i1])      # sekunder per kanal
    stigning = float(np.median(stigninger))

    if abs(stigning) < 1e-9:
        return 0.0, 0.0

    fart_ms = kanalavstand_m / stigning
    fart_kmt = fart_ms * 3.6

    # Skarphet: hvor stor andel av de aktive kanalene ligger nær den tilpassede linjen.
    skjaering = float(np.median(t - stigning * k))
    forventet = stigning * k + skjaering
    naer = np.abs(t - forventet) < 0.5      # innenfor et halvt sekund
    skarphet = float(np.mean(naer))

    return fart_kmt, skarphet


if __name__ == "__main__":
    # Kjør med:  python3 -m src.features
    # Vi tester fartsestimatet mot generatorens FASIT — den eneste måten å vite om
    # metoden faktisk virker. Merk vinduet: 300 kanaler (3 km) over 60 s, altså stort
    # nok til at bilen rekker å krysse mange kanaler.
    from .synthetic_das import DASKonfig, legg_til_bil, farget_stoy

    print("--- Test av fartsestimat mot kjent fasit ---")
    print("Vindu: 300 kanaler x 60 s (3 km vei). En bil om gangen.\n")
    print(f"{'Sann fart':>11} {'Estimert':>11} {'Avvik':>9} {'Skarphet':>10}")
    print("-" * 45)

    avvik_liste = []
    for i, sann_fart in enumerate([30, 50, 70, 80, 90, 110, 130]):
        for retning in (1, -1):
            konfig = DASKonfig(seed=100 + i * 2 + (retning > 0))
            rng = np.random.default_rng(konfig.seed)
            wf = farget_stoy(konfig, rng)
            et = np.zeros(wf.shape, dtype=np.int8)
            start_m = 0.0 if retning > 0 else konfig.veilengde_m
            legg_til_bil(wf, et, konfig, rng, fart_kmt=sann_fart,
                         start_m=start_m, start_s=0.0, tyngde=1.0, retning=retning)

            est, skarp = estimer_fart(wf, konfig.kanalavstand_m, konfig.samplingsrate_hz)
            fasit = sann_fart * retning
            avvik = abs(est - fasit)
            avvik_liste.append(avvik)
            print(f"{fasit:10.0f}  {est:10.1f}  {avvik:8.1f}  {skarp:9.2f}")

    print("-" * 45)
    print(f"Median avvik: {np.median(avvik_liste):.1f} km/t")
    print(f"Andel innenfor 10 km/t: {np.mean(np.array(avvik_liste) < 10):.0%}")
