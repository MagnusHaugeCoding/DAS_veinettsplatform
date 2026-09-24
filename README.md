# Trygg ferdsel med DAS

Kan fiberoptiske kabler som allerede ligger langs norske veier brukes til å varsle om at
en veistrekning er blitt utrygg — under flom, jordskjelv, ulykker eller krig?

Dette prosjektet er laget under en 48-timers casekonkurranse fra Kongsberggruppen.
Veikartet ligger i [`docs/overordnet_plan.md`](docs/overordnet_plan.md).

---

## Problem

Under en krise er det avgjørende, og i dag vanskelig, å vite om veien fra A til B fortsatt
er trygg. Informasjonen kommer typisk fra folk som melder inn i etterkant, fra kameraer
som bare dekker enkeltpunkter, eller fra værmodeller som ikke ser den konkrete veibanen.
Det finnes ingen sammenhengende, sanntids måling av veinettets tilstand.

## Hvorfor DAS

Distributed Acoustic Sensing sender laserpulser gjennom en vanlig optisk fiber og måler
det lyset som spres tilbake. Små strekninger i fiberen som tøyes av vibrasjoner endrer
tilbakespredningen, og man kan lese av vibrasjon **per meter langs hele kabelen**.

Det avgjørende poenget: **fiberen ligger der allerede.** Telekomfiber følger veinettet.
DAS gjør denne kabelen om til tusenvis av vibrasjonssensorer uten at det legges én meter
ny infrastruktur.

Dataene er en matrise med kanal (posisjon langs fiberen) på én akse og tid på den andre.
Visualisert kalles det et *waterfall*-plott. En bil blir en skrå linje, der stigningen er
farten. Et jordskjelv blir en nesten vannrett stripe, fordi bølgen treffer mange kanaler
nesten samtidig.

## Kjernepåstanden

Vi prøver ikke å slå fast at noe *er* en flom. Vi oppdager at en strekning **avviker fra
sin egen normaltilstand** — og et avvik er i seg selv handlingsrelevant informasjon:
"noe er unormalt her, snu eller vær forsiktig."

Det er en lavere teknisk terskel enn en flomklassifikator, og mer robust: systemet varsler
også om hendelser vi aldri har trent på. Varslene er graderte (grønn / gul / rød), fordi
for mange falske alarmer gjør at folk slutter å stole på appen.

## Kom i gang

```bash
git clone <repo-url>
cd DAS_veinettsplatform
```

### Virtuelt miljø (anbefalt framgangsmåte)

```bash
python3 -m venv .venv          # lager et isolert Python-miljø i mappen .venv/
source .venv/bin/activate      # tar det i bruk (prompten viser (.venv))
pip install -r requirements.txt
```

Hvorfor gjøre dette? Et virtuelt miljø holder prosjektets pakker for seg selv. Uten det
deler alle prosjekter på maskinen samme pakkeversjoner, og to prosjekter som trenger ulike
versjoner av for eksempel numpy vil ødelegge for hverandre. Miljøet gjør også at andre kan
gjenskape nøyaktig ditt oppsett fra `requirements.txt`.

Deaktiver med `deactivate` når du er ferdig.

> **Merk:** i denne konkurransen kjører vi mot det globale Python-miljøet, fordi
> TensorFlow allerede var installert der. Et nytt venv ville betydd å laste ned
> TensorFlow på nytt (flere hundre MB) — dårlig bruk av 48 timer. Oppskriften over er
> likevel den som gjør prosjektet reproduserbart på en ny maskin.

### Mappestruktur

| Mappe | Innhold |
|---|---|
| `src/` | Kodemoduler. Kjøres som `python3 -m src.<modul>` |
| `data/raw/` | Nedlastede ekte DAS-data (ikke i git — lastes ned ved behov) |
| `data/processed/` | Genererte syntetiske datasett (ikke i git) |
| `figures/` | Pitch-figurer (**er** i git — de er leveransen) |
| `app/` | Streamlit-demo |
| `docs/` | Overordnet plan |
| `notebooks/` | Utforskning underveis |

## Metode

### Syntetisk generator (`src/synthetic_das.py`)
Lager (kanal, tid)-matriser med bakgrunnsstøy og merkede hendelser. Bakgrunnen er
**farget støy** (1/f), ikke hvit støy — ekte DAS-støy har klart mest energi på lave
frekvenser, og hvit støy ville gjort oppgaven kunstig lett for modellene. Biler legges
inn som en bevegelig gaussisk innhylling (skrå linje, stigning = fart), jordskjelv som
P- og S-ankomst med endelig bølgehastighet, og avvik som bredbåndet støy over et
kanalintervall (flom) eller hele fiberen (regn).

### Ekte data (`src/real_data.py`)
Laster ned én SAFOD-fil, fjerner instrumentglitcher, og båndpassfiltrerer 2–40 Hz.

## Resultater

### Fase 1

**Figur 1** (`figures/fig1_waterfall_sammenligning.png`) viser syntetisk trafikk,
syntetisk jordskjelv og ekte jordskjelv side om side. De tre signaturene har hver sin
geometriske form i (kanal, tid)-planet: bil = skrå linje, jordskjelv = nesten vannrett
stripe, flom = loddrett søyle. Det er denne formforskjellen hele systemet hviler på.

**Figur 1b** (`figures/fig1b_glitch_vs_skjelv.png`) dokumenterer det viktigste funnet i
fasen, se under.

#### Funn 1: det sterkeste utslaget var ikke jordskjelvet  `[EKTE DATA]`

Vi fant først en kraftig "hendelse" ved t = 28,80 s og holdt på å bruke den i figurene.
Tre kontroller viste at den ikke er et jordskjelv:

1. Signalet hopper fra 3e-12 til 1,3e-5 på **én sample** — faktor fem millioner, momentant.
2. Alle 800 kanaler topper på **nøyaktig samme sample**. En bølge i 4000 m/s bruker minst
   0,2 s på 800 m fiber. Null forsinkelse er fysisk umulig.
3. Verdiene etter toppen avtar som en fortegnsvekslende geometrisk rekke — impulsresponsen
   til anti-aliasfilteret fra nedsamplingen 2,5 kHz → 250 Hz.

Det ekte jordskjelvet ligger ved **t = 9,56 s**, er bare **3,2x** over støygulvet, og har
tilsynelatende hastighet **3862–4513 m/s** — riktig område for berggrunn. Artefaktet er
altså rundt **8000 ganger sterkere** enn hendelsen vi faktisk er ute etter.

Dette er direkte relevant for systemdesignet: en ren amplitudeterskel ville slått ut på
artefaktet og oversett skjelvet. Derfor bygger vi på **koherens i (kanal, tid)-planet**,
ikke på styrke.

#### Funn 2: stabling over kanaler virker ikke som teorien sier  `[EKTE DATA]`

Lærebøkene sier at stabling av N kanaler gir √N ganger bedre signal/støy — 28x for våre
800 kanaler. Vi målte:

| Hendelse | Enkeltkanal | Stablet | Endring |
|---|---|---|---|
| M2.46, 11,7 km | 316 100 | 18 164 | **0,1x** (dårligere) |
| M2.86, 87 km | 3,5 | 4,0 | 1,2x |

To grunner til at teorien svikter: DAS-støy er **romlig korrelert** (laserstøy,
storskala bakkebevegelse), så den kansellerer ikke. Og DAS måler tøyning *langs* fiberen,
så fortegnet snur der fiberen endrer retning — å midle kanaler med motsatt fortegn
utsletter signalet. Vi bruker derfor ikke stabling som deteksjonsmetode.

## Begrensninger og ærlighet

Vi skiller tydelig mellom hva som er vist på **ekte** data og hva som er vist på
**syntetiske** data. Alle figurer og tall merkes `[EKTE DATA]` eller `[SYNTETISK]`.

- **Ekte data:** én fil fra [`ariellellouch/DASDetection`](https://github.com/ariellellouch/DASDetection)
  (SAFOD-arrayet), 250 Hz, 1 m kanalavstand. Gir ekte jordskjelvsignatur i ekte DAS-støy.
- **Det ekte datasettet er ikke fra en vei.** SAFOD er et forsknings-/borehullsarray.
  Vi har altså **ingen ekte veitrafikk** i dette prosjektet. Offentlig tilgjengelige
  DAS-datasett med veitrafikk finnes ikke lett tilgjengelig i
  [awesome-das](https://github.com/DAS-RCN/awesome-das)-listen.
- **Trafikk-, flom- og anomaliresultatene er derfor syntetiske**, generert av vår egen
  modell av hvordan slike signaturer ser ut. De viser at metoden fungerer på signaler med
  de rette egenskapene — ikke at den er validert på ekte veidata.
- Demoruten er fiktiv med oppdiktede stedsnavn. Vi bruker bevisst ikke ekte kartdata over
  kritisk infrastruktur.

Neste steg mot et reelt system ville være en pilot på en faktisk veistrekning med fiber,
for å samle ekte normaltilstand over tid.

## Logg

### Fase 0 — Oppsett *(ferdig)*
Prosjektstruktur opprettet (`src/`, `data/`, `figures/`, `app/`, `docs/`, `notebooks/`).
`requirements.txt` med pinnede versjoner av seks vanlige pakker — ingen DAS-spesifikke
biblioteker viste seg nødvendige. Miljøet var nesten komplett fra før: numpy 2.1.0,
scipy 1.16.3, matplotlib 3.10.8, TensorFlow 2.20.0 og pandas 2.3.3 var installert, så kun
streamlit 1.64.0 måtte legges til. `docs/overordnet_plan.md` skrevet som veikart for
fase 1-5, inkludert en forhåndsbestemt nedprioriteringsrekkefølge.

Kildevalg avklart i denne fasen: `ariellellouch/DASDetection` gir ekte jordskjelvdata i
filer på ~48 MB hver, så vi kan nøye oss med å laste ned én fil framfor hele repoet på
~1,5 GB. Samtidig ble en viktig begrensning tydelig — ingen av de lett tilgjengelige
offentlige DAS-datasettene inneholder veitrafikk, noe som avgjør hvor grensen mellom
demonstrert og antatt går i dette prosjektet.

### Fase 1 — Forstå dataene *(ferdig)*
Bygget syntetisk generator (`src/synthetic_das.py`) med farget bakgrunnsstøy, biler,
jordskjelv, flom og regn, alt med fasit. Lastet ned to ekte SAFOD-hendelser (48 MB hver)
og verifiserte formen: 800 kanaler × 14999 tidssteg = 60,0 s ved 250 Hz.

Mest tid gikk til å finne ut at det sterkeste utslaget i opptaket var en instrumentglitch
og ikke jordskjelvet — se Funn 1 over. Det førte til at `fjern_glitcher()` ble lagt til,
og at vi forkastet en planlagt figur om kanalstabling fordi målingene ikke støttet
påstanden (Funn 2).

To rettelser i generatoren underveis, begge funnet ved å se på figurene:
waterfall-plottene hadde feil orientering (tid måtte gå nedover, ellers blir jordskjelv
loddrett i stedet for vannrett), og alle biler startet samtidig med opptaket, noe som ga
kunstig tom vei i den ene enden. Ekte trafikk er allerede på veien når målingen starter.

### Fase 2+3 — Klassifikator
*Ikke startet.*
