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

### Klassifikator (`src/features.py`, `src/classifier.py`)
Utsnitt på 64 kanaler × 2 s. To modeller sammenlignes: en referanse uten maskinlæring
(energiterskel + koherens, altså to tall) og et CNN med 22 595 parametre. Begge får
tilpasse parametrene sine på de samme treningsdataene, slik at sammenligningen er
rettferdig. Trening/test deles **etter tid med karantenesone**, og hver kanal
normaliseres mot sin egen langtidsbakgrunn.

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

### Fase 2+3

**Figur 2** (`figures/fig2_klassifikator.png`) og **Figur 3**
(`figures/fig3_deteksjonsdemo.png` — demonstrasjonsfiguren).

Fire klasser: bakgrunn, bil, jordskjelv og **"noe annet"**. Den siste er bevisst vagt
definert, og det er selve poenget: systemet påstår ikke å vite at det er en flom, bare
at strekningen avviker fra sin normaltilstand.

#### Syntetiske testdata  `[SYNTETISK]`
1680 testutsnitt, delt etter tid med karantenesone.

| Klasse | Referanse F1 | CNN F1 |
|---|---|---|
| bakgrunn | 0,79 | 0,75 |
| bil | 0,74 | 0,76 |
| jordskjelv | 0,23 | 0,73 |
| noe annet | 0,00 | 0,68 |
| **nøyaktighet** | **0,653** | **0,742** |

Referansen har ingen regel for "noe annet" og får derfor 0 der. **Fartsestimat:**
medianavvik 0,5 km/t.

#### Ekte data  `[EKTE DATA]`

| Klasse | Referanse F1 | CNN F1 |
|---|---|---|
| bakgrunn | 1,00 | 0,99 |
| jordskjelv | 1,00 | 1,00 |

**Forbehold:** ett jordskjelv, 6 jordskjelvutsnitt av 81. Klassene "bil" og "noe annet"
forekommer ikke i SAFOD-data, så de er ikke testet på ekte data i det hele tatt.

#### Funn 3: CNN-et overførte først IKKE — og det var treningsdataene som var feil

Første runde feilet CNN-et fullstendig på ekte data: jordskjelv F1 **0,00**, alle seks
ekte jordskjelvutsnitt ble kalt "bil". Referansen traff perfekt. Nettet hadde lært
teksturen i vår egen støymodell, ikke fysikken.

Vi målte oss fram til tre årsaker og rettet dem:

1. **En bug i generatoren.** Kanalfølsomhet ble bare lagt på støyen, ikke på hendelsene.
   En "død" kanal fikk dermed 100x svakere støy men full hendelsesamplitude, og etter
   normalisering ble hendelsen der 3244x for sterk. Fysisk galt — dårlig kobling demper
   alt kanalen registrerer.
2. **Feil kalibreringsgrunnlag.** Vi normaliserte mot opptaket selv, som inneholdt
   flommen, og fjernet dermed delvis det vi skulle oppdage. Nå normaliserer vi mot en
   uavhengig normalperiode fra samme fiber — slik et driftssystem ville brukt historikk.
3. **For smal støymodell.** Vi innførte **domenerandomisering**: spektralhelning
   (f^+2,4 til f^-2,4), kanalfølsomhet (5x til over 1000x), andel døde kanaler (0–8 %)
   og jordskjelvfrekvens (2,5–30 Hz) varierer nå bredt mellom scener. Nettet *kan* da
   ikke lene seg på hvordan støyen ser ut, og må lære geometrien.

Etter dette: jordskjelv F1 **1,00** på ekte data.

Lærdommen er generell nok til pitchen: **når en modell ikke overfører, er det som regel
treningsdataene som må rettes, ikke modellen.**

#### Funn 2: stabling over kanaler virker ikke som teorien sier  `[EKTE DATA]`

Lærebøkene sier at stabling av N kanaler gir √N ganger bedre signal/støy — 28x for våre
800 kanaler. Vi målte:

| Hendelse | Enkeltkanal | Stablet | Endring |
|---|---|---|---|
| M2.46, 11,7 km | 316 100 | 18 164 | **0,1x** (dårligere) |
| M2.86, 87 km | 3,5 | 4,0 | 1,2x |

**Korrigert forklaring.** Vi antok først at årsaken var romlig korrelert støy. Det er
**feil**, og vi målte det: korrelasjonen mellom nabokanaler i støy er r = 0,02 ved 1 m og
praktisk talt null lenger unna. Støyen er altså uavhengig, og den oppfører seg faktisk
som teorien — amplituden faller med faktor 0,070 ved stabling av 800 kanaler, nær det
teoretiske 0,035.

Det er **signalet** som ikke overlever. Målt i jordskjelvvinduet:

| Kanalavstand | Signalkorrelasjon | Støykorrelasjon |
|---|---|---|
| 1 m | +0,91 | +0,02 |
| 5 m | +0,85 | +0,01 |
| 25 m | **−0,20** | −0,01 |
| 50 m | **−0,37** | +0,01 |

Signalet er sterkt koherent over noen få meter, men **snur fortegn** på 25–50 m. Fortegnet
på signaltoppen er 50,1 % positivt og 49,9 % negativt over arrayet. Signalet faller derfor
med nøyaktig samme faktor som støyen (0,070x), og netto SNR-endring blir 0,99x.

Årsaken er sannsynligvis at DAS måler tøyning *langs* fiberen, kombinert med at
SAFOD-fiberen ikke ligger rett. Konsekvensen er uansett klar: **blind stabling over
kanaler er ubrukelig som deteksjonsmetode her.** Riktig framgangsmåte er å utnytte
koherens innenfor den avstanden signalet faktisk henger sammen over.

## Hvor troverdige er de syntetiske dataene?

Vi målte det i stedet for å anta. Tre egenskaper, mot to ekte referanser:

| Egenskap | Ekte veikant | Ekte borehull | Vår generator |
|---|---|---|---|
| Spektralhelning (effekt ~ f^x) | f^−0,05 | f^+1,88 | f^−2,17 til f^+2,37 ✅ |
| Kurtose (impulsivitet) | 12,9 | 0,0 | −0,9 til 55 ✅ |
| Kanalfølsomhet, forhold | — (kun 2 kanaler) | 1387x | ca. 5x til >1000x ✅ |

**Det som er troverdig: geometrien, og det er den som betyr mest.** Bølgehastigheten vi
genererer (2500–6000 m/s) omslutter det vi målte i ekte data (3862–4513 m/s).
Formforskjellen mellom bilens skrå linje og jordskjelvets vannrette stripe er ren
kinematikk — en bil i 22 m/s mot en bølge i 4000 m/s. Den fysikken er riktig uansett hvor
god støymodellen er, og det er nettopp den modellene lærer.

**Det som ikke er troverdig:** vi treffer ikke noen enkelt ekte støymodell, og prøver
heller ikke. Strategien er domenerandomisering — dekke et bredt spenn som omslutter
virkeligheten — fordi ekte DAS-støy uansett varierer mellom fiberstrekninger, utstyr,
årstid og vær. Vi kan vise at spennet dekker begge de ekte referansene vi har, men vi
har bare to, og den ene har bare to kanaler.

**Det vi ikke kan si noe om i det hele tatt:** om trafikksignaturene våre ligner ekte
trafikk. Til det trengs et flerkanals veikantdatasett, og vi har ikke funnet et
tilgjengelig et.

## Begrensninger og ærlighet

Vi skiller tydelig mellom hva som er vist på **ekte** data og hva som er vist på
**syntetiske** data. Alle figurer og tall merkes `[EKTE DATA]` eller `[SYNTETISK]`.

- **Ekte data:** én fil fra [`ariellellouch/DASDetection`](https://github.com/ariellellouch/DASDetection)
  (SAFOD-arrayet), 250 Hz, 1 m kanalavstand. Gir ekte jordskjelvsignatur i ekte DAS-støy.
- **Ekte veikantfiber:** to kanaler fra Farmers Loop Road, Alaska
  ([FiberOpticEarthquakes](https://github.com/eileenrmartin/FiberOpticEarthquakes)),
  1000 Hz, 120 s. Så vidt vi kan se det eneste lett tilgjengelige veikant-DAS i hele
  awesome-das-listen. **Bare 2 kanaler**, så vi kan ikke lage waterfall-plott eller se
  bilens skrå linje. Opptaket er dessuten kl. 03:23 lokal tid — nesten ingen trafikk.
  Vi bruker det ikke til trening eller testing, men til å **måle hvordan ekte
  veikantstøy ser ut**, og dermed vurdere troverdigheten til generatoren vår.
- **Det ekte jordskjelvdatasettet er ikke fra en vei.** SAFOD er et
  forsknings-/borehullsarray. Vi har derfor **ingen brukbar ekte veitrafikk** i dette
  prosjektet. PubDAS inneholder veikant-array med rikelig trafikk (blant annet FORESEE),
  men ligger bak Globus, som krever egen konto og klient — utenfor rekkevidde på 48 timer.
  Det er den naturlige neste datakilden hvis prosjektet skal videre.
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

### Fase 2+3 — Klassifikator *(ferdig)*
Bygget referanseklassifikator, CNN og fartsestimat. Tre feil ble funnet og rettet
underveis, alle ved å teste mot fasit i stedet for å anta:

1. **Testsettet inneholdt null jordskjelv.** Generatoren plasserte alltid skjelvet i
   første halvdel av opptaket, og tidsdelingen la dermed alle i treningsdelen.
   Ankomsttiden spres nå over hele vinduet.
2. **Per-kanal normalisering over 2 s skalerte bort bilene** (gjenkalling 0,11). En bil
   fyller nesten hele tidsvinduet i de kanalene den berører, så den blåser opp MAD-en og
   blir delt på seg selv. Rettet ved å hente normaliseringen fra kanalens
   langtidsbakgrunn — som også er slik et driftssystem ville fungert. Bil-F1 gikk fra
   0,00 til 0,83.
3. **Referansen var en stråmann.** Den brukte medianen over kanaler som styrkemål og
   fant derfor ingen biler i det hele tatt. Med 90-persentilen og tilpassede terskler
   gikk den fra 0,52 til 0,80 nøyaktighet — og gjorde sammenligningen ærlig.

Fartsestimat: første forsøk med slant stack feilet fullstendig fordi et 2-sekunders
vindu ikke gir nok åpning (en bil i 80 km/t krysser bare 4–5 kanaler på 2 s). Erstattet
med samme moveout-tilpasning som virket på det ekte skjelvet i fase 1.

### Fase 2+3, runde 2 — optimalisering *(ferdig)*
La til fjerde klasse "noe annet". Rettet en generatorbug (kanalfølsomhet gjaldt bare
støy), byttet til uavhengig normalperiode som kalibreringsgrunnlag, og innførte
domenerandomisering. Resultat: CNN-et gikk fra F1 0,00 til 1,00 på ekte jordskjelvdata.
Laget `figures/fig3_deteksjonsdemo.png`, som viser hele scenarier med modellens vurdering
lagt oppå som fargede ruter.

### Troverdighetsmåling mot ekte veikantfiber *(ferdig)*
Lastet ned de to Farmers Loop Road-sporene og målte generatorens støy mot dem. Fant at
spektralhelningen allerede var dekket av domenerandomiseringen, men at **impulsiviteten
ikke var det i det hele tatt** — ekte veikantstøy har kurtose 12,9, vår rent gaussiske
modell hadde -0,7. La til impulsiv støy som randomisert parameter. Etter dette spenner
generatoren fra kurtose -0,9 til 55, altså over både veikant (12,9) og borehull (0,0).

Resultat etter endringen: syntetisk nøyaktighet 0,752 (opp fra 0,742), og på ekte data
bakgrunn F1 1,00 og jordskjelv F1 0,91.

### Fase 4 — Avvik fra normaltilstand
*Ikke startet.*
