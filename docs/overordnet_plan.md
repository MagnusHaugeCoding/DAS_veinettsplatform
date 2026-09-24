# Overordnet plan: Trygg ferdsel med DAS

Dette dokumentet er prosjektets veikart. Det beskriver hva vi bygger, hvorfor, i hvilken
rekkefølge, og hva vi kutter først hvis tiden blir knapp. Det er skrevet for å kunne
brukes direkte som grunnlag for pitchen.

---

## 1. Visjon og kjernepåstand

Det ligger allerede telekomfiber langs svært mange norske veier. Distributed Acoustic
Sensing (DAS) gjør denne fiberen om til tusenvis av vibrasjonssensorer — uten å legge
én meter ny infrastruktur. Vi skal vise at denne eksisterende fiberen kan drive en app
som gir trygg ferdsel fra A til B under kriser og krig.

**Kjernepåstanden, og det viktigste å forstå ved prosjektet:**

> Vi trenger ikke å si med 100 % sikkerhet at noe *er* en flom. Det er nok å oppdage at
> en veistrekning avviker fra sin egen normaltilstand. Avviket er i seg selv et nyttig
> signal: "noe er unormalt her — snu eller vær forsiktig."

Dette er en vesentlig lavere teknisk terskel enn å bygge en flomklassifikator, og det er
samtidig mer robust: systemet varsler også om hendelser vi aldri har sett før og aldri
har trent på. Derfor er **anomalideteksjon hovedmotoren** i systemet.

Klassifisering av biler og jordskjelv er **støttefunksjoner**. De tjener to formål:
de definerer hva "normalt" betyr for en strekning (normal trafikkmengde, normale farter),
og de gjør oss i stand til å *forklare* et varsel i stedet for bare å utløse det.

## 2. Hvorfor gradert varsling

Varslene skal være graderte — grønn, gul, rød — med en underliggende risikoscore.
Grunnen er ikke teknisk, men menneskelig: for mange falske alarmer får folk til å slutte
å stole på appen, og en app ingen stoler på redder ingen. Vi skal derfor vise avveiningen
mellom falske alarmer og tapte hendelser **eksplisitt**, som en kurve, og si høyt at
valget av terskel er et operativt og politisk valg — ikke et teknisk.

## 3. Datagrunnlag

### Ekte data
`ariellellouch/DASDetection` (SAFOD-arrayet), funnet via
[awesome-das](https://github.com/DAS-RCN/awesome-das).

- Én `.npy`-fil per jordskjelv, ca. 48 MB. Vi laster ned **én** fil, ikke hele repoet (~1,5 GB).
- 250 Hz samplingsrate (nedsamplet fra 2,5 kHz), 1 m kanalavstand, 10 m gauge length.
- `catalog.csv` gir magnitude, avstand og azimut per hendelse.

### Syntetisk data
Egen generator (`src/synthetic_das.py`) for veitrafikk, jordskjelv og avvikshendelser
(vannstøy, regn, trafikk som stopper). Konfigurerbar og med fast seed, så alt er
reproduserbart.

### Signaturer vi leter etter i waterfall-plottet
| Fenomen | Signatur |
|---|---|
| Bil | Skrå linje. Stigningen tilsvarer farten. |
| Jordskjelv | Nesten vannrett stripe — bølgen treffer mange kanaler nesten samtidig. Svak P-ankomst, så sterkere S. |
| Vann / flom (indirekte) | Økt bredbåndet støy nær vannet, regnstøy, og trafikk som sakner ned og til slutt forsvinner. |

## 4. Ærlighetskravet

Vi skiller tydelig mellom ekte og syntetiske resultater — i koden, i figurene, i appen og
i README. Hver figur og hvert tall merkes `[SYNTETISK]` eller `[EKTE DATA]`.

**Den viktigste begrensningen å eie åpent:** SAFOD er et forsknings-/borehullsarray, ikke
en vei. Datasettet gir oss ekte jordskjelvsignaturer i ekte DAS-støy, men **ingen ekte
veitrafikk**. Offentlig tilgjengelige DAS-datasett med veitrafikk finnes ikke lett
tilgjengelig i awesome-das-listen. Trafikk- og flomresultatene er derfor syntetiske.

Dette er ikke en svakhet vi skjuler — det er en styrke å presentere: vi vet presis hvor
grensen mellom demonstrert og antatt går, og en jury vil lete etter nettopp det.

## 5. Faser og leveranser

### Fase 0 — Oppsett *(0,5-1 t)*
Prosjektstruktur, `requirements.txt`, README som prosjektlogg, dette dokumentet.

### Fase 1 — Forstå dataene *(3-4 t)*
- `src/synthetic_das.py`: generator med farget bakgrunnsstøy (ekte DAS-støy har mer
  energi på lave frekvenser enn hvit støy), biler med variabel fart/tyngde/retning,
  jordskjelv med P- og S-ankomst, og avvikshendelser.
- `src/real_data.py`: laster ned og leser én SAFOD-fil. Skriver ut faktisk shape og
  dtype før vi bygger noe på antagelser.
- **Leveranse:** `figures/fig1_waterfall_sammenligning.png` — syntetisk vs. ekte side om
  side, med forklaring av hva vi ser.

### Fase 2+3 — Klassifikator for bil og jordskjelv *(4-5 t)*
Slått sammen til **én 3-klasse-klassifikator** (bakgrunn / bil / jordskjelv) for å spare tid.

- Klassisk baseline først: energiterskel over kanaler og tid. Viser hva CNN-et faktisk
  tilfører, og er en ærlig sammenligning.
- Lite CNN i Keras på utsnitt à 64 kanaler × 2 s (~20-50k parametre, minutter på CPU).
- Fartsestimat fra linjens stigning.
- **Metodekrav:** splitt trening/test **etter tid, ikke tilfeldig** (naboutsnitt i tid er
  nesten identiske — tilfeldig splitt ville lekket). Normaliser **hver kanal for seg**
  (kanalfølsomheten varierer langs fiberen). Rapporter **presisjon og gjenkalling per
  klasse**, ikke bare samlet nøyaktighet, som lyver ved ubalanserte klasser.
- Test også på det ekte jordskjelvutsnittet og rapporter ærlig om modellen overfører
  fra syntetisk til ekte — begge utfall er interessante funn.
- **Leveranse:** `figures/fig2_klassifikator.png` + resultattabell i README.

### Fase 4 — Avvik fra normaltilstand **(hovedfokus, 7-8 t)**
- Autoencoder trent **kun på normale data** per strekning. Den lærer å rekonstruere
  normaltilstanden og feiler når noe uvant skjer — derfor trenger vi ingen
  flom-eksempler i treningen. Flaskehalsen må holdes smal: en autoencoder med for stor
  kapasitet lærer å kopiere alt, også avvikene, og blir ubrukelig.
- Avviksscore = rekonstruksjonsfeil per strekning per tidsvindu.
- Kombineres med forklarbare indikatorer: trafikkmengde nå vs. normalt, energi i lave vs.
  høye frekvensbånd, stabilitet i fartsfordelingen.
- Samlet risikoscore 0-100 → grønn / gul / rød. Vekter settes manuelt og begrunnes;
  vi later ikke som de er optimalisert.
- **Scenariotest:** gradvis oversvømmelse over ~20 min simulert tid. Målet er å vise at
  scoren krysser "gul" **før** trafikken har stoppet helt — at systemet varsler i tid,
  ikke i etterkant.
- **Leveranse:** `fig3_avviksscore_tidslinje.png`, `fig4_terskel_avveining.png`,
  `fig5_flomscenario.png`. Fig. 5 er figuren som selger konseptet.

### Fase 5 — Demo for pitch *(3-4 t)*
`app/streamlit_app.py`: fiktiv rute A→B delt i 6-8 navngitte strekninger, fargekodet
etter risikonivå, som oppdateres mens flomscenariet spilles av. Simulert telefonvarsel
("Avvik oppdaget på strekning X. Snu eller velg alternativ rute"), og waterfall-plott
for valgt strekning ved siden av, så man ser *hvorfor* fargen endret seg.

Fiktiv rute med oppdiktede navn. **Ingen ekte kartdata over kritisk infrastruktur.**

## 6. Tidsbudsjett

| Fase | Innhold | Estimat |
|---|---|---|
| 0 | Oppsett | 0,5-1 t |
| 1 | Generator + ekte data + figurer | 3-4 t |
| 2+3 | Klassifikator (sammenslått) | 4-5 t |
| 4 | Anomalideteksjon + risikoscore | 7-8 t |
| 5 | Streamlit-demo + pitch-figurer | 3-4 t |
| — | README, pitch-forberedelse, buffer | 3-4 t |

## 7. Nedprioritering hvis tiden blir knapp

Avgjort på forhånd, så vi slipper å diskutere det når det haster:

1. **Kuttes først:** CNN-et i fase 2+3. Vi beholder den klassiske energiterskelen som
   bil-/jordskjelvdetektor. Historien overlever.
2. **Kuttes deretter:** fartsestimatet, og testen av klassifikatoren på ekte data.
3. **Ufravikelig:** fase 1, 4 og 5. Historien om at *avvik fra normaltilstand gir varsel*
   er det viktigste vi skal vise, og den krever generatoren, autoencoderen og demoen.
