# Prompt til Claude Code: konseptskisse av DAS-varsling i en trafikkapp

Kopier alt under streken og lim det inn i Claude Code, startet fra rotmappen til `DAS_veinettsplatform`.

---

Jeg trenger en **visuell konseptskisse** (mockup) til en rapport. Den skal se ut som et skjermbilde fra en trafikkapp for norske veier, der vi har lagt til en ny funksjon: varsling basert på DAS (Distributed Acoustic Sensing) i fiberkabler som allerede ligger langs veien. Jeg skal ta skjermbilde av resultatet og lime det inn i rapporten. Det skal altså **ikke** være en fungerende app, bare se troverdig og profesjonelt ut.

## Kontekst (les dette først)

Les `README.md` for å forstå prosjektet. Det viktigste:

- DAS gjør en vanlig fiberkabel om til tusenvis av vibrasjonssensorer, én per meter langs veien.
- Dataene er en matrise med **kanal (posisjon langs fiberen)** på x-aksen og **tid nedover** på y-aksen. Dette kalles et waterfall-plott.
- Signaturene har ulik form: **bil = skrå linje** (stigningen er farten), **jordskjelv = nesten vannrett stripe**, **flom = loddrett søyle** over et avgrenset kanalintervall.
- Kjernepåstanden: systemet sier ikke "dette er en flom". Det sier at **en strekning avviker fra sin egen normaltilstand**. Varslene er graderte: grønn, gul, rød.
- Vi skiller alltid mellom ekte og syntetiske data. Denne skissen er fiktiv og skal merkes slik.

## Harde regler

1. **Ikke bruk, importer eller endre noe av den eksisterende koden** i `src/`, `app/`, `notebooks/` eller `data/`. Lag alt nytt i en egen mappe: `konseptskisse/`.
2. **Én enkelt fil:** `konseptskisse/index.html` med all HTML, CSS og JavaScript inline. Ingen rammeverk, ingen byggesteg, ingen npm. Den skal kunne åpnes ved å dobbeltklikke på filen.
3. **Ingen ekte kart, ekte stedsnavn eller ekte veinummer.** Kartet tegnes som stilisert SVG. Alle navn er oppdiktet.
4. **Ikke kopier Statens vegvesen.** Appen skal *ligne på* en norsk trafikkapp i stil og oppbygning (kart, varsler, strekningsliste), men bruk et oppdiktet appnavn, egen logo (enkel geometrisk form) og ingen av Vegvesenets logoer, navn eller nøyaktige fargeprofil.
5. **All "data" er fiktiv og hardkodet** eller generert med en fast seed, slik at bildet blir likt hver gang siden lastes.

## Scenario

- Oppdiktet region etter ekstremværet **"Ragna"**, sent på kvelden (klokka i appen viser 22:47).
- Trafikanten kjører fra **Elvdal** til **Fjellstad** på den oppdiktede veien **Fv 7xx** (skriv det som "Fv 7xx" eller finn på et nummer som tydelig ikke finnes, f.eks. "Fv 912").
- Ruten er delt i 7 navngitte strekninger, for eksempel:
  1. Elvdal sentrum – Moen (grønn)
  2. Moen – Storbrua (grønn)
  3. Storbrua – Nedre Lia (**rød**: tydelig avvik, loddrett søyle i waterfall)
  4. Nedre Lia – Skarsvingen (**gul**: svakt avvik, økende)
  5. Skarsvingen – Tjernet (grønn)
  6. Tjernet – Høgåsen (grønn)
  7. Høgåsen – Fjellstad (grønn)
- En alternativ rute via **Bakkegrenda** vises stiplet, "+14 min".

## Layout (dette er det viktigste)

Lag en fast komposisjon på **1600 × 1000 px**, sentrert på en rolig, lys bakgrunn, laget for å bli skjermbildet i sin helhet:

### Venstre: telefon (hovedmotivet)

En realistisk telefonramme (avrundede hjørner, notch/Dynamic Island, statuslinje med klokke 22:47, batteri og signal). Inni:

- **Push-varsel øverst** som glir ned over appen (statisk er greit): rødt ikon, teksten
  **"Avvik oppdaget: Storbrua – Nedre Lia"** og undertekst
  *"Fiberen langs veien måler uvanlige vibrasjoner. Snu eller velg alternativ rute."*
- **Kart** (stilisert SVG): terreng i dempede grønne/grå toner, en elv som krysser veien ved Storbrua, rutelinjen tegnet i segmenter farget etter status (grønn/gul/rød), en blå prikk for brukerens posisjon på strekning 2, og den stiplede alternative ruten.
- **Bunnark** (bottom sheet) som dekker nedre tredjedel:
  - Tittel "Din rute: Elvdal → Fjellstad"
  - Liste over de 7 strekningene med fargeprikk, navn og kort status ("Normal", "Svakt avvik", "Kraftig avvik")
  - Liten linje: "Kilde: fiberføler langs veien · sist målt for 12 s siden"
  - To knapper: **"Velg alternativ rute (+14 min)"** (primær) og **"Se detaljer"** (sekundær)
- Et lite merke i hjørnet av kartet: "DAS aktiv" med en pulserende prikk.

### Høyre: forklaringspanel ("Hvorfor er strekningen rød?")

Et rent panel ved siden av telefonen som viser *hva appen ser bak kulissene* for strekning 3. Det skal gjøre det lett for en leser av rapporten å forstå sammenhengen mellom DAS-signalet og fargen på kartet.

1. **To waterfall-plott side om side**, tegnet med `<canvas>` og generert i JavaScript med fast seed:
   - **"Normaltilstand (siste 7 døgn, samme tidspunkt)"**: farget bakgrunnsstøy og 3–5 skrå linjer (biler i ulik fart).
   - **"Nå"**: samme type bakgrunn og biler, men med en tydelig **loddrett søyle** med bredbåndet støy over et kanalintervall midt i plottet (der elva krysser). Bilene skal *stoppe opp eller forsvinne* rundt søylen, som om trafikken snur.
   - Akser: "Posisjon langs fiberen (m)" på x, "Tid (s)" nedover på y. Bruk en perseptuelt jevn fargeskala (viridis eller inferno) og en liten fargeskala-legende.
2. **Avviksscore over tid**: et enkelt linjediagram (SVG) for de siste 30 minuttene. Linja ligger lavt og flatt, stiger gjennom et gult bånd og krysser den røde terskelen ca. 22:41. Terskelene tegnes som stiplede horisontale linjer merket "Gul" og "Rød".
3. **Tre korte forklaringslinjer** i klarspråk:
   - "Strekningen oppfører seg annerledes enn sin egen normaltilstand."
   - "Mønsteret (loddrett søyle over ca. 180 m) ligner rennende vann, men systemet varsler på avviket, ikke på en gjetning om årsak."
   - "Ingen biler har passert strekningen de siste 6 minuttene."
4. En diskret merkelapp nederst i panelet: **"KONSEPTSKISSE · fiktive stedsnavn og syntetiske data"**.

## Visuell stil

- Rent, moderne, "offentlig tjeneste"-preg: mye luft, tydelig hierarki, avrundede kort, myke skygger.
- Skrift: `Inter` fra Google Fonts, med systemfont som reserve.
- Statusfarger: grønn `#2E9E5B`, gul `#F2B01E`, rød `#D64545`, blå for bruker/alternativ rute `#2F6FDB`. Nøytrale gråtoner ellers.
- Waterfall-plottene skal se ut som ekte vitenskapelige plott (skarpe piksler, `image-rendering: pixelated`), i kontrast til det glatte app-designet. Denne kontrasten er poenget: app for trafikanten, fysikk for fagfolk.
- Ingen emojier. Enkle SVG-ikoner.

## Ekstra: flere tilstander (valgfritt, men nyttig)

Legg til en URL-parameter `?tilstand=` så jeg kan ta tre bilder til rapporten:

- `?tilstand=normal`: alle strekninger grønne, ingen push-varsel, "Nå"-plottet ser ut som normaltilstanden.
- `?tilstand=gul`: strekning 3 er gul, svak søyle, varsel sier "Svakt avvik, kjør forsiktig".
- `?tilstand=rod` (standard): som beskrevet over.

Legg også til en liten, nesten usynlig knapperad nederst i hjørnet utenfor 1600×1000-rammen for å bytte tilstand, slik at den ikke kommer med på skjermbildet.

## Når du er ferdig

1. Åpne siden og kontroller at alt får plass innenfor 1600 × 1000 uten scrolling.
2. Hvis Playwright eller en headless Chrome er tilgjengelig, ta skjermbilder av de tre tilstandene og lagre dem som `konseptskisse/skjermbilde_normal.png`, `skjermbilde_gul.png` og `skjermbilde_rod.png` (med `deviceScaleFactor: 2` for skarp utskrift). Hvis ikke, forklar meg hvordan jeg tar skjermbildet selv i Chrome.
3. Gi meg en kort oppsummering av hva du laget og hvilke valg du tok.
