\# Anonimizacija dogodkov z (hierarhično) k-anonimnostjo

Skripta `anonQ\_2.py` v Pythonu iz CSV datoteke z dogodki ustvari anonimiziran izvoz, ki dosega k-anonimnost na kombinaciji izbranih kvazi-identifikatorjev (npr. `generalized\_event`, `week\_number`, `weekday`, `time\_period`).

\## Kratek povzetek delovanja skripte za anonimizacijo

\*   Prebere CSV (delilnik `;`) z dogodki za vse \*\*GUID\*\* uporabnike v tabeli.

\*   \*\*Časovna generalizacija\*\* iz ISO časovnih žigov generalizira datum (npr. teden v letu, del dneva) in kvantizira čas (npr. v 3-urne bloke). 

\* \*\*Klasifikacija dogodkov\*\* na osnovi slovarja preslika dogodke v razrede (npr. `power\_event`, `system\_event`);  `dogodek` → `generalized\_event`.

\*   Sestavi \*\*kvazi‑identifikator\*\* `anonymized\_attribute` kot konkatenacijo izbranih stolpcev (`generalized\_event`, `week\_number`, `weekday`, `time\_period`) in nad združenim atributom zagotovi \*\*k-anonimnost\*\*.

&nbsp; \*   Najprej poskusi odpraviti kršitve k‑anonimnosti s \*\*hierarhičnim sploščenjem\*\* po enem od komponent kvazi-identifikatorja (privzeto `week\_number` → 100 za problematične skupine). 

&nbsp; \*   Če to ne zadošča, odstrani \*\*celotne skupine dogodkov\*\* (`REMOVE\_USERS=False`, privzeto) ali \*\*uporabnike\*\* (če `REMOVE\_USERS=True`).

&nbsp; \*   Izpiše \*\*povzetek\*\* (št. uporabnikov/dogodkov prej in potem) ter poskrbi za izvoz \*\*anonimiziranega nabora\*\*, \*\*povzetka odstranitev po dogodkih\*\* in \*\*poročila o odstranjenih uporabnikih\*\*.:

&nbsp;     \*   `anonymized\_events.csv`,

&nbsp;     \*   `event\_removal\_report.csv`,

&nbsp;     \*   `user\_removal\_report\_counts.csv`.

\*\*\*

\## Vhodni podatki

Pričakovani stolpci vhodne CSV datoteke (`;` separator):

\- `OD\_ISO` – časovni žig v ISO formatu,

\- `GUID` – identifikator uporabnika/namestitve,

\- `dogodek` – izvorni identifikator dogodka.

Na vrhu skripte so definirane poti do datotek in stikala za način delovanja:

\*   `INPUT\_FILE\_PATH`, `OUTPUT\_FILE\_PATH`, `USER\_REMOVAL\_REPORT\_FILE\_PATH`, `EVENT\_REMOVAL\_REPORT\_FILE\_PATH`

\*   `REMOVE\_USERS = False` (privzeto se odstranjujejo \*dogodki\*, ne uporabniki)

\*   `generalized\_event\_to\_remove`: seznam razredov dogodkov, ki jih pred obdelavo izločiš iz nabora (trenutno `measurement\_event`, `power\_event`, `system\_event`).

> Opomba: dodatni stolpci niso nujni, a ostanejo vmesno v DataFrame in se pred izvozom počistijo.

\## Nastavitve (na vrhu skripte)

\- `INPUT\_FILE\_PATH` – pot do vhodnega CSV (privzeto: `G:\\TS\\Izvoz\_newtimes\_month.csv`).

\- `OUTPUT\_FILE\_PATH` – datoteka z anonimiziranimi dogodki (privzeto: `anonymized\_events.csv`).

\- `USER\_REMOVAL\_REPORT\_FILE\_PATH` – poročilo o odstranjenih uporabnikih.

\- `EVENT\_REMOVAL\_REPORT\_FILE\_PATH` – poročilo o zmanjšanju (po kombinacijah kvazi-identifikatorjev).

\- `REMOVE\_USERS` – če `True`, se za dosego k-anonimnosti odstranjujejo \*uporabniki\*, sicer \*dogodki\*.

\- `generalized\_event\_to\_remove` – seznam razredov dogodkov, ki jih \*\*pred\*\* anonimizacijo filtriramo iz podatkov.

\- `contributing\_columns` – seznam kvazi-identifikatorjev, ki tvorijo `anonymized\_attribute`.

\- `flatten\_column` – ime stolpca, ki ga hierarhično sploščimo (privzeto `week\_number`).

\- `k` – raven anonimnosti (privzeto `5`).

\- `hour\_block\_duration` – širina bloka za kvantizacijo ur (privzeto `3`).

> Opomba: na voljo je tudi“hitro stikalo” z zakomentirano vrstico `generalized\_event\_to\_remove = \[]` za primere, ko želite to opraviti brez predhodnega filtra.

\## Izhodni podatki

\- `anonymized\_events.csv` – anonimizirani podatki z ohranjenimi stolpci `GUID` + izbrani kvazi-identifikatorji.

\- `event\_removal\_report.csv` – primerjava števila unikatnih `GUID` na kombinacijo kvazi-identifikatorjev \*\*pred\*\* in \*\*po\*\* anonimizaciji (vključno s seštevki po prvem stolpcu `generalized\_event`).

\- `user\_removal\_report\_counts.csv` – število odstranjenih dogodkov po uporabniku (samo, če je do odstranjevanja uporabnikov prišlo).

\## Delovanje

1\. \*\*Branje in priprava\*\*: CSV se prebere z ločevalnikom `;`. Iz `OD\_ISO` se izluščijo leto/mesec/dan/ura, ISO teden, dan v tednu in del dneva (`map\_hour\_to\_period`).

&nbsp; \*   `timestamp` = `pd.to\_datetime(OD\_ISO)` in iz njega: `year`, `month`, `day`, `hour`.

&nbsp; \*   `quantized\_hour` s korakom `hour\_block\_duration = 3` (npr. 14:xx → 12, 15:xx → 15, …).

&nbsp; \*   `week\_number` iz `dt.isocalendar().week`, `weekday` (`0–6`) in `time\_period` s funkcijo `map\_hour\_to\_period` (`night/morning/daytime/afternoon`).

2\. \*\*Posploševanje dogodkov\*\*: `dogodek` se preslika v `generalized\_event` preko slovarja `event\_mapping`.

&nbsp; \*   Na podlagi slovarja `event\_mapping` različne baterijske/napajalne/supervizorske dogodke razvrstimo v `power\_event` ali `system\_event`; alarmne v `emergency\_button` ipd.

&nbsp; \*   `dogodek` se preimenuje v `event`.

3\. \*\*Sestava kvazi-identifikatorja\*\*: iz `contributing\_columns` se zgradi niz `anonymized\_attribute`.

&nbsp; \*   `contributing\_columns = \['generalized\_event', 'week\_number', 'weekday', 'time\_period']` (po želji lahko dodamo tudi `event`, `year`, …).

&nbsp; \*   `anonymized\_attribute` nastane kot niz s `'-'.join` vrednosti na teh stolpcih.

4\. \*\*Hierarhično sploščenje\*\*: kršitve k-anonimnosti se najprej poskušajo omiliti s sploščenjem `flatten\_column` (npr. nastavitev `week\_number` na 100) za vse problematične skupine; če to ne zadošča, se takšne skupine odstranijo iz nabora.

5\. \*\*Doseganje k-anonimnosti\*\*:

\*\*Validacija anonimnosti:\*\* `check\_k\_anonymity(df, k)` zahteva, da ima vsaka vrednost `anonymized\_attribute` vsaj `k` \*unikatnih\* `GUID`.

&nbsp;   1.  Sploščanje

&nbsp;	Funkcija `maximize\_data\_retention\_flatten\_events(df, k, flatten='week\_number')`: za vse problematične `anonymized\_attribute` splošči po stolpcu `flatten` ( npr. začasno generalizira `week\_number), nato ponovno preveri anonimnost in vse, kar je še pod `k`, označi z `NaN`.

&nbsp;	> S spoščanjem najprej poskusimo posplošiti časovni atribut dogodka po izbranem stolpcu (npr. teden v letu - manj granularna časovna ločljivost), šele nato odstranimo skupine, ki ostanejo preveč redke.

&nbsp;	2. Redukcija

&nbsp;		\*   \*\*Strategija A – odstranjevanje dogodkov (privzeto):\*\*

Funkcija `maximize\_data\_retention\_remove\_events(df, k)` izbriše \*vse\* zapise `anonymized\_attribute`, kjer je število unikatnih `GUID` < `k`.

&nbsp;		\*   \*\*Strategija B – odstranjevanje uporabnikov (iterativno):\*\*  

&nbsp;   iterativno odstranjuje uporabnike, ki najmanj prispevajo k problematičnim skupinam, dokler ni pogoj izpolnjen. Funkcija `maximize\_data\_retention(df, k)` vsakič poišče problematično skupino, identificira uporabnika, ki najmanj prispeva v to skupino (po `value\_counts()`), in ga odstrani, kar ponavlja do izpolnitve k‑anonimnosti.

&nbsp;  > Izbiro med A/B določamo z `REMOVE\_USERS`.

&nbsp; 

6\. \*\*Shranjevanje rezultatov\*\* in izpis povzetkov na konzolo.

&nbsp;	\*   Po anonimizaciji se odstranijo neuporabni stolpci (vsi, ki \*\*niso\*\* v `contributing\_columns` ali `GUID`) in se shrani `anonymized\_events.csv`.

&nbsp;	\*   Ustvari se `event\_removal\_report.csv` s primerjavo \*pred/po\* po skupinah `contributing\_columns` (z dodatnimi “total” vrsticami po prvi koloni, tj. `generalized\_event`).

&nbsp;	\*   `user\_removal\_report\_counts.csv` zabeleži št. odstranjenih zapisov po `GUID`.

&nbsp;	\*   Na konzoli se izpiše povzetek: št. uporabnikov/dogodkov skupno, po čiščenju in po anonimizaciji; ter informacija, ali nabor dosega k‑anonimnost.

\*\*\*

\## Zagon

```bash

python anonQ\_2.py

```

> Skripto lahko poljubno preimenujete ; poskrbite, da poti do datotek v glavi skripte ustrezajo vaši strukturi.

\## Prilagoditve

\- Spremenite `contributing\_columns`, da dosežete željeno ravnovesje med utiliteto in zasebnostjo.

\- Razširite `event\_mapping` ali `generalized\_event\_to\_remove` glede na potrebe.

\- Zamenjajte `flatten\_column` (npr. na `weekday` ali `time\_period`) za drugačno hierarhijo.

\## Omejitve in opombe

\- k-anonimnost varuje pred identifikacijo po kvazi-identifikatorjih, \*\*ne\*\* pred napadi s pomočjo zunanjih virov (npr. povezovanje z drugimi zbirkami) in ne prepreči \*\*sklepanja o občutljivih atributih\*\* (preučite še l-divergenco ali t-closness).

\- Funkcija `check\_k\_anonymity` preverja k-anonimnost izključno nad `anonymized\_attribute`.

\- Če matrika dogodkov/časovnih vzorcev močno ni enakomerna, lahko odstranitev povzroči izgubo utilitete; razmislite o spremembi `k`, kolone `flatten\_column`, ali o dodatnem združevanju kategorij.

\## Primer pričakovane strukture izhoda

`anonymized\_events.csv` bo vseboval:

\- `GUID`

\- `generalized\_event`

\- `week\_number`

\- `weekday`

\- `time\_period`

\## Licenca in skladnost

\- Prepričajte se, da imate pravno podlago za obdelavo podatkov in da je anonimizacija del vaših pravilnikov (npr. DPIA/ROPA). 

\- Ta koda je namenjena raziskovalni/operativni uporabi brez garancije. Uporabnik je odgovoren za validacijo anonimizacije na svojih podatkih.
