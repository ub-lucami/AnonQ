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
# Anonimizacija dogodkov z (hierarhično) k-anonimnostjo

Skripta `anonQ_2.py` v Pythonu iz CSV datoteke z dogodki ustvari anonimiziran izvoz, ki dosega k-anonimnost na kombinaciji izbranih kvazi-identifikatorjev (npr. `generalized_event`, `week_number`, `weekday`, `time_period`).

## Kratek povzetek delovanja skripte za anonimizacijo

* Prebere CSV (delilnik `;`) z dogodki za vse **GUID** uporabnike v tabeli.

* **Časovna generalizacija** iz ISO časovnih žigov generalizira datum (npr. teden v letu, del dneva) in kvantizira čas (npr. v 3-urne bloke).

* **Klasifikacija dogodkov** na osnovi slovarja preslika dogodke v razrede (npr. `power_event`, `system_event`); `dogodek` → `generalized_event`.

* Sestavi **kvazi‑identifikator** `anonymized_attribute` kot konkatenacijo izbranih stolpcev (`generalized_event`, `week_number`, `weekday`, `time_period`) in nad združenim atributom zagotovi **k-anonimnost**.

	* Najprej poskusi odpraviti kršitve k‑anonimnosti s **hierarhičnim sploščenjem** po enem od komponent kvazi-identifikatorja (privzeto `week_number` → 100 za problematične skupine).

	* Če to ne zadošča, odstrani **celotne skupine dogodkov** (`REMOVE_USERS=False`, privzeto) ali **uporabnike** (če `REMOVE_USERS=True`).

	* Izpiše **povzetek** (št. uporabnikov/dogodkov prej in potem) ter poskrbi za izvoz **anonimiziranega nabora**, **povzetka odstranitev po dogodkih** in **poročila o odstranjenih uporabnikih**:

		* `anonymized_events.csv`,
		* `event_removal_report.csv`,
		* `user_removal_report_counts.csv`.

***

## Vhodni podatki

Pričakovani stolpci vhodne CSV datoteke (`;` separator):

- `OD_ISO` – časovni žig v ISO formatu,

- `GUID` – identifikator uporabnika/namestitve,

- `dogodek` – izvorni identifikator dogodka.

Na vrhu skripte so definirane poti do datotek in stikala za način delovanja:

- `INPUT_FILE_PATH`, `OUTPUT_FILE_PATH`, `USER_REMOVAL_REPORT_FILE_PATH`, `EVENT_REMOVAL_REPORT_FILE_PATH`

- `REMOVE_USERS = False` (privzeto se odstranjujejo *dogodki*, ne uporabniki)

- `generalized_event_to_remove`: seznam razredov dogodkov, ki jih pred obdelavo izločiš iz nabora (trenutno `measurement_event`, `power_event`, `system_event`).

> Opomba: dodatni stolpci niso nujni, a ostanejo vmesno v DataFrame in se pred izvozom počistijo.

## Nastavitve (na vrhu skripte)

- `INPUT_FILE_PATH` – pot do vhodnega CSV (privzeto: `G:\TS\Izvoz_newtimes_month.csv`).

- `OUTPUT_FILE_PATH` – datoteka z anonimiziranimi dogodki (privzeto: `anonymized_events.csv`).

- `USER_REMOVAL_REPORT_FILE_PATH` – poročilo o odstranjenih uporabnikih.

- `EVENT_REMOVAL_REPORT_FILE_PATH` – poročilo o zmanjšanju (po kombinacijah kvazi-identifikatorjev).

- `REMOVE_USERS` – če `True`, se za dosego k-anonimnosti odstranjujejo *uporabniki*, sicer *dogodki*.

- `generalized_event_to_remove` – seznam razredov dogodkov, ki jih **pred** anonimizacijo filtriramo iz podatkov.

- `contributing_columns` – seznam kvazi-identifikatorjev, ki tvorijo `anonymized_attribute`.

- `flatten_column` – ime stolpca, ki ga hierarhično sploščimo (privzeto `week_number`).

- `k` – raven anonimnosti (privzeto `5`).

- `hour_block_duration` – širina bloka za kvantizacijo ur (privzeto `3`).

> Opomba: na voljo je tudi “hitro stikalo” z zakomentirano vrstico `generalized_event_to_remove = []` za primere, ko želite to opraviti brez predhodnega filtra.

## Izhodni podatki

- `anonymized_events.csv` – anonimizirani podatki z ohranjenimi stolpci `GUID` + izbrani kvazi-identifikatorji.

- `event_removal_report.csv` – primerjava števila unikatnih `GUID` na kombinacijo kvazi-identifikatorjev **pred** in **po** anonimizaciji (vključno s seštevki po prvem stolpcu `generalized_event`).

- `user_removal_report_counts.csv` – število odstranjenih dogodkov po uporabniku (samo, če je do odstranjevanja uporabnikov prišlo).

## Delovanje

1. **Branje in priprava**: CSV se prebere z ločevalnikom `;`. Iz `OD_ISO` se izluščijo leto/mesec/dan/ura, ISO teden, dan v tednu in del dneva (`map_hour_to_period`).

	* `timestamp` = `pd.to_datetime(OD_ISO)` in iz njega: `year`, `month`, `day`, `hour`.

	* `quantized_hour` s korakom `hour_block_duration = 3` (npr. 14:xx → 12, 15:xx → 15, …).

	* `week_number` iz `dt.isocalendar().week`, `weekday` (`0–6`) in `time_period` s funkcijo `map_hour_to_period` (`night/morning/daytime/afternoon`).

2. **Posploševanje dogodkov**: `dogodek` se preslika v `generalized_event` preko slovarja `event_mapping`.

	* Na podlagi slovarja `event_mapping` različne baterijske/napajalne/supervizorske dogodke razvrstimo v `power_event` ali `system_event`; alarmne v `emergency_button` ipd.

	* `dogodek` se preimenuje v `event`.

3. **Sestava kvazi-identifikatorja**: iz `contributing_columns` se zgradi niz `anonymized_attribute`.

	* `contributing_columns = ['generalized_event', 'week_number', 'weekday', 'time_period']` (po želji lahko dodamo tudi `event`, `year`, …).

	* `anonymized_attribute` nastane kot niz s `'-'.join` vrednosti na teh stolpcih.

4. **Hierarhično sploščenje**: kršitve k-anonimnosti se najprej poskušajo omiliti s sploščenjem `flatten_column` (npr. nastavitev `week_number` na 100) za vse problematične skupine; če to ne zadošča, se takšne skupine odstranijo iz nabora.

5. **Doseganje k-anonimnosti**:

**Validacija anonimnosti:** `check_k_anonymity(df, k)` zahteva, da ima vsaka vrednost `anonymized_attribute` vsaj `k` *unikatnih* `GUID`.

	1. Sploščanje

		Funkcija `maximize_data_retention_flatten_events(df, k, flatten='week_number')`: za vse problematične `anonymized_attribute` splošči po stolpcu `flatten` ( npr. začasno generalizira `week_number), nato ponovno preveri anonimnost in vse, kar je še pod `k`, označi z `NaN`.

		> S spoščanjem najprej poskusimo posplošiti časovni atribut dogodka po izbranem stolpcu (npr. teden v letu - manj granularna časovna ločljivost), šele nato odstranimo skupine, ki ostanejo preveč redke.

		2. Redukcija

			* **Strategija A – odstranjevanje dogodkov (privzeto):**

		Funkcija `maximize_data_retention_remove_events(df, k)` izbriše *vse* zapise `anonymized_attribute`, kjer je število unikatnih `GUID` < `k`.

			* **Strategija B – odstranjevanje uporabnikov (iterativno):**

		iterativno odstranjuje uporabnike, ki najmanj prispevajo k problematičnim skupinam, dokler ni pogoj izpolnjen. Funkcija `maximize_data_retention(df, k)` vsakič poišče problematično skupino, identificira uporabnika, ki najmanj prispeva v to skupino (po `value_counts()`), in ga odstrani, kar ponavlja do izpolnitve k‑anonimnosti.

		> Izbiro med A/B določamo z `REMOVE_USERS`.

6. **Shranjevanje rezultatov** in izpis povzetkov na konzolo.

	* Po anonimizaciji se odstranijo neuporabni stolpci (vsi, ki **niso** v `contributing_columns` ali `GUID`) in se shrani `anonymized_events.csv`.

	* Ustvari se `event_removal_report.csv` s primerjavo **pred/po** po skupinah `contributing_columns` (z dodatnimi “total” vrsticami po prvi koloni, tj. `generalized_event`).

	* `user_removal_report_counts.csv` zabeleži št. odstranjenih zapisov po `GUID`.

	* Na konzoli se izpiše povzetek: št. uporabnikov/dogodkov skupno, po čiščenju in po anonimizaciji; ter informacija, ali nabor dosega k‑anonimnost.

***

## Zagon

```bash
python anonQ_2.py
```

> Skripto lahko poljubno preimenujete ; poskrbite, da poti do datotek v glavi skripte ustrezajo vaši strukturi.

## Prilagoditve

- Spremenite `contributing_columns`, da dosežete željeno ravnovesje med utiliteto in zasebnostjo.

- Razširite `event_mapping` ali `generalized_event_to_remove` glede na potrebe.

- Zamenjajte `flatten_column` (npr. na `weekday` ali `time_period`) za drugačno hierarhijo.

## Omejitve in opombe

- k-anonimnost varuje pred identifikacijo po kvazi-identifikatorjih, **ne** pred napadi s pomočjo zunanjih virov (npr. povezovanje z drugimi zbirkami) in ne prepreči **sklepanja o občutljivih atributih** (preučite še l-divergenco ali t-closness).

- Funkcija `check_k_anonymity` preverja k-anonimnost izključno nad `anonymized_attribute`.

- Če matrika dogodkov/časovnih vzorcev močno ni enakomerna, lahko odstranitev povzroči izgubo utilitete; razmislite o spremembi `k`, kolone `flatten_column`, ali o dodatnem združevanju kategorij.

## Primer pričakovane strukture izhoda

`anonymized_events.csv` bo vseboval:

- `GUID`

- `generalized_event`

- `week_number`

- `weekday`

- `time_period`

## Licenca in skladnost

- Prepričajte se, da imate pravno podlago za obdelavo podatkov in da je anonimizacija del vaših pravilnikov (npr. DPIA/ROPA).

- Ta koda je namenjena raziskovalni/operativni uporabi brez garancije. Uporabnik je odgovoren za validacijo anonimizacije na svojih podatkih.

***
