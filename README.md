# Anonimizacija dogodkov z (hierarhično) k-anonimnostjo

Skripta `anonQ_2.py` v Pythonu iz CSV datoteke z dogodki ustvari anonimiziran izvoz, ki dosega k-anonimnost na kombinaciji izbranih kvazi-identifikatorjev (npr. `generalized_event`, `week_number`, `weekday`, `time_period`).

## Ključne zmožnosti
- **Klasifikacija dogodkov** iz ID-jev v razrede (npr. `power_event`, `system_event`).
- **Časovna generalizacija** (teden v letu, del dneva) in **kvantizacija ur** (npr. v 3-urne bloke).
- **k-anonimnost** nad združenim atributom `anonymized_attribute`.
- **Dva načina zmanjševanja tveganja razkritja**:
  - odstranitev *uporabnikov* z redkimi vzorci (če `REMOVE_USERS=True`),
  - odstranitev *dogodkov* (če `REMOVE_USERS=False`, privzeto).
- **Hierarhično sploščenje** enega izmed kvazi-identifikatorjev (privzeto `week_number`) za reševanje kršitev k-anonimnosti.
- Izvoz **anonimiziranega nabora**, **povzetka odstranitev po dogodkih** in **poročila o odstranjenih uporabnikih**.

## Vhodni podatki
Pričakovani stolpci vhodne CSV datoteke (`;` separator):
- `OD_ISO` – časovni žig v ISO formatu,
- `GUID` – identifikator uporabnika/namestitve,
- `dogodek` – izvorni identifikator dogodka.

> Opomba: Dodatni stolpci niso nujni, a ostanejo vmesno v DataFrame in se pred izvozom počistijo.

## Izhodne datoteke
- `anonymized_events.csv` – anonimizirani podatki z ohranjenimi stolpci `GUID` + izbrani kvazi-identifikatorji.
- `event_removal_report.csv` – primerjava števila unikatnih `GUID` na kombinacijo kvazi-identifikatorjev **pred** in **po** anonimizaciji (vključno s seštevki po prvem stolpcu `generalized_event`).
- `user_removal_report_counts.csv` – število odstranjenih dogodkov po uporabniku (samo, če je do odstranjevanja uporabnikov prišlo).

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

## Kako deluje
1. **Branje in priprava**: CSV se prebere z `;`. Iz `OD_ISO` se izluščijo leto/mesec/dan/ura, ISO teden, dan v tednu in del dneva (`map_hour_to_period`).
2. **Posploševanje dogodkov**: `dogodek` se preslika v `generalized_event` preko slovarja `event_mapping`.
3. **Sestava kvazi-identifikatorja**: iz `contributing_columns` se zgradi niz `anonymized_attribute`.
4. **Hierarhično sploščenje**: kršitve k-anonimnosti se najprej poskušajo omiliti s sploščenjem `flatten_column` (npr. nastavitev `week_number` na 100) za vse problematične skupine; če to ne zadošča, se takšne skupine odstranijo iz nabora.
5. **Doseganje k-anonimnosti**:
   - *Odstranjevanje dogodkov*: izbriše vse zapise z `anonymized_attribute`, kjer je število unikatnih `GUID` < `k`.
   - *Odstranjevanje uporabnikov*: iterativno odstranjuje uporabnike, ki najmanj prispevajo k problematičnim skupinam, dokler ni pogoj izpolnjen.
6. **Shranjevanje rezultatov** in izpis povzetkov na konzolo.

## Zagon
```bash
python anonQ_2.py
```
> Skripto lahko preimenujete poljubno; poskrbite, da poti do datotek v glavi skripte ustrezajo vaši strukturi.

## Prilagoditve
- Spremenite `contributing_columns`, da dosežete željeno ravnovesje med utiliteto in zasebnostjo.
- Razširite `event_mapping` ali `generalized_event_to_remove` glede na potrebe.
- Zamenjajte `flatten_column` (npr. na `weekday` ali `time_period`) za drugačno hierarhijo.

## Omejitve in opombe
- k-anonimnost varuje pred identifikacijo po kvazi-identifikatorjih, **ne** pred napadi s pomočjo zunanjih virov (npr. povezovanje z drugimi zbirkami) in ne prepreči **sklepanja o občutljivih atributih** (poglejte še l-divergenco ali t-closness).
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

