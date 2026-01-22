**Python skripta** `anonymized_analysis.py`, omogoča hitro **analizo in vizualizacijo** datoteke v formatu `anonymized_events.csv` s stolpci `GUID;week_number;weekday;time_period;generalized_event`

Skripta vključuje:

*   osnovno **statistiko**,
*   **distribucijo** po tednih, dneh in delih dneva,
*   **heatmap** (teden × del dneva, dan × generaliziran dogodek),
*   **top dogodke** po skupinah,
*   možnost filtriranja po posameznem `GUID`.

***

# 🔍 Kaj dobiš s skripto?

### hitri vpogled

*   koliko je GUID-ov
*   koliko je dogodkov
*   katerih dogodkov je največ

### vizualizacijo

*   porazdelitev dogodkov po **tednih**
*   porazdelitev po **dneh v tednu**
*   porazdelitev po **delih dneva** (night/morning/daytime/afternoon)
*   bar chart najpogostejših `generalized_event`

### heatmap

*   **week\_number × time\_period** → dobiš sezonske/tedenske vzorce
*   **weekday × generalized\_event** → vidiš obnašanje dogodkov po ciklih

### analizo za posamezen GUID

*   porazdelitev dogodkov po času
*   frekvenca posameznih generaliziranih dogodkov
*   prilagojen graf za posameznega uporabnika/senzorja



