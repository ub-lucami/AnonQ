# Anonimizacija dogodkov z (hierarhično) k-anonimnostjo

Opis orodij in skript

- `anonQ_2.py`: v Pythonu iz CSV datoteke z dogodki ustvari anonimiziran izvoz, ki dosega k-anonimnost na kombinaciji izbranih kvazi-identifikatorjev (npr. `generalized_event`, `week_number`, `weekday`, `time_period`).
- `anonymized_analysis.py`: analiza podatkov v anonimizirani tabeli: osnovna statistika, distribucija po tednih, dneh in delih dneva, heatmap (teden × del dneva, dan × generaliziran dogodek), top dogodki po skupinah, možnost filtriranja po posameznem GUID.
- `dashboard.py`: interaktivna analiza podatkov v Streamlit. 
- `evaluate_utility_extended.py`: razširjeno ovrednotenje prediktivnih modelov lightGBM in LTSM (Bootstrap)
- `export_model_outputs.py`: ovrednotenje + report modelov
- `guid_distribution.py`: porazdelitveni plot po GUID
- `lightGBM_model.py`
- `lstm_next_event.py`
- `requirements.txt`

Vsako orodje ima podroben opis v `./docs`

Poleg tega so na voljo še dokumenti:
- `hierarchical_anonimity_SOA_beyond.md`
- `modeling.md`
- `tbd.md`

## Licenca in skladnost
- Prepričajte se, da imate pravno podlago za obdelavo podatkov in da je anonimizacija del vaših pravilnikov (npr. DPIA/ROPA). 
- Ta koda je namenjena raziskovalni/operativni uporabi brez garancije. Uporabnik je odgovoren za validacijo anonimizacije na svojih podatkih.

