Anonimizirana struktura (`GUID`, `week_number`, `weekday`, `time_period`, `generalized_event`) omogoča izgradnjo **napovednega modela za prihodnje dogodke**. Ta dokument navaja:

1.  **Priporočeno arhitekturo modela**,
2.  **Feature engineering**,
3.  **Primer celotne Python skripte** za učenje modela (klasični ML),
4.  **Primer naprednega LSTM/Transformer pristopa** (za sekvenčno modeliranje uporabnih vzorcev),
5.  **Priporočila za evalvacijo**.

***

# Cilj modela

**Napovedati najverjetnejši `generalized_event` glede na časovne značilnosti.**

To je klasičen problem za nadzorovano učenje:

*   **X (značilke)**:
    *   `week_number`
    *   `weekday`
    *   `time_period`
    *   `GUID` (opcijsko, če ne želiš napovedovati globalno, ampak personalizirano)

*   **y (ciljna spremenljivka)**:
    *   `generalized_event`

***

# Feature engineering (minimalni set za dober model)

## Kategorični atributi → integer encoding

*   `GUID`
*   `time_period`
*   `generalized_event`

## Numerični atributi → lahko ostanejo integer:

*   `week_number` (1–52)
*   `weekday` (0–6)

## Dodatne ciklične značilke (priporočljivo):

Za tedenski/dnevni ritem uporabiš sinus/kosinus:

```python
df["week_sin"] = np.sin(2 * np.pi * df["week_number"] / 52)
df["week_cos"] = np.cos(2 * np.pi * df["week_number"] / 52)

df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
```

***

# Modeli, ki dobro delujejo

### Za hitro produkcijsko uporabo:

*   **XGBoost**
*   **LightGBM**
*   **Random Forest**

→ Zelo robustni za tablične podatke, odlični za nepopolne vzorce.

### Za sekvenčno napoved (priporočam!):

Za **napovedovanje verjetnosti zaporedij** dogodkov (npr. “po `system_event` običajno sledi `power_event`”), so možnosti:

*   **LSTM**
*   **GRU**
*   **Transformer Encoder** (najboljši, a potrebuje več podatkov)

***

# Python skripta (LightGBM model)

Skripta je v datoteki `lightGBM_model.py`. Uporabi se na podatkih `anonymized_events.csv`.

### Zakaj LightGBM?

*   Dela zelo hitro,
*   dobro obvlada kategorikalne featurje,
*   odličen na podatkih z neenakomernimi razredi (kar velja za tvoje evente).

***

# Sekvenčno modeliranje (LSTM/Transformer) — napoved naslednjega dogodka

Če želiš **zaporedno napovedovanje**, moraš podatke pretvoriti v sekvence:

    GUID1: sys_event → power_event → system_event → ...
    GUID2: measurement → measurement → power_event → ...

Potem zgradiš dataset:

```python
sequence_length = 5
# X = (e1,e2,e3,e4,e5)
# y = e6
```

In uporabiš PyTorch Keras:

### Mini primer (psevdokoda):

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_events, 32),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(num_events, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(X_seq, y_seq, epochs=20, batch_size=64)
```

Ta pristop deluje **podobno kot jezikovni model (npr. predvidevanje naslednje besede)**.

***

# Evalvacija modela

Priporočam naslednje metrike:

### klasifikacijske:

*   **F1-score za vsak event**
*   **macro-F1** (ker imaš neuravnotežene razrede)
*   **confusion matrix**

### za sekvence:

*   **top‑k accuracy** (npr. ali je pravi dogodek med top 3)
*   **perplexity** (če uporabljaš verjetnostni model)

***

# Dodatne možnosti

✅ **Streamlit dashboard**:

*   prikaz podatkov
*   prikaz porazdelitev
*   učenje modela
*   prikaz napovedi

ali

✅ **napredni Transformer model** (kot mini GPT za dogodke)  
ali  
✅ **celovit Jupyter notebook** s poročili, grafi, tuningom, feature engineeringom.

***

**A) klasični ML model**,  
**B) sekvenčni LSTM/Transformer**,  
**C) personaliziran model za vsak GUID**,  
**D) generalni model za vse uporabnike**?

# ✅ Obseg podatkov za modeliranje?
**26 764 dogodkov** je dovolj velika količina podatkov, za izdelavo *zelo solidnega* napovednega modela. Za napovedovanje dogodkov v pametnem domu / oskrbovanih stanovanjih je **26k dogodkov povsem zadostnih**, ker imajo dogodki jasno sezonsko cikličnost (teden, ura dneva, dan v tednu). Glede na obseg razpoložljive baze velja, da je to:

*   dovolj za **klasične ML modele** (LightGBM, XGBoost, RandomForest),
*   dovolj tudi za **sekvenčne modele (LSTM/GRU)**, *če* je na voljo \~200–300 dogodkov na GUID,
*   premalo za velik **Transformer (GPT‑podoben)** model — vendar je možno uporabiti “mini-transformer”, ki dela z manjšimi sekvencami.

***

# Predlogi za izvedbo

## Kombiniran pristop:

### 1) **Globalni LightGBM model** (hitro, robustno)

Začnemo s tabličnim modelom, ki napove `generalized_event` glede na:

*   `week_number`
*   `weekday`
*   `time_period`
*   `GUID`
*   (optionally) sinus/kosinus transformacije

Ta model bo dal baseline.

### 2) **Personaliziran sekvenčni model (LSTM)**

Za GUID‑e z dovolj zaporedij (npr. nad 200 dogodkov) se naj izvede *sekvenčna napoved naslednjega dogodka*.

→ to drastično dvigne natančnost tam, kjer uporabniki/senzorji kažejo ponavljajoče vzorce.

***

# 📊 Minimalno število dogodkov za LSTM?

*   **≥ 10k dogodkov**: dovolj za globalni LSTM
*   **≥ 300 dogodkov na GUID**: dovolj za personaliziran LSTM
*   **< 150 dogodkov na GUID**: sekvenčni model ne bo bistveno boljši od LightGBM

Ker je na voljo **26 764**, bo 99% zadoščalo za globalni LSTM, mogoče tudi za per‑GUID modele, če posamezni GUID‑i niso preveč redki.

***

# Konkreten predlog cevovoda za podatke

## 1) **Analiza porazdelitve dogodkov po GUID**

Najprej se preveri:

```python
df["GUID"].value_counts().describe()
```

da vemo:

*   koliko GUID‑ov ima >100 dogodkov,
*   ali imamo dolg rep GUID‑ov z malo dogodki (kar boš odstranil iz LSTM modela).

***

# 2) **Enostaven baseline LightGBM**

(že na voljo — se uporabi.)

***

# 3) ⚡ LSTM model za napovedovanje naslednjega dogodka

**LSTM model** za 26 764 dogodkov je v datoteki `lstm_next_event.py`

***

# 🔎 Zakaj je to optimalno za tvoj dataset 26k zapisov?

### LightGBM:

*   zelo stabilen za manjše in srednje velike datasete
*   razume kategorije in časovne cipherje
*   ni prenaučen pri 26k vzorcih

### LSTM:

*   z 26k dogodki imamo dovolj primerov za zaporedja dolžine 10
*   nauči se periodičnih vzorcev
*   nauči se »tipičnih poti« med dogodki (npr. power → system → supervisory)

### Napredni Transformer:

*   pri 26k dogodkih lahko uporabimo mini‑transformerje (3–6 attention headov)
*   vendar ni bistveno boljši od LSTM pri tako majhnem obsegu podatkov

***

# Kaj lahko pričakujemo od rezultatov?

Realistična natančnost:

*   **LightGBM**: 60–80%
*   **LSTM (global)**: 65–82%
*   **LSTM (per GUID)**: lahko 85–95% za GUID‑e z močnimi vzorci
*   **Transformer**: podoben LSTM, razen če imaš >100k dogodkov

***

# Možnosti za nadaljevanje:

### **Production‑ready pipeline**:

*   ingest → feature engineering → modeli → evalvacije → shranjeni modeli → API

### **Dashboard za napovedi** (Streamlit):

*   vizualizacija dogodkov
*   napovedi naslednjega dogodka
*   per‑GUID analitika

### **Mini‑GPT za dogodke**:

*   Transformer, optimiziran za dataset <50k dogodkov
*   deluje kot "next event predictor"

***

Če želiš, lahko pogledava še natančno porazdelitev dogodkov po GUID — to je ključno za odločitev ali gremo v **per‑GUID LSTM modele**.

