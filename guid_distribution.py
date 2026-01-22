
import pandas as pd
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------
# Load CSV (local-only, no network calls)
# ----------------------------------------------------
FILE = "G:/TS_2025/anonymized_events.csv"   # prilagodi ime datoteke

df = pd.read_csv(FILE, sep=";")
os.makedirs("png", exist_ok=True)
# ----------------------------------------------------
# Count events per GUID
# ----------------------------------------------------
counts = df["GUID"].value_counts()

print("\n### PORAZDELITEV DOGODKOV PO GUID ###")
print(counts)

print("\n### OSNOVNA STATISTIKA ###")
print(counts.describe())

print("\n### TOP 20 GUID z največ dogodki ###")
print(counts.head(20))

# ----------------------------------------------------
# Plot histogram
# ----------------------------------------------------
plt.figure(figsize=(8,5))
counts.hist(bins=40)
plt.title("Porazdelitev števila dogodkov na GUID")
plt.xlabel("Število dogodkov")
plt.ylabel("Število GUID-ov")
plt.grid(True)
plt.tight_layout()
plt.savefig("png/guid_distribution_histogram.png", dpi=150)
plt.show()

# ----------------------------------------------------
# Logarithmic histogram
# ----------------------------------------------------
plt.figure(figsize=(8,5))
plt.hist(counts, bins=40, log=True)
plt.title("Porazdelitev dogodkov na GUID (log-scale)")
plt.xlabel("Število dogodkov")
plt.ylabel("Število GUID-ov (log)")
plt.grid(True)
plt.tight_layout()
plt.savefig("png/guid_distribution_log_histogram.png", dpi=150)
plt.show()

print("\nGrafi so shranjeni kot PNG datoteke v mapi png/.")
