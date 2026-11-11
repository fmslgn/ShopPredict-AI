import pandas as pd

# Veri setini oku
veri = pd.read_csv("online_shoppers_intention.csv")

# Veri setinin boyutunu göster
print("=== SHOPPREDICT AI - VERI ANALIZI ===")
print(f"Satir sayisi: {veri.shape[0]}")
print(f"Sutun sayisi: {veri.shape[1]}")

# Ilk 5 kaydi göster
print("\n=== ILK 5 KAYIT ===")
print(veri.head().to_string())

# Sutun isimlerini göster
print("\n=== SUTUN ISIMLERI ===")
for sutun in veri.columns:
    print("-", sutun)
print("\n=== VERI TURLERI ===")
print(veri.dtypes)

print("\n=== EKSIK VERI KONTROLU ===")
print(veri.isnull().sum())

print("\n=== SATIN ALMA DAGILIMI ===")
print(veri["Revenue"].value_counts())

print("\n=== SATIN ALMA YUZDELERI ===")
print((veri["Revenue"].value_counts(normalize=True) * 100).round(2))
import matplotlib.pyplot as plt
from pathlib import Path

# Grafiklerin kaydedilecegi klasoru olustur
Path("grafikler").mkdir(exist_ok=True)

# Satin alma sayilarini hesapla
satin_alma = veri["Revenue"].value_counts().reindex(
    [False, True], fill_value=0
)

# Grafigi olustur
plt.figure(figsize=(8, 5))
barlar = plt.bar(
    ["Satin alma yok", "Satin alma var"],
    satin_alma.values
)

plt.title("E-Ticaret Satin Alma Dagilimi")
plt.xlabel("Satin Alma Durumu")
plt.ylabel("Oturum Sayisi")

# Sutunlarin uzerine sayilari yaz
for bar in barlar:
    yukseklik = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yukseklik + 100,
        str(int(yukseklik)),
        ha="center"
    )

plt.ylim(0, satin_alma.max() * 1.15)
plt.tight_layout()

# Grafigi kaydet ve goster
plt.savefig(
    "grafikler/01_satin_alma_dagilimi.png",
    dpi=200
)
plt.show()
# GRAFIK 2 - AYLARA GORE SATIN ALMA ORANI

ay_sirasi = [
    "Jan", "Feb", "Mar", "Apr", "May", "June",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

aylik_oran = (
    veri.groupby("Month")["Revenue"].mean() * 100
).reindex(ay_sirasi).dropna()

plt.figure(figsize=(10, 5))
plt.bar(aylik_oran.index, aylik_oran.values)

plt.title("Aylara Gore Satin Alma Oranlari")
plt.xlabel("Ay")
plt.ylabel("Satin Alma Orani (%)")
plt.ylim(0, aylik_oran.max() * 1.2)

for i, oran in enumerate(aylik_oran.values):
    plt.text(
        i, oran + 0.4,
        f"%{oran:.1f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    "grafikler/02_aylik_satin_alma_oranlari.png",
    dpi=200
)
plt.show()
# GRAFIK 3 - ZIYARETCI TURUNE GORE SATIN ALMA

ziyaretci_analizi = veri.groupby("VisitorType")["Revenue"].agg(
    ["mean", "size"]
)
ziyaretci_analizi["oran"] = ziyaretci_analizi["mean"] * 100

plt.figure(figsize=(9, 5))
barlar = plt.bar(
    ziyaretci_analizi.index,
    ziyaretci_analizi["oran"]
)

plt.title("Ziyaretci Turune Gore Satin Alma Oranlari")
plt.xlabel("Ziyaretci Turu")
plt.ylabel("Satin Alma Orani (%)")
plt.ylim(0, ziyaretci_analizi["oran"].max() * 1.25)

for bar, oran, sayi in zip(
    barlar,
    ziyaretci_analizi["oran"],
    ziyaretci_analizi["size"]
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"%{oran:.1f}\n(n={sayi})",
        ha="center"
    )

plt.tight_layout()
plt.savefig("grafikler/03_ziyaretci_turu.png", dpi=200)
plt.show()

print("\n=== ZIYARETCI ANALIZI ===")
print(ziyaretci_analizi)
# ADIM 14 - VERI KALITESI KONTROLU

print("\n=== TEKRAR EDEN SATIRLAR ===")
print("Tekrar eden satir sayisi:", veri.duplicated().sum())

print("\n=== SAYISAL DEGISKENLERIN OZETI ===")

sayisal_sutunlar = [
    "Administrative",
    "Informational",
    "ProductRelated",
    "BounceRates",
    "ExitRates",
    "PageValues"
]

print(
    veri[sayisal_sutunlar]
    .describe()
    .round(2)
    .to_string()
)
# ADIM 15 - TEKRAR EDEN KAYITLAR

print("\n=== VERI TEMIZLEME ===")

veri_temiz = veri.drop_duplicates().copy()

print("Orijinal kayit sayisi:", len(veri))
print("Tekillestirilmis kayit sayisi:", len(veri_temiz))
print("Cikarilan tekrar sayisi:", len(veri) - len(veri_temiz))

# Orijinal CSV dosyasini degistirmeden yeni dosya olustur
veri_temiz.to_csv(
    "online_shoppers_clean.csv",
    index=False
)

print("Temiz veri dosyasi kaydedildi.")