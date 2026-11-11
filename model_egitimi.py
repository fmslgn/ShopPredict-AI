import pandas as pd
from sklearn.model_selection import train_test_split

# Temizlenmis veri setini oku
veri = pd.read_csv("online_shoppers_clean.csv")

# Girdi ve hedef degiskenlerini ayir
X = veri.drop(columns=["Revenue"])
y = veri["Revenue"].astype(int)

# %70 egitim, %30 gecici veri
X_train, X_gecici, y_train, y_gecici = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Kalan veriyi dogrulama ve test olarak ayir
X_val, X_test, y_val, y_test = train_test_split(
    X_gecici, y_gecici,
    test_size=0.50,
    random_state=42,
    stratify=y_gecici
)

print("=== VERI SETI AYRIMI ===")
print("Egitim:", len(X_train))
print("Dogrulama:", len(X_val))
print("Test:", len(X_test))

print("\n=== SATIN ALMA ORANLARI ===")
print(f"Egitim: %{y_train.mean() * 100:.2f}")
print(f"Dogrulama: %{y_val.mean() * 100:.2f}")
print(f"Test: %{y_test.mean() * 100:.2f}")

print("\n=== KONTROL ===")
print(
    "Toplam kayit:",
    len(X_train) + len(X_val) + len(X_test)
)
print("Girdi degiskeni sayisi:", X_train.shape[1])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Kategorik degiskenler
kategorik_sutunlar = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

# Sayisal degiskenler
sayisal_sutunlar = [
    sutun for sutun in X_train.columns
    if sutun not in kategorik_sutunlar
]

# Veri on isleme yapisi
on_isleme = ColumnTransformer(
    transformers=[
        ("sayisal", StandardScaler(), sayisal_sutunlar),
        (
            "kategorik",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            kategorik_sutunlar
        )
    ]
)

# Yalnizca egitim verisinden donusumleri ogren
X_train_hazir = on_isleme.fit_transform(X_train)

# Dogrulama verisini donustur
X_val_hazir = on_isleme.transform(X_val)

print("\n=== VERI ON ISLEME ===")
print("Sayisal sutun sayisi:", len(sayisal_sutunlar))
print("Kategorik sutun sayisi:", len(kategorik_sutunlar))
print("Egitim verisi boyutu:", X_train_hazir.shape)
print("Dogrulama verisi boyutu:", X_val_hazir.shape)
print("Donusum basariyla tamamlandi.")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Logistic Regression modelini olustur
lojistik_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Modeli yalnizca egitim verileriyle egit
lojistik_model.fit(X_train_hazir, y_train)

# Dogrulama verileriyle tahmin yap
y_val_tahmin = lojistik_model.predict(X_val_hazir)

print("\n=== LOGISTIC REGRESSION SONUCLARI ===")

print(
    f"Accuracy:  "
    f"{accuracy_score(y_val, y_val_tahmin):.4f}"
)
print(
    f"Precision: "
    f"{precision_score(y_val, y_val_tahmin, zero_division=0):.4f}"
)
print(
    f"Recall:    "
    f"{recall_score(y_val, y_val_tahmin, zero_division=0):.4f}"
)
print(
    f"F1-Score:  "
    f"{f1_score(y_val, y_val_tahmin, zero_division=0):.4f}"
)

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_val, y_val_tahmin))
from sklearn.ensemble import RandomForestClassifier

# Random Forest modelini olustur
orman_model = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

# Modeli yalnizca egitim verileriyle egit
orman_model.fit(X_train_hazir, y_train)

# Dogrulama verileriyle tahmin yap
y_val_orman = orman_model.predict(X_val_hazir)

print("\n=== RANDOM FOREST SONUCLARI ===")
print(
    f"Accuracy:  "
    f"{accuracy_score(y_val, y_val_orman):.4f}"
)
print(
    f"Precision: "
    f"{precision_score(y_val, y_val_orman, zero_division=0):.4f}"
)
print(
    f"Recall:    "
    f"{recall_score(y_val, y_val_orman, zero_division=0):.4f}"
)
print(
    f"F1-Score:  "
    f"{f1_score(y_val, y_val_orman, zero_division=0):.4f}"
)

print("\n=== RANDOM FOREST CONFUSION MATRIX ===")
print(confusion_matrix(y_val, y_val_orman))
import matplotlib.pyplot as plt
from pathlib import Path

# Grafik klasorunu hazirla
Path("grafikler").mkdir(exist_ok=True)

# Dogrulama metriklerini hesapla
model_sonuclari = pd.DataFrame({
    "Accuracy": [
        accuracy_score(y_val, y_val_tahmin),
        accuracy_score(y_val, y_val_orman)
    ],
    "Precision": [
        precision_score(y_val, y_val_tahmin),
        precision_score(y_val, y_val_orman)
    ],
    "Recall": [
        recall_score(y_val, y_val_tahmin),
        recall_score(y_val, y_val_orman)
    ],
    "F1-Score": [
        f1_score(y_val, y_val_tahmin),
        f1_score(y_val, y_val_orman)
    ]
}, index=["Logistic Regression", "Random Forest"])

print("\n=== MODEL KARSILASTIRMASI (%) ===")
print((model_sonuclari * 100).round(2).to_string())

# Grafik olustur
ax = (model_sonuclari.T * 100).plot(
    kind="bar",
    figsize=(10, 6),
    rot=0
)

ax.set_title(
    "Makine Ogrenmesi Modellerinin Karsilastirilmasi"
)
ax.set_xlabel("Performans Metrikleri")
ax.set_ylabel("Basari Orani (%)")
ax.set_ylim(0, 105)
ax.legend(title="Modeller")

plt.tight_layout()

plt.savefig(
    "grafikler/04_model_karsilastirma.png",
    dpi=200
)

plt.show()
import seaborn as sns

# Random Forest hata matrisi
hata_matrisi = confusion_matrix(y_val, y_val_orman)

plt.figure(figsize=(7, 5))

sns.heatmap(
    hata_matrisi,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Satin alma yok", "Satin alma var"],
    yticklabels=["Satin alma yok", "Satin alma var"]
)

plt.title("Random Forest - Hata Matrisi")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gercek Durum")

plt.tight_layout()
plt.savefig(
    "grafikler/05_random_forest_hata_matrisi.png",
    dpi=200
)
plt.show()

# Hata sayilarini incele
tn, fp, fn, tp = hata_matrisi.ravel()

print("\n=== RANDOM FOREST HATA ANALIZI ===")
print("Dogru negatif (TN):", tn)
print("Yanlis pozitif (FP):", fp)
print("Yanlis negatif (FN):", fn)
print("Dogru pozitif (TP):", tp)

print(
    f"Kacirilan satin alma orani: "
    f"%{fn / (fn + tp) * 100:.2f}"
)
# ADIM 22 - YANLIS TAHMINLERIN INCELENMESI

# Dogrulama verilerine tahmin etiketlerini ekle
hata_verisi = X_val.copy()
hata_verisi["Gercek"] = y_val.to_numpy()
hata_verisi["Tahmin"] = y_val_orman

# Kacirilan satin almalar
yanlis_negatifler = hata_verisi[
    (hata_verisi["Gercek"] == 1) &
    (hata_verisi["Tahmin"] == 0)
]

# Dogru tahmin edilen satin almalar
dogru_pozitifler = hata_verisi[
    (hata_verisi["Gercek"] == 1) &
    (hata_verisi["Tahmin"] == 1)
]

print("\n=== AYRINTILI HATA ANALIZI ===")
print("Kacirilan satin almalar:", len(yanlis_negatifler))
print("Dogru bulunan satin almalar:", len(dogru_pozitifler))

# Incelenecek ozellikler
ozellikler = [
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues"
]

karsilastirma = pd.DataFrame({
    "Kacirilan_Satin_Almalar":
        yanlis_negatifler[ozellikler].median(),
    "Dogru_Bulunan_Satin_Almalar":
        dogru_pozitifler[ozellikler].median()
})

print("\n=== OZELLIK KARSILASTIRMASI (ORTANCA) ===")
print(karsilastirma.round(3).to_string())
# ADIM 23 - PAGEVALUES OLMADAN RANDOM FOREST

# PageValues degiskenini kaldir
X_train_yeni = X_train.drop(columns=["PageValues"])
X_val_yeni = X_val.drop(columns=["PageValues"])

# Sayisal sutun listesini guncelle
sayisal_yeni = [
    sutun for sutun in sayisal_sutunlar
    if sutun != "PageValues"
]

# Yeni veri on isleme yapisi
on_isleme_yeni = ColumnTransformer(
    transformers=[
        ("sayisal", StandardScaler(), sayisal_yeni),
        (
            "kategorik",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            kategorik_sutunlar
        )
    ]
)

# Yalnizca egitim verisinden ogren
X_train_yeni_hazir = on_isleme_yeni.fit_transform(
    X_train_yeni
)
X_val_yeni_hazir = on_isleme_yeni.transform(
    X_val_yeni
)

# Ayni parametrelerle yeni model
orman_yeni = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

orman_yeni.fit(X_train_yeni_hazir, y_train)
y_val_yeni = orman_yeni.predict(X_val_yeni_hazir)

print("\n=== PAGEVALUES OLMADAN RANDOM FOREST ===")
print(f"Accuracy:  {accuracy_score(y_val, y_val_yeni):.4f}")
print(f"Precision: {precision_score(y_val, y_val_yeni, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_val, y_val_yeni, zero_division=0):.4f}")
print(f"F1-Score:  {f1_score(y_val, y_val_yeni, zero_division=0):.4f}")

print("\n=== YENI CONFUSION MATRIX ===")
print(confusion_matrix(y_val, y_val_yeni))
import numpy as np

# PageValues olmayan modelin satin alma olasiliklari
olasiliklar = orman_yeni.predict_proba(
    X_val_yeni_hazir
)[:, 1]

# Farkli karar esiklerini dene
esikler = np.arange(0.05, 0.51, 0.05)
esik_sonuclari = []

for esik in esikler:
    tahminler = (olasiliklar >= esik).astype(int)

    esik_sonuclari.append({
        "Esik": round(float(esik), 2),
        "Precision": precision_score(
            y_val, tahminler, zero_division=0
        ),
        "Recall": recall_score(
            y_val, tahminler, zero_division=0
        ),
        "F1": f1_score(
            y_val, tahminler, zero_division=0
        )
    })

sonuc_tablosu = pd.DataFrame(esik_sonuclari)

print("\n=== TAHMIN ESIGI KARSILASTIRMASI ===")
print(sonuc_tablosu.round(4).to_string(index=False))

# En yuksek F1 degerine sahip esigi sec
en_iyi = sonuc_tablosu.loc[
    sonuc_tablosu["F1"].idxmax()
]
en_iyi_esik = float(en_iyi["Esik"])

y_val_esik = (
    olasiliklar >= en_iyi_esik
).astype(int)

print("\n=== EN IYI F1 ESIGI ===")
print(f"Esik: {en_iyi_esik:.2f}")
print(f"Precision: {en_iyi['Precision']:.4f}")
print(f"Recall: {en_iyi['Recall']:.4f}")
print(f"F1-Score: {en_iyi['F1']:.4f}")

print("\n=== YENI HATA MATRISI ===")
print(confusion_matrix(y_val, y_val_esik))
# ADIM 25 - NIHAI TEST DEGERLENDIRMESI

# Model 1: PageValues dahil
X_test_hazir = on_isleme.transform(X_test)
test_tahmin_tam = orman_model.predict(X_test_hazir)

# Model 2: PageValues haric, esik 0.25
X_test_yeni = X_test.drop(columns=["PageValues"])

X_test_yeni_hazir = on_isleme_yeni.transform(
    X_test_yeni
)

test_olasilik_yeni = orman_yeni.predict_proba(
    X_test_yeni_hazir
)[:, 1]

test_tahmin_yeni = (
    test_olasilik_yeni >= en_iyi_esik
).astype(int)

def test_sonuclarini_goster(ad, gercek, tahmin):
    print(f"\n=== {ad} ===")
    print(
        f"Accuracy:  "
        f"{accuracy_score(gercek, tahmin):.4f}"
    )
    print(
        f"Precision: "
        f"{precision_score(gercek, tahmin, zero_division=0):.4f}"
    )
    print(
        f"Recall:    "
        f"{recall_score(gercek, tahmin, zero_division=0):.4f}"
    )
    print(
        f"F1-Score:  "
        f"{f1_score(gercek, tahmin, zero_division=0):.4f}"
    )
    print("Confusion Matrix:")
    print(confusion_matrix(gercek, tahmin))

print("\n=== NIHAI TEST DEGERLENDIRMESI ===")

test_sonuclarini_goster(
    "PAGEVALUES DAHIL RANDOM FOREST",
    y_test,
    test_tahmin_tam
)

test_sonuclarini_goster(
    "PAGEVALUES HARIC RANDOM FOREST",
    y_test,
    test_tahmin_yeni
)
from pathlib import Path

# Sonuc klasorunu olustur
Path("sonuclar").mkdir(exist_ok=True)

# Nihai test metriklerini kaydet
kayitlar = []

for model_adi, tahmin in [
    ("Random Forest - PageValues Dahil", test_tahmin_tam),
    ("Random Forest - PageValues Haric", test_tahmin_yeni)
]:
    tn, fp, fn, tp = confusion_matrix(
        y_test, tahmin
    ).ravel()

    kayitlar.append({
        "Model": model_adi,
        "Accuracy": accuracy_score(y_test, tahmin),
        "Precision": precision_score(
            y_test, tahmin, zero_division=0
        ),
        "Recall": recall_score(
            y_test, tahmin, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, tahmin, zero_division=0
        ),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

nihai_tablo = pd.DataFrame(kayitlar)

nihai_tablo.to_csv(
    "sonuclar/nihai_test_sonuclari.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== KAYDEDILEN NIHAI TEST SONUCLARI ===")
print(nihai_tablo.round(4).to_string(index=False))
print("\nSonuc dosyasi basariyla olusturuldu.")