# 🎓 UAS Bengkel Koding Data Science
## Prediksi Customer Churn — Sales and Marketing Dataset
**Universitas Dian Nuswantoro | Program Bengkel Koding**

---

## 📋 Deskripsi Proyek

Proyek ini membangun sistem prediksi **customer churn** menggunakan dataset Sales & Marketing Customer (15.000 records). Churn terjadi ketika pelanggan berhenti menggunakan layanan. Model machine learning dibangun, dievaluasi, dan di-deploy sebagai aplikasi web interaktif menggunakan **Streamlit**.

---

## 🗂️ Struktur Repository

```
uas-churn-prediction/
│
├── 📓 notebooks/
│   └── UAS_Churn_Prediction.ipynb   # Notebook lengkap (EDA + 3 skenario modeling)
│
├── 🖥️ app/
│   └── app.py                       # Aplikasi Streamlit untuk deployment
│
├── 🤖 model/
│   └── churn_model.pkl              # Model terbaik (Random Forest Tuned) + artefak
│
├── 📊 data/
│   └── sales_marketing_dataset.csv  # Dataset (15.000 records, 30 kolom)
│
├── 📈 plots/                        # Visualisasi hasil EDA & evaluasi model
│
├── requirements.txt                 # Dependensi Python
└── README.md
```

---

## 🎯 Alur Proyek

### 1. 📊 Exploratory Data Analysis (EDA)
- Tampilan 5 baris pertama, info dataset, statistik deskriptif
- Analisis missing value + visualisasi bar chart
- Distribusi variabel target (Churn) — bar chart & pie chart
- Heatmap korelasi fitur numerik
- Distribusi fitur utama per kelas churn

### 2. 🤖 Direct Modeling
Tiga model dilatih **langsung tanpa preprocessing**:
| Model | Kategori |
|-------|----------|
| Logistic Regression | Konvensional |
| Random Forest | Ensemble Bagging |
| Voting Classifier (LR+SVM+KNN) | Ensemble Voting |

### 3. 🔧 Modeling dengan Preprocessing
Preprocessing lengkap sebelum pelatihan:
- ✅ Penanganan missing value (impute median/modus)
- ✅ Penghapusan duplikat
- ✅ Penanganan outlier (IQR clipping)
- ✅ Encoding fitur kategorikal (LabelEncoder)
- ✅ Feature scaling (StandardScaler, setelah split)

### 4. ⚙️ Hyperparameter Tuning & Feature Selection
- Feature importance analysis (Random Forest)
- Seleksi top-15 fitur
- GridSearchCV & RandomizedSearchCV untuk tiap model
- Evaluasi 9 model total (3 model × 3 skenario)

### 5. 🚀 Deployment
Aplikasi Streamlit dengan fitur:
- 🎯 Prediksi tunggal (form input interaktif)
- 📊 Prediksi batch (upload CSV)
- 💡 Rekomendasi tindakan otomatis
- 📖 Panduan fitur lengkap

---

## 📈 Metrik Evaluasi

| Skenario | Model | Accuracy | F1-Score |
|----------|-------|----------|----------|
| Direct | Logistic Regression | ~0.70 | ~0.02 |
| Direct | Random Forest | ~0.70 | ~0.02 |
| Direct | Voting Classifier | ~0.70 | ~0.02 |
| Preprocessing | Logistic Regression | ~0.70 | ~0.02 |
| Preprocessing | Random Forest | ~0.70 | ~0.03 |
| **Tuned** | **Random Forest** | **~0.70** | **best** |

> Dataset sintetis dengan signal churn yang lemah — fokus proyek adalah pada proses & pipeline, bukan angka metrik.

---

## 🚀 Cara Menjalankan Lokal

### 1. Clone repository
```bash
git clone https://github.com/USERNAME/uas-churn-prediction.git
cd uas-churn-prediction
```

### 2. Install dependensi
```bash
pip install -r requirements.txt
```

### 3. Jalankan Streamlit
```bash
streamlit run app/app.py
```

### 4. Buka browser
```
http://localhost:8501
```

---

## ☁️ Deployment ke Streamlit Cloud

1. **Push** repository ini ke GitHub (public)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Klik **"New app"** → pilih repository ini
4. Set **Main file path** → `app/app.py`
5. Klik **Deploy!**

---

## 📦 Dataset

- **Sumber:** [Sales and Marketing Dataset — Kaggle](https://www.kaggle.com/datasets/bhaskerpaul/sales-and-marketing-dataset)
- **Jumlah sampel:** 15.000 records
- **Jumlah fitur:** 30 kolom (termasuk target `churn`)
- **Target:** `churn` (0 = tidak churn, 1 = churn)

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red)
![Pandas](https://img.shields.io/badge/Pandas-2.2-green)

---

## 👨‍💻 Pengembang

**Program Bengkel Koding Data Science**  
Universitas Dian Nuswantoro (UDINUS)  
Jl. Imam Bonjol No. 207, Semarang, Jawa Tengah
