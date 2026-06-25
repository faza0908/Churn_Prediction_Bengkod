import streamlit as st
import pandas as pd
import numpy as np
import joblib, os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor | UDINUS",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Model Path — mencari churn_model.pkl di folder yang sama dengan file ini ──
# Ini bekerja baik di lokal maupun Streamlit Cloud (flat structure)
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'churn_model.pkl')

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model tidak ditemukan di: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

artifacts  = load_model()
model      = artifacts['model']
scaler     = artifacts['scaler']
features   = artifacts['features']
le_map     = artifacts['le_map']
model_name = artifacts.get('model_name', 'Random Forest (Tuned)')
f1_val     = artifacts.get('f1_score',  'N/A')
acc_val    = artifacts.get('accuracy',  'N/A')

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f4f6fb; }
.main-header {
    background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
    padding: 2rem 2.5rem; border-radius: 14px; margin-bottom: 1.8rem;
    box-shadow: 0 6px 20px rgba(26,35,126,.35);
}
.main-header h1 { color:#fff; margin:0; font-size:2.1rem; font-weight:700; }
.main-header p  { color:#c5cae9; margin:.4rem 0 0; font-size:1rem; }
.metric-card {
    background:#fff; padding:1.1rem 1.3rem; border-radius:10px;
    border-left:4px solid #3949ab;
    box-shadow:0 2px 8px rgba(0,0,0,.07); margin-bottom:.7rem;
}
.metric-card h3 { margin:0; color:#1a237e; font-size:.78rem; text-transform:uppercase; letter-spacing:.6px; }
.metric-card p  { margin:.25rem 0 0; color:#1a1a2e; font-size:1.35rem; font-weight:700; }
.result-churn    { background:linear-gradient(135deg,#b71c1c,#ef5350); padding:1.8rem; border-radius:14px; color:#fff; text-align:center; }
.result-nochurn  { background:linear-gradient(135deg,#1b5e20,#43a047); padding:1.8rem; border-radius:14px; color:#fff; text-align:center; }
.result-churn h2,.result-nochurn h2 { margin:0; font-size:1.6rem; }
.result-churn p,.result-nochurn p   { margin:.5rem 0 0; opacity:.9; font-size:1rem; }
.stButton>button {
    background:linear-gradient(135deg,#1a237e,#3949ab);
    color:#fff; border:none; border-radius:8px;
    padding:.65rem 2rem; font-size:1rem; font-weight:600; width:100%;
    transition:all .25s;
}
.stButton>button:hover { opacity:.88; transform:translateY(-1px); }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔮 Customer Churn Prediction System</h1>
    <p>UAS Bengkel Koding Data Science — Universitas Dian Nuswantoro Semarang</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Info Model")
    st.markdown(f"""
    <div class="metric-card"><h3>Model</h3><p style="font-size:1rem">{model_name}</p></div>
    <div class="metric-card"><h3>Accuracy</h3><p>{acc_val}</p></div>
    <div class="metric-card"><h3>F1-Score</h3><p>{f1_val}</p></div>
    <div class="metric-card"><h3>Jumlah Fitur</h3><p>{len(features)}</p></div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.info(
        "Sistem ini memprediksi apakah pelanggan berpotensi **churn** "
        "(berhenti berlangganan) berdasarkan data perilaku dan transaksi."
    )
    st.markdown("---")
    with st.expander("📋 Daftar Fitur"):
        for i, f in enumerate(features, 1):
            st.markdown(f"`{i:2d}.` {f}")
    st.markdown("---")
    st.caption("🏫 UDINUS Semarang  \nBengkel Koding Data Science")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯  Prediksi Tunggal", "📊  Prediksi Batch (CSV)", "📖  Panduan & Info"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI TUNGGAL
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📝 Masukkan Data Pelanggan")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**📊 Aktivitas Digital**")
        avg_session_time           = st.number_input("Rata-rata Waktu Sesi (menit)", 0.0, 200.0, 10.0, 0.5, key="ast")
        pages_per_session          = st.number_input("Halaman per Sesi", 1.0, 50.0, 5.0, 0.5, key="pps")
        total_visits               = st.number_input("Total Kunjungan", 1, 500, 30, key="tv")
        email_open_rate            = st.slider("Email Open Rate", 0.0, 1.0, 0.30, 0.01, format="%.2f", key="eor")
        email_click_rate           = st.slider("Email Click Rate", 0.0, 1.0, 0.10, 0.01, format="%.2f", key="ecr")

    with c2:
        st.markdown("**💰 Transaksi & Nilai**")
        total_spent                = st.number_input("Total Pengeluaran (USD)", 0.0, 10000.0, 500.0, 10.0, key="ts")
        avg_order_value            = st.number_input("Rata-rata Nilai Transaksi (USD)", 0.0, 2000.0, 100.0, 5.0, key="aov")
        lifetime_value             = st.number_input("Lifetime Value (USD)", 0.0, 50000.0, 2000.0, 100.0, key="lv")
        last_3_month_purchase_freq = st.number_input("Frek. Pembelian 3 Bulan Terakhir", 0, 50, 3, key="lmpf")
        marketing_spend_per_user   = st.number_input("Biaya Marketing per User (USD)", 0.0, 500.0, 20.0, 1.0, key="mspu")

    with c3:
        st.markdown("**👤 Profil & Kepuasan**")
        age                 = st.number_input("Usia Pelanggan", 18, 80, 35, key="age")
        satisfaction_score  = st.slider("Skor Kepuasan (1–5)", 1.0, 5.0, 3.5, 0.1, key="ss")
        nps_score           = st.slider("NPS Score", -100, 100, 10, key="nps")
        support_tickets     = st.number_input("Jumlah Tiket Support", 0, 20, 1, key="st_")
        delivery_delay_days = st.number_input("Keterlambatan Pengiriman (hari)", 0, 30, 2, key="ddd")

    st.markdown("---")
    predict_btn = st.button("🔮  Prediksi Churn Sekarang", use_container_width=True)

    if predict_btn:
        feature_values = {
            'avg_session_time'           : avg_session_time,
            'pages_per_session'          : pages_per_session,
            'marketing_spend_per_user'   : marketing_spend_per_user,
            'age'                        : float(age),
            'email_open_rate'            : email_open_rate,
            'total_visits'               : float(total_visits),
            'email_click_rate'           : email_click_rate,
            'nps_score'                  : float(nps_score),
            'satisfaction_score'         : satisfaction_score,
            'lifetime_value'             : lifetime_value,
            'total_spent'                : total_spent,
            'avg_order_value'            : avg_order_value,
            'last_3_month_purchase_freq' : float(last_3_month_purchase_freq),
            'delivery_delay_days'        : float(delivery_delay_days),
            'support_tickets'            : float(support_tickets),
        }
        input_vec    = np.array([[feature_values.get(f, 0.0) for f in features]])
        input_scaled = scaler.transform(input_vec)
        pred         = model.predict(input_scaled)[0]
        prob         = model.predict_proba(input_scaled)[0]

        st.markdown("---")
        st.markdown("### 📊 Hasil Prediksi")

        r1, r2, r3 = st.columns([2, 1, 1])
        with r1:
            if pred == 1:
                st.markdown(f"""
                <div class="result-churn">
                    <h2>⚠️ PELANGGAN BERPOTENSI CHURN</h2>
                    <p>Pelanggan ini diprediksi akan <strong>berhenti</strong> menggunakan layanan.</p>
                    <p style="font-size:1.25rem;margin-top:.8rem">
                        Probabilitas Churn: <strong>{prob[1]*100:.1f}%</strong>
                    </p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-nochurn">
                    <h2>✅ PELANGGAN TIDAK CHURN</h2>
                    <p>Pelanggan ini diprediksi akan <strong>tetap</strong> menggunakan layanan.</p>
                    <p style="font-size:1.25rem;margin-top:.8rem">
                        Probabilitas Tidak Churn: <strong>{prob[0]*100:.1f}%</strong>
                    </p>
                </div>""", unsafe_allow_html=True)
        with r2:
            st.metric("Prob. Churn",       f"{prob[1]*100:.1f}%")
        with r3:
            st.metric("Prob. Tidak Churn", f"{prob[0]*100:.1f}%")

        if prob[1] > 0.65:
            risk_label = "🔴 Risiko Tinggi"
        elif prob[1] > 0.40:
            risk_label = "🟡 Risiko Sedang"
        else:
            risk_label = "🟢 Risiko Rendah"
        st.info(f"**Tingkat Risiko Churn:** {risk_label}")

        st.markdown("### 💡 Rekomendasi Tindakan")
        if pred == 1:
            recs = []
            if satisfaction_score < 3.0:
                recs.append("📞 Hubungi pelanggan dan lakukan survei kepuasan mendalam.")
            if support_tickets > 3:
                recs.append("🛠️ Tindak lanjuti tiket support yang belum terselesaikan.")
            if last_3_month_purchase_freq == 0:
                recs.append("🎁 Kirim penawaran eksklusif / voucher reaktivasi.")
            if nps_score < 0:
                recs.append("📩 Lakukan outreach personal untuk meningkatkan kepuasan.")
            if delivery_delay_days > 5:
                recs.append("🚚 Evaluasi proses pengiriman — keterlambatan tinggi.")
            if not recs:
                recs.append("📋 Masukkan ke program retensi prioritas segera.")
            for r in recs:
                st.warning(r)
            st.error("🚨 **Tindakan prioritas:** Jadwalkan follow-up dalam 7 hari ke depan!")
        else:
            st.success("👍 Pertahankan kualitas layanan dan lakukan engagement rutin.")
            if satisfaction_score >= 4.5:
                st.success("⭐ Pelanggan potensial jadi brand ambassador — tawarkan program referral!")
            if last_3_month_purchase_freq >= 5:
                st.success("🛍️ Pelanggan aktif — tawarkan program loyalitas / upgrade langganan.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDIKSI BATCH
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📤 Upload File CSV untuk Prediksi Massal")
    st.info(f"File CSV harus memiliki kolom: `{', '.join(features)}`")

    uploaded = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.markdown(f"**File dimuat:** {df_batch.shape[0]:,} baris × {df_batch.shape[1]} kolom")
        with st.expander("👁️ Preview Data (5 baris pertama)"):
            st.dataframe(df_batch.head(), use_container_width=True)

        missing_cols = [f for f in features if f not in df_batch.columns]
        if missing_cols:
            st.error(f"❌ Kolom tidak ditemukan: `{missing_cols}`")
        else:
            if st.button("🚀 Jalankan Prediksi Batch", use_container_width=True):
                with st.spinner("Memproses prediksi..."):
                    X_b   = df_batch[features].fillna(0).values
                    X_bsc = scaler.transform(X_b)
                    preds = model.predict(X_bsc)
                    probs = model.predict_proba(X_bsc)[:, 1]

                df_res = df_batch.copy()
                df_res['Prediksi_Churn']    = preds
                df_res['Probabilitas_Churn']= np.round(probs, 4)
                df_res['Status']            = df_res['Prediksi_Churn'].map({0:'Tidak Churn', 1:'Churn'})
                df_res['Tingkat_Risiko']    = pd.cut(probs, bins=[-0.01, 0.40, 0.65, 1.01],
                                                      labels=['Rendah','Sedang','Tinggi'])

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Data",           f"{len(df_res):,}")
                m2.metric("Prediksi Churn",        f"{preds.sum():,}", delta=f"{preds.mean()*100:.1f}%")
                m3.metric("Prediksi Tidak Churn",  f"{(preds==0).sum():,}")
                m4.metric("Avg Prob. Churn",       f"{probs.mean()*100:.1f}%")

                st.dataframe(
                    df_res[['Prediksi_Churn','Probabilitas_Churn','Status','Tingkat_Risiko']],
                    use_container_width=True, height=300)

                csv_out = df_res.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Hasil Prediksi (CSV)", csv_out,
                    file_name="hasil_prediksi_churn.csv", mime="text/csv",
                    use_container_width=True)
    else:
        st.markdown("#### 📝 Template Kolom yang Diperlukan")
        sample_df = pd.DataFrame([{f: 0.0 for f in features}])
        st.dataframe(sample_df, use_container_width=True)
        csv_tmpl = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Template CSV", csv_tmpl,
                           file_name="template_input_churn.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PANDUAN
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📖 Panduan Fitur Input")
        feat_guide = {
            'avg_session_time'           : ("Waktu Sesi (menit)",      "Durasi rata-rata per sesi penggunaan.",          "0 – 200"),
            'pages_per_session'          : ("Halaman per Sesi",        "Halaman rata-rata yang dikunjungi per sesi.",    "1 – 50"),
            'marketing_spend_per_user'   : ("Biaya Marketing (USD)",   "Pengeluaran marketing untuk pelanggan ini.",     "0 – 500"),
            'age'                        : ("Usia",                    "Usia pelanggan dalam tahun.",                    "18 – 80"),
            'email_open_rate'            : ("Email Open Rate",         "Proporsi email yang dibuka (0=0%, 1=100%).",     "0.0 – 1.0"),
            'total_visits'               : ("Total Kunjungan",         "Jumlah total kunjungan ke platform.",            "1 – 500"),
            'email_click_rate'           : ("Email Click Rate",        "Proporsi klik tautan dalam email.",              "0.0 – 1.0"),
            'nps_score'                  : ("NPS Score",               "Net Promoter Score — ukuran loyalitas.",         "-100 – 100"),
            'satisfaction_score'         : ("Skor Kepuasan",           "Penilaian kepuasan layanan (1=buruk,5=baik).",   "1.0 – 5.0"),
            'lifetime_value'             : ("Lifetime Value (USD)",    "Total nilai pelanggan selama berlangganan.",     "0 – 50000"),
            'total_spent'                : ("Total Pengeluaran (USD)", "Akumulasi pengeluaran pelanggan.",               "0 – 10000"),
            'avg_order_value'            : ("Nilai Transaksi Avg",     "Rata-rata nilai per transaksi.",                 "0 – 2000"),
            'last_3_month_purchase_freq' : ("Frek. Beli 3 Bulan",     "Jumlah pembelian dalam 3 bulan terakhir.",       "0 – 50"),
            'delivery_delay_days'        : ("Keterlambatan (hari)",    "Rata-rata keterlambatan pengiriman.",            "0 – 30"),
            'support_tickets'            : ("Tiket Support",           "Total tiket dukungan yang pernah dibuat.",       "0 – 20"),
        }
        for feat in features:
            if feat in feat_guide:
                name, desc, rng = feat_guide[feat]
                st.markdown(f"**{name}** `{feat}`  \n{desc} | Range: `{rng}`")
                st.markdown("---")

    with col_right:
        st.markdown("### ℹ️ Informasi Model & Metodologi")
        st.markdown(f"""
        #### 🤖 Model yang Digunakan
        **{model_name}** dengan **{len(features)} fitur** terpilih dari *feature importance* analysis.

        #### 🔬 Skenario Pemodelan (9 Model Total)
        | Skenario | Keterangan |
        |----------|-----------|
        | Direct | Tanpa preprocessing — baseline |
        | Preprocessing | Cleaning + encoding + scaling |
        | **Tuned** | Feature selection + hyperparameter tuning ← *Terbaik* |

        #### 📊 Dataset
        - **Sumber:** Sales & Marketing Customer Dataset (Kaggle)
        - **Ukuran:** 15.000 records × 30 kolom
        - **Target:** `churn` (0 = tidak churn, 1 = churn)

        #### ⚠️ Catatan
        - Hasil bersifat **probabilistik** — gunakan sebagai alat bantu keputusan.
        - Isi `0` untuk kolom yang tidak diketahui nilainya.
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#888;font-size:.82rem;padding:.4rem'>
    🎓 <strong>UAS Bengkel Koding Data Science</strong> | Universitas Dian Nuswantoro Semarang<br>
    Model: Random Forest (Tuned) | Dataset: Sales & Marketing Customer (15.000 records)
</div>
""", unsafe_allow_html=True)
