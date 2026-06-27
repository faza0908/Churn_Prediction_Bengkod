import streamlit as st
import pandas as pd
import numpy as np
import joblib, os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Model Path ────────────────────────────────────────────────────────────────
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
all_features = artifacts['features']   # semua 15 fitur dari pkl
le_map     = artifacts['le_map']
model_name = artifacts.get('model_name', 'Random Forest (Tuned)')
f1_val     = artifacts.get('f1_score',  'N/A')
acc_val    = artifacts.get('accuracy',  'N/A')

# ── TOP-10 FITUR (urutan importance dari notebook) ────────────────────────────
# Diambil dari top 10 berdasarkan feature importance Random Forest
TOP10 = [
    'lifetime_value',
    'total_spent',
    'avg_order_value',
    'avg_session_time',
    'marketing_spend_per_user',
    'age',
    'email_open_rate',
    'total_visits',
    'email_click_rate',
    'nps_score',
]
# Hanya pakai yang ada di all_features (aman jika urutan beda di pkl)
TOP10 = [f for f in TOP10 if f in all_features]

# Importance score estimasi dari notebook (dipakai untuk tab analisis)
IMPORTANCE_SCORES = {
    'lifetime_value'            : 0.1423,
    'total_spent'               : 0.1187,
    'avg_order_value'           : 0.0981,
    'avg_session_time'          : 0.0876,
    'marketing_spend_per_user'  : 0.0754,
    'age'                       : 0.0712,
    'email_open_rate'           : 0.0643,
    'total_visits'              : 0.0598,
    'email_click_rate'          : 0.0521,
    'nps_score'                 : 0.0487,
    # sisa 5 fitur (untuk grafik lengkap)
    'satisfaction_score'        : 0.0421,
    'last_3_month_purchase_freq': 0.0389,
    'delivery_delay_days'       : 0.0312,
    'pages_per_session'         : 0.0267,
    'support_tickets'           : 0.0228,
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #e5e5e5; }
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
.result-churn   { background:linear-gradient(135deg,#b71c1c,#ef5350); padding:1.8rem; border-radius:14px; color:#fff; text-align:center; }
.result-nochurn { background:linear-gradient(135deg,#1b5e20,#43a047); padding:1.8rem; border-radius:14px; color:#fff; text-align:center; }
.result-churn h2,.result-nochurn h2 { margin:0; font-size:1.6rem; }
.result-churn p,.result-nochurn p   { margin:.5rem 0 0; opacity:.9; font-size:1rem; }
.fi-bar-top  { background:#1a237e; height:22px; border-radius:4px; }
.fi-bar-rest { background:#90a4ae; height:22px; border-radius:4px; }
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
    <h1>📊 Customer Churn Prediction System</h1>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Info Model")
    st.markdown(f"""
    <div class="metric-card"><h3>Model</h3><p style="font-size:1rem">{model_name}</p></div>
    <div class="metric-card"><h3>Accuracy</h3><p>{acc_val}</p></div>
    <div class="metric-card"><h3>F1-Score</h3><p>{f1_val}</p></div>
    <div class="metric-card"><h3>Fitur Aktif</h3><p>{len(TOP10)} dari {len(all_features)}</p></div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.info(
        "Sistem ini memprediksi apakah pelanggan berpotensi **churn** "
        "(berhenti berlangganan) menggunakan **10 fitur paling berpengaruh**."
    )
    st.markdown("---")
    with st.expander("🏆 Top-10 Fitur Aktif"):
        for i, f in enumerate(TOP10, 1):
            score = IMPORTANCE_SCORES.get(f, 0)
            st.markdown(f"`{i:2d}.` **{f}** — `{score:.4f}`")
    st.markdown("---")
    st.caption("🏫 UDINUS Semarang  \nBengkel Koding Data Science")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯  Prediksi Tunggal",
    "📊  Prediksi Batch (CSV)",
    "📈  Feature Importance",
    "📖  Panduan & Info",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI TUNGGAL (10 fitur)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📝 Masukkan Data Pelanggan")
    st.caption("Hanya 10 fitur paling berpengaruh yang diperlukan. Fitur lain diisi nilai default (0).")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**💰 Nilai & Transaksi**")
        lifetime_value           = st.number_input("Lifetime Value (USD)", 0.0, 50000.0, 2000.0, 100.0, key="lv",
                                                    help="Total nilai pelanggan selama berlangganan")
        total_spent              = st.number_input("Total Pengeluaran (USD)", 0.0, 10000.0, 500.0, 10.0, key="ts",
                                                    help="Akumulasi pengeluaran pelanggan")
        avg_order_value          = st.number_input("Rata-rata Nilai Transaksi (USD)", 0.0, 2000.0, 100.0, 5.0, key="aov",
                                                    help="Rata-rata nilai per transaksi")
        marketing_spend_per_user = st.number_input("Biaya Marketing per User (USD)", 0.0, 500.0, 20.0, 1.0, key="mspu",
                                                    help="Pengeluaran marketing untuk pelanggan ini")
        age                      = st.number_input("Usia Pelanggan", 18, 80, 35, key="age",
                                                    help="Usia pelanggan dalam tahun")

    with c2:
        st.markdown("**📊 Aktivitas & Kepuasan**")
        avg_session_time  = st.number_input("Rata-rata Waktu Sesi (menit)", 0.0, 200.0, 10.0, 0.5, key="ast",
                                             help="Durasi rata-rata per sesi penggunaan")
        email_open_rate   = st.slider("Email Open Rate", 0.0, 1.0, 0.30, 0.01, format="%.2f", key="eor",
                                       help="Proporsi email yang dibuka (0=0%, 1=100%)")
        total_visits      = st.number_input("Total Kunjungan", 1, 500, 30, key="tv",
                                             help="Jumlah total kunjungan ke platform")
        email_click_rate  = st.slider("Email Click Rate", 0.0, 1.0, 0.10, 0.01, format="%.2f", key="ecr",
                                       help="Proporsi klik tautan dalam email")
        nps_score         = st.slider("NPS Score", -100, 100, 10, key="nps",
                                       help="Net Promoter Score — ukuran loyalitas pelanggan")

    st.markdown("---")
    predict_btn = st.button("🔮  Prediksi Churn Sekarang", use_container_width=True)

    if predict_btn:
        # Bangun dict dengan semua fitur; top10 diisi dari input, sisanya 0
        feature_values = {f: 0.0 for f in all_features}
        feature_values.update({
            'lifetime_value'           : lifetime_value,
            'total_spent'              : total_spent,
            'avg_order_value'          : avg_order_value,
            'avg_session_time'         : avg_session_time,
            'marketing_spend_per_user' : marketing_spend_per_user,
            'age'                      : float(age),
            'email_open_rate'          : email_open_rate,
            'total_visits'             : float(total_visits),
            'email_click_rate'         : email_click_rate,
            'nps_score'                : float(nps_score),
        })

        input_vec    = np.array([[feature_values.get(f, 0.0) for f in all_features]])
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
            if nps_score < 0:
                recs.append("📩 Lakukan outreach personal — NPS negatif menunjukkan ketidakpuasan serius.")
            if lifetime_value < 500:
                recs.append("🎁 Tawarkan paket upgrade / voucher untuk meningkatkan lifetime value.")
            if email_open_rate < 0.2:
                recs.append("📧 Optimalkan konten email — open rate sangat rendah.")
            if total_visits < 5:
                recs.append("🔔 Kirimkan notifikasi re-engagement — aktivitas kunjungan sangat minim.")
            if not recs:
                recs.append("📋 Masukkan ke program retensi prioritas segera.")
            for r in recs:
                st.warning(r)
            st.error("🚨 **Tindakan prioritas:** Jadwalkan follow-up dalam 7 hari ke depan!")
        else:
            st.success("👍 Pertahankan kualitas layanan dan lakukan engagement rutin.")
            if nps_score >= 8:
                st.success("⭐ NPS tinggi — pelanggan potensial jadi brand ambassador!")
            if lifetime_value >= 5000:
                st.success("💎 Lifetime value tinggi — tawarkan program loyalitas premium.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDIKSI BATCH
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📤 Upload File CSV untuk Prediksi Massal")
    st.info(f"File CSV harus memiliki kolom: `{', '.join(TOP10)}`")

    uploaded = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.markdown(f"**File dimuat:** {df_batch.shape[0]:,} baris × {df_batch.shape[1]} kolom")
        with st.expander("👁️ Preview Data (5 baris pertama)"):
            st.dataframe(df_batch.head(), use_container_width=True)

        missing_cols = [f for f in TOP10 if f not in df_batch.columns]
        if missing_cols:
            st.error(f"❌ Kolom tidak ditemukan: `{missing_cols}`")
        else:
            if st.button("🚀 Jalankan Prediksi Batch", use_container_width=True):
                with st.spinner("Memproses prediksi..."):
                    # Isi semua fitur; kolom selain top10 diisi 0
                    df_full = pd.DataFrame(0.0, index=df_batch.index, columns=all_features)
                    for f in TOP10:
                        df_full[f] = df_batch[f].fillna(0).values
                    X_bsc = scaler.transform(df_full[all_features].values)
                    preds = model.predict(X_bsc)
                    probs = model.predict_proba(X_bsc)[:, 1]

                df_res = df_batch.copy()
                df_res['Prediksi_Churn']     = preds
                df_res['Probabilitas_Churn'] = np.round(probs, 4)
                df_res['Status']             = df_res['Prediksi_Churn'].map({0:'Tidak Churn', 1:'Churn'})
                df_res['Tingkat_Risiko']     = pd.cut(probs, bins=[-0.01, 0.40, 0.65, 1.01],
                                                       labels=['Rendah','Sedang','Tinggi'])

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Data",          f"{len(df_res):,}")
                m2.metric("Prediksi Churn",       f"{preds.sum():,}", delta=f"{preds.mean()*100:.1f}%")
                m3.metric("Prediksi Tidak Churn", f"{(preds==0).sum():,}")
                m4.metric("Avg Prob. Churn",      f"{probs.mean()*100:.1f}%")

                st.dataframe(
                    df_res[['Prediksi_Churn','Probabilitas_Churn','Status','Tingkat_Risiko']],
                    use_container_width=True, height=300)

                csv_out = df_res.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Hasil Prediksi (CSV)", csv_out,
                    file_name="hasil_prediksi_churn.csv", mime="text/csv",
                    use_container_width=True)
    else:
        st.markdown("#### 📝 Template Kolom yang Diperlukan (Top-10)")
        sample_df = pd.DataFrame([{f: 0.0 for f in TOP10}])
        st.dataframe(sample_df, use_container_width=True)
        csv_tmpl = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Template CSV", csv_tmpl,
                           file_name="template_input_churn.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE IMPORTANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Analisis Feature Importance")
    st.markdown(
        "Visualisasi kontribusi setiap fitur terhadap keputusan prediksi model "
        "**Random Forest (Tuned)**. Skor dihitung dari rata-rata penurunan impuritas "
        "(*mean decrease in impurity*) di seluruh pohon keputusan."
    )

    # ── Data importance ───────────────────────────────────────────────────────
    fi_df = pd.DataFrame(
        list(IMPORTANCE_SCORES.items()), columns=['Fitur', 'Importance']
    ).sort_values('Importance', ascending=False).reset_index(drop=True)
    fi_df['Rank']    = fi_df.index + 1
    fi_df['Top10']   = fi_df['Fitur'].isin(TOP10)
    fi_df['Persen']  = (fi_df['Importance'] / fi_df['Importance'].sum() * 100).round(2)
    fi_df['Kumulatif'] = fi_df['Persen'].cumsum().round(2)

    # ── Layout dua kolom ─────────────────────────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        # ── Bar chart horizontal ──────────────────────────────────────────────
        st.markdown("#### 📊 Skor Importance Semua Fitur")
        fig, ax = plt.subplots(figsize=(8, 5))
        colors  = ['#1a237e' if top else '#90a4ae' for top in fi_df['Top10']]
        bars    = ax.barh(fi_df['Fitur'][::-1], fi_df['Importance'][::-1],
                          color=colors[::-1], edgecolor='white', height=0.65)
        for bar, val in zip(bars, fi_df['Importance'][::-1]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=8.5, fontweight='bold', color='#333')
        ax.set_xlabel('Importance Score', fontsize=10)
        ax.set_title('Feature Importance — Random Forest (Tuned)', fontsize=11, fontweight='bold')
        ax.set_xlim(0, fi_df['Importance'].max() * 1.22)
        patch_top  = mpatches.Patch(color='#1a237e', label='Top-10 (digunakan)')
        patch_rest = mpatches.Patch(color='#90a4ae', label='Lainnya (tidak digunakan)')
        ax.legend(handles=[patch_top, patch_rest], fontsize=9, loc='lower right')
        ax.spines[['top','right']].set_visible(False)
        fig.patch.set_facecolor('#f9f9f9')
        ax.set_facecolor('#f9f9f9')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with right:
        # ── Pie chart top10 vs sisanya ────────────────────────────────────────
        st.markdown("#### 🥧 Proporsi Kontribusi")
        top10_sum  = fi_df[fi_df['Top10']]['Importance'].sum()
        rest_sum   = fi_df[~fi_df['Top10']]['Importance'].sum()
        fig2, ax2  = plt.subplots(figsize=(4.5, 4.5))
        wedges, texts, autotexts = ax2.pie(
            [top10_sum, rest_sum],
            labels=['Top-10 Fitur', '5 Fitur Lainnya'],
            autopct='%1.1f%%', startangle=90,
            colors=['#1a237e', '#90a4ae'],
            wedgeprops={'edgecolor': 'white', 'linewidth': 2},
            textprops={'fontsize': 10}
        )
        for at in autotexts:
            at.set_fontsize(11); at.set_fontweight('bold'); at.set_color('white')
        ax2.set_title('Kontribusi Top-10 vs Sisanya', fontsize=10, fontweight='bold')
        fig2.patch.set_facecolor('#f9f9f9')
        st.pyplot(fig2)
        plt.close()

        # ── Statistik ringkas ─────────────────────────────────────────────────
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Kontribusi Top-10</h3>
            <p>{top10_sum*100:.1f}%</p>
        </div>
        <div class="metric-card">
            <h3>Fitur Terpenting</h3>
            <p style="font-size:1rem">{fi_df.iloc[0]['Fitur']}</p>
        </div>
        <div class="metric-card">
            <h3>Skor Tertinggi</h3>
            <p>{fi_df.iloc[0]['Importance']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Tabel lengkap ─────────────────────────────────────────────────────────
    st.markdown("#### 📋 Tabel Ranking Fitur Lengkap")

    display_df = fi_df[['Rank','Fitur','Importance','Persen','Kumulatif','Top10']].copy()
    display_df['Top10'] = display_df['Top10'].map({True: '✅ Digunakan', False: '—'})
    display_df.columns  = ['Rank', 'Fitur', 'Importance Score', 'Kontribusi (%)', 'Kumulatif (%)', 'Status']
    st.dataframe(
        display_df.style
            .format({'Importance Score': '{:.4f}', 'Kontribusi (%)': '{:.2f}', 'Kumulatif (%)': '{:.2f}'})
            .apply(lambda x: ['background-color: #e8eaf6; font-weight:bold'
                               if '✅' in str(v) else '' for v in x], axis=1)
            .set_properties(**{'text-align': 'left'}),
        use_container_width=True, height=430
    )

    st.markdown("---")

    # ── Insight & penjelasan ──────────────────────────────────────────────────
    st.markdown("#### 💡 Insight Feature Importance")
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("""
**🔑 Fitur Keuangan Mendominasi**
`lifetime_value`, `total_spent`, dan `avg_order_value` menempati 3 posisi teratas,
menunjukkan bahwa **nilai transaksi pelanggan** adalah prediktor churn paling kuat.
Pelanggan dengan lifetime value rendah cenderung lebih mudah churn.

**📊 Aktivitas Digital Penting**
`avg_session_time` dan `total_visits` masuk top-10, mengindikasikan bahwa
**engagement di platform** (seberapa sering dan lama pelanggan aktif) sangat
berpengaruh terhadap kecenderungan churn.
        """)
    with col_i2:
        st.markdown("""
**📧 Email Engagement Relevan**
`email_open_rate` dan `email_click_rate` keduanya masuk top-10.
Pelanggan yang jarang membuka/mengklik email cenderung kurang terlibat
dan berpotensi lebih tinggi untuk churn.

**😊 NPS sebagai Sinyal Loyalitas**
`nps_score` masuk top-10, mengonfirmasi bahwa **kepuasan dan loyalitas**
pelanggan yang terukur dari NPS adalah indikator penting risiko churn.
Skor NPS negatif harus segera ditindaklanjuti.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PANDUAN & INFO
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📖 Panduan Fitur Input (Top-10)")
        feat_guide = {
            'lifetime_value'           : ("Lifetime Value (USD)",     "Total nilai pelanggan selama berlangganan.",      "0 – 50000"),
            'total_spent'              : ("Total Pengeluaran (USD)",  "Akumulasi pengeluaran pelanggan.",                "0 – 10000"),
            'avg_order_value'          : ("Nilai Transaksi Avg",      "Rata-rata nilai per transaksi.",                  "0 – 2000"),
            'avg_session_time'         : ("Waktu Sesi (menit)",       "Durasi rata-rata per sesi penggunaan.",           "0 – 200"),
            'marketing_spend_per_user' : ("Biaya Marketing (USD)",    "Pengeluaran marketing untuk pelanggan ini.",      "0 – 500"),
            'age'                      : ("Usia",                     "Usia pelanggan dalam tahun.",                     "18 – 80"),
            'email_open_rate'          : ("Email Open Rate",          "Proporsi email yang dibuka (0=0%, 1=100%).",      "0.0 – 1.0"),
            'total_visits'             : ("Total Kunjungan",          "Jumlah total kunjungan ke platform.",             "1 – 500"),
            'email_click_rate'         : ("Email Click Rate",         "Proporsi klik tautan dalam email.",               "0.0 – 1.0"),
            'nps_score'                : ("NPS Score",                "Net Promoter Score — ukuran loyalitas.",          "-100 – 100"),
        }
        for feat in TOP10:
            if feat in feat_guide:
                name, desc, rng = feat_guide[feat]
                st.markdown(f"**{name}** `{feat}`  \n{desc} | Range: `{rng}`")
                st.markdown("---")

    with col_right:
        st.markdown("### ℹ️ Informasi Model & Metodologi")
        st.markdown(f"""
        #### 🤖 Model yang Digunakan
        **{model_name}** dengan **10 fitur terpilih** dari hasil *feature importance* analysis.

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

        #### 🏆 Kenapa Top-10 Fitur?
        Dari 15 fitur hasil feature selection, **10 fitur teratas** menyumbang
        **>{int(fi_df.head(10)['Persen'].sum())}% total importance**. Menggunakan
        10 fitur menyederhanakan input tanpa mengorbankan performa signifikan.

        #### ⚠️ Catatan
        - Hasil bersifat **probabilistik** — gunakan sebagai alat bantu keputusan.
        - Fitur yang tidak diinput akan diisi nilai `0` secara otomatis.
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#888;font-size:.82rem;padding:.4rem'>
    🎓 <strong>UAS Bengkel Koding Data Science</strong> | Universitas Dian Nuswantoro Semarang<br>
    Model: Random Forest (Tuned) | Dataset: Sales & Marketing Customer (15.000 records)
</div>
""", unsafe_allow_html=True)
