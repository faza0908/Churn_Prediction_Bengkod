import streamlit as st
import pandas as pd
import numpy as np
import joblib, os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor | UDINUS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Model ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'churn_model.pkl')

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model tidak ditemukan di: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

artifacts        = load_model()
model            = artifacts['model']
scaler           = artifacts['scaler']
features         = artifacts['features']          # top 10
feat_importance  = artifacts.get('feature_importance', {})
all_importances  = artifacts.get('all_importances', feat_importance)
le_map           = artifacts['le_map']
model_name       = artifacts.get('model_name', 'Random Forest (Top-10 Features)')
f1_val           = artifacts.get('f1_score', 'N/A')
acc_val          = artifacts.get('accuracy',  'N/A')

# ── CSS ───────────────────────────────────────────────────────────────────────
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
.metric-card h3 { margin:0; color:#1a237e; font-size:.78rem;
                  text-transform:uppercase; letter-spacing:.6px; }
.metric-card p  { margin:.25rem 0 0; color:#1a1a2e; font-size:1.35rem; font-weight:700; }
.result-churn   { background:linear-gradient(135deg,#b71c1c,#ef5350);
                  padding:1.8rem; border-radius:14px; color:#fff; text-align:center; }
.result-nochurn { background:linear-gradient(135deg,#1b5e20,#43a047);
                  padding:1.8rem; border-radius:14px; color:#fff; text-align:center; }
.result-churn h2,.result-nochurn h2 { margin:0; font-size:1.6rem; }
.result-churn p,.result-nochurn p   { margin:.5rem 0 0; opacity:.9; font-size:1rem; }
.fi-card {
    background:#fff; border-radius:12px; padding:1.2rem 1.5rem;
    box-shadow:0 2px 10px rgba(0,0,0,.08); margin-bottom:1rem;
}
.stButton>button {
    background:linear-gradient(135deg,#1a237e,#3949ab);
    color:#fff; border:none; border-radius:8px;
    padding:.65rem 2rem; font-size:1rem; font-weight:600;
    width:100%; transition:all .25s;
}
.stButton>button:hover { opacity:.88; transform:translateY(-1px); }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1> Customer Churn Prediction System</h1>
    <p>UAS Bengkel Koding Data Science — Universitas Dian Nuswantoro Semarang</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Info Model")
    st.markdown(f"""
    <div class="metric-card"><h3>Model</h3>
        <p style="font-size:.95rem">{model_name}</p></div>
    <div class="metric-card"><h3>Accuracy</h3><p>{acc_val}</p></div>
    <div class="metric-card"><h3>F1-Score</h3><p>{f1_val}</p></div>
    <div class="metric-card"><h3>Jumlah Fitur</h3><p>{len(features)} Fitur Terpilih</p></div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.info(
        "Sistem ini memprediksi apakah pelanggan berpotensi **churn** "
        "berdasarkan **10 fitur paling berpengaruh**."
    )
    st.markdown("---")
    with st.expander("📋 10 Fitur Aktif"):
        for i, f in enumerate(features, 1):
            imp = feat_importance.get(f, 0)
            st.markdown(f"`{i:2d}.` **{f}** — `{imp:.4f}`")
    st.markdown("---")
    st.caption("🏫 UDINUS Semarang\nBengkel Koding Data Science")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Prediksi Tunggal",
    "📊 Prediksi Batch (CSV)",
    "📈 Feature Importance",
    "📖 Panduan & Info"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI TUNGGAL
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📝 Masukkan Data Pelanggan")
    st.caption("Hanya 10 fitur paling berpengaruh yang digunakan untuk prediksi.")

    # Metadata tiap fitur: (label, min, max, default, step, tipe)
    feat_meta = {
        'nps_score'               : ("NPS Score",                     -100, 100,   10,   1,    "slider"),
        'pages_per_session'       : ("Halaman per Sesi",               1.0, 50.0,  5.0,  0.5,  "number"),
        'email_open_rate'         : ("Email Open Rate (0–1)",          0.0,  1.0,  0.30, 0.01, "slider"),
        'email_click_rate'        : ("Email Click Rate (0–1)",         0.0,  1.0,  0.10, 0.01, "slider"),
        'age'                     : ("Usia Pelanggan (tahun)",         18,   80,   35,   1,    "number"),
        'marketing_spend_per_user': ("Biaya Marketing per User (USD)", 0.0, 500.0, 20.0, 1.0,  "number"),
        'avg_session_time'        : ("Rata-rata Waktu Sesi (menit)",   0.0, 200.0, 10.0, 0.5,  "number"),
        'total_visits'            : ("Total Kunjungan",                1,   500,   30,   1,    "number"),
        'total_spent'             : ("Total Pengeluaran (USD)",        0.0,10000.0,500.0,10.0, "number"),
        'lifetime_value'          : ("Lifetime Value (USD)",           0.0,50000.0,2000.,100., "number"),
    }

    input_values = {}
    cols = st.columns(2)

    for idx, feat in enumerate(features):
        meta  = feat_meta.get(feat)
        col   = cols[idx % 2]
        label, mn, mx, default, step, widget = meta

        with col:
            if widget == "slider":
                input_values[feat] = st.slider(
                    label, float(mn), float(mx), float(default), float(step),
                    key=f"inp_{feat}")
            else:
                if isinstance(default, float):
                    input_values[feat] = st.number_input(
                        label, float(mn), float(mx), float(default), float(step),
                        key=f"inp_{feat}")
                else:
                    input_values[feat] = st.number_input(
                        label, int(mn), int(mx), int(default), key=f"inp_{feat}")

    st.markdown("---")
    predict_btn = st.button("🔮 Prediksi Churn Sekarang", use_container_width=True)

    if predict_btn:
        input_vec    = np.array([[float(input_values[f]) for f in features]])
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
            st.metric("Prob. Churn",        f"{prob[1]*100:.1f}%")
        with r3:
            st.metric("Prob. Tidak Churn",  f"{prob[0]*100:.1f}%")

        risk = ("🔴 Risiko Tinggi" if prob[1] > 0.65
                else "🟡 Risiko Sedang" if prob[1] > 0.40
                else "🟢 Risiko Rendah")
        st.info(f"**Tingkat Risiko Churn:** {risk}")

        # Gauge bar
        st.markdown("**Probabilitas Churn:**")
        gauge_color = "#e74c3c" if prob[1] > 0.5 else "#f39c12" if prob[1] > 0.35 else "#2ecc71"
        st.markdown(f"""
        <div style="background:#e0e0e0;border-radius:8px;height:22px;width:100%">
            <div style="background:{gauge_color};width:{prob[1]*100:.1f}%;height:100%;
                        border-radius:8px;transition:width .5s;
                        display:flex;align-items:center;justify-content:center;
                        color:white;font-weight:bold;font-size:.85rem">
                {prob[1]*100:.1f}%
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("### 💡 Rekomendasi Tindakan")
        if pred == 1:
            recs = []
            nps  = input_values.get('nps_score', 0)
            eor  = input_values.get('email_open_rate', 0)
            ecr  = input_values.get('email_click_rate', 0)
            lv   = input_values.get('lifetime_value', 0)
            ts   = input_values.get('total_spent', 0)
            tv   = input_values.get('total_visits', 0)
            if nps < 0:
                recs.append("📩 NPS negatif — lakukan outreach personal dan survei kepuasan.")
            if eor < 0.2:
                recs.append("📧 Email open rate rendah — perbaiki subject line & waktu pengiriman.")
            if ecr < 0.05:
                recs.append("🖱️ Email click rate rendah — tingkatkan CTA dan konten email.")
            if lv < 1000:
                recs.append("💎 Lifetime value rendah — tawarkan program loyalitas / upgrade.")
            if tv < 1000:
                recs.append("🛒 Total pengeluaran rendah — kirimkan voucher atau penawaran eksklusif.")
            if tv < 10:
                recs.append("👁️ Kunjungan sangat sedikit — lakukan kampanye re-engagement.")
            if not recs:
                recs.append("📋 Masukkan ke program retensi prioritas segera.")
            for r in recs:
                st.warning(r)
            st.error("🚨 **Tindakan prioritas:** Jadwalkan follow-up dalam 7 hari ke depan!")
        else:
            st.success("👍 Pertahankan kualitas layanan dan lakukan engagement rutin.")
            nps = input_values.get('nps_score', 0)
            lv  = input_values.get('lifetime_value', 0)
            if nps > 50:
                st.success("⭐ NPS tinggi — pelanggan potensial jadi brand ambassador!")
            if lv > 5000:
                st.success("💎 Lifetime value tinggi — pertimbangkan program VIP / reward eksklusif.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDIKSI BATCH
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📤 Upload File CSV untuk Prediksi Massal")
    st.info(f"Kolom yang diperlukan: `{', '.join(features)}`")

    uploaded = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.markdown(f"**File dimuat:** {df_batch.shape[0]:,} baris × {df_batch.shape[1]} kolom")
        with st.expander("👁️ Preview (5 baris pertama)"):
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
                df_res['Prediksi_Churn']     = preds
                df_res['Probabilitas_Churn'] = np.round(probs, 4)
                df_res['Status']             = df_res['Prediksi_Churn'].map(
                                                {0: 'Tidak Churn', 1: 'Churn'})
                df_res['Tingkat_Risiko']     = pd.cut(
                                                probs,
                                                bins=[-0.01, 0.40, 0.65, 1.01],
                                                labels=['Rendah', 'Sedang', 'Tinggi'])

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Data",           f"{len(df_res):,}")
                m2.metric("Prediksi Churn",        f"{preds.sum():,}",
                          delta=f"{preds.mean()*100:.1f}%")
                m3.metric("Prediksi Tidak Churn",  f"{(preds==0).sum():,}")
                m4.metric("Avg Prob. Churn",       f"{probs.mean()*100:.1f}%")

                # Pie distribusi hasil
                fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
                cnt = [int((preds==0).sum()), int(preds.sum())]
                ax_pie.pie(cnt, labels=['Tidak Churn','Churn'],
                           autopct='%1.1f%%', colors=['#2ecc71','#e74c3c'],
                           startangle=90,
                           wedgeprops={'linewidth':2,'edgecolor':'white'})
                ax_pie.set_title('Distribusi Hasil Prediksi Batch', fontweight='bold')
                c_pie1, c_pie2 = st.columns([1, 2])
                with c_pie1:
                    st.pyplot(fig_pie, use_container_width=True)
                with c_pie2:
                    st.dataframe(
                        df_res[['Prediksi_Churn','Probabilitas_Churn',
                                'Status','Tingkat_Risiko']].head(30),
                        use_container_width=True, height=280)

                csv_out = df_res.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Hasil Prediksi (CSV)", csv_out,
                    file_name="hasil_prediksi_churn.csv", mime="text/csv",
                    use_container_width=True)
    else:
        st.markdown("#### 📝 Template CSV")
        sample_df = pd.DataFrame([{f: 0.0 for f in features}])
        st.dataframe(sample_df, use_container_width=True)
        st.download_button(
            "⬇️ Download Template CSV",
            sample_df.to_csv(index=False).encode('utf-8'),
            file_name="template_input_churn.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE IMPORTANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Feature Importance Analysis")
    st.markdown(
        "Analisis kontribusi setiap fitur terhadap prediksi churn "
        "menggunakan **Random Forest Feature Importance**."
    )

    # ── Data importance ────────────────────────────────────────────
    fi_data = pd.Series(all_importances).sort_values(ascending=False)
    fi_df   = fi_data.reset_index()
    fi_df.columns = ['Fitur', 'Importance Score']
    fi_df['Rank']       = range(1, len(fi_df)+1)
    fi_df['Digunakan']  = fi_df['Fitur'].isin(features)
    fi_df['Persen (%)'] = (fi_df['Importance Score'] / fi_df['Importance Score'].sum() * 100).round(2)

    # ── Ringkasan metrik ───────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Fitur Dievaluasi",  len(fi_df))
    m2.metric("Fitur Digunakan Model",   len(features))
    m3.metric("Fitur Tidak Digunakan",   len(fi_df) - len(features))
    top_feat = fi_df.iloc[0]
    m4.metric("Fitur Paling Berpengaruh", top_feat['Fitur'],
              delta=f"{top_feat['Persen (%)']:.1f}% kontribusi")

    st.markdown("---")

    # ── Horizontal bar chart ───────────────────────────────────────
    col_chart, col_info = st.columns([3, 2])

    with col_chart:
        st.markdown("#### 📊 Grafik Feature Importance")

        fig, ax = plt.subplots(figsize=(9, 7))
        bar_colors = ['#3949ab' if used else '#b0bec5'
                      for used in fi_df['Digunakan']]
        bars = ax.barh(fi_df['Fitur'][::-1],
                       fi_df['Importance Score'][::-1],
                       color=bar_colors[::-1], edgecolor='white', height=0.7)

        for bar, val in zip(bars, fi_df['Importance Score'][::-1]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', ha='left', fontsize=8.5,
                    fontweight='bold')

        # Legend
        used_patch   = mpatches.Patch(color='#3949ab', label='✅ Digunakan Model (Top-10)')
        unused_patch = mpatches.Patch(color='#b0bec5', label='❌ Tidak Digunakan')
        ax.legend(handles=[used_patch, unused_patch], loc='lower right', fontsize=9)

        ax.set_xlabel('Importance Score', fontsize=11)
        ax.set_title('Feature Importance — Random Forest\n(Biru = digunakan model)',
                     fontsize=12, fontweight='bold', pad=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(0, fi_df['Importance Score'].max() * 1.18)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col_info:
        st.markdown("#### 🏆 Ranking Fitur")
        # Tabel dengan warna
        display_df = fi_df[['Rank','Fitur','Importance Score','Persen (%)','Digunakan']].copy()
        display_df['Digunakan'] = display_df['Digunakan'].map({True: '✅', False: '❌'})
        st.dataframe(
            display_df.set_index('Rank').style
            .background_gradient(cmap='Blues', subset=['Importance Score'])
            .format({'Importance Score': '{:.4f}', 'Persen (%)': '{:.2f}%'}),
            use_container_width=True, height=420
        )

    st.markdown("---")

    # ── Pie chart kontribusi top-10 ────────────────────────────────
    st.markdown("#### 🥧 Kontribusi Top-10 Fitur terhadap Total Importance")

    top10_fi   = fi_df[fi_df['Digunakan'] == '✅'] if '✅' in fi_df['Digunakan'].values \
                 else fi_df[fi_df['Fitur'].isin(features)]
    other_val  = fi_df[~fi_df['Fitur'].isin(features)]['Importance Score'].sum()

    pie_labels = list(top10_fi['Fitur']) + ['Fitur Lainnya']
    pie_values = list(top10_fi['Importance Score']) + [other_val]
    pie_colors = plt.cm.tab10.colors[:10] + [(0.7, 0.7, 0.7)]

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    wedges, texts, autotexts = ax2.pie(
        pie_values, labels=None, autopct='%1.1f%%',
        colors=pie_colors, startangle=140,
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},
        pctdistance=0.82)
    for at in autotexts:
        at.set_fontsize(8)
        at.set_fontweight('bold')

    ax2.legend(wedges, pie_labels, loc='center left',
               bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    ax2.set_title('Distribusi Kontribusi Feature Importance',
                  fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)

    st.markdown("---")

    # ── Penjelasan tiap fitur top-10 ──────────────────────────────
    st.markdown("#### 💡 Interpretasi Fitur Paling Berpengaruh")

    interpretasi = {
        'nps_score'               : ("🌟 NPS Score",
            "Net Promoter Score mengukur seberapa besar kemungkinan pelanggan "
            "merekomendasikan layanan. NPS rendah/negatif sangat erat kaitannya "
            "dengan risiko churn tinggi."),
        'pages_per_session'       : ("📄 Halaman per Sesi",
            "Semakin banyak halaman yang dikunjungi per sesi, semakin tinggi "
            "tingkat keterlibatan (engagement) pelanggan dengan platform."),
        'email_open_rate'         : ("📧 Email Open Rate",
            "Pelanggan yang sering membuka email cenderung lebih engaged dan "
            "memiliki risiko churn lebih rendah."),
        'email_click_rate'        : ("🖱️ Email Click Rate",
            "Klik pada email menunjukkan ketertarikan aktif terhadap konten "
            "dan penawaran — indikator loyalitas yang kuat."),
        'age'                     : ("👤 Usia Pelanggan",
            "Kelompok usia tertentu memiliki pola churn berbeda. "
            "Analisis segmen usia membantu personalisasi strategi retensi."),
        'marketing_spend_per_user': ("💰 Biaya Marketing per User",
            "Investasi marketing per pelanggan mencerminkan nilai yang "
            "diberikan perusahaan — mempengaruhi ekspektasi dan kepuasan pelanggan."),
        'avg_session_time'        : ("⏱️ Rata-rata Waktu Sesi",
            "Durasi sesi yang panjang menandakan pelanggan aktif menggunakan "
            "platform dan menemukan nilai dari layanan."),
        'total_visits'            : ("🔢 Total Kunjungan",
            "Frekuensi kunjungan mencerminkan kebiasaan penggunaan. "
            "Penurunan kunjungan adalah sinyal awal potensi churn."),
        'total_spent'             : ("💳 Total Pengeluaran",
            "Pelanggan yang telah mengeluarkan lebih banyak uang "
            "umumnya lebih terikat dengan layanan dan kurang cenderung churn."),
        'lifetime_value'          : ("📈 Lifetime Value",
            "LTV yang tinggi menunjukkan pelanggan bernilai tinggi. "
            "Fokuskan upaya retensi pada segmen ini untuk ROI maksimal."),
    }

    cols_interp = st.columns(2)
    for idx, feat in enumerate(features):
        col = cols_interp[idx % 2]
        if feat in interpretasi:
            icon_name, desc = interpretasi[feat]
            imp_score = feat_importance.get(feat, 0)
            pct = imp_score / sum(feat_importance.values()) * 100 if feat_importance else 0
            with col:
                st.markdown(f"""
                <div class="fi-card">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <strong style="font-size:1rem;color:#1a237e">{icon_name}</strong>
                        <span style="background:#e8eaf6;color:#3949ab;padding:.2rem .6rem;
                                     border-radius:20px;font-size:.8rem;font-weight:700">
                            {imp_score:.4f} &nbsp;|&nbsp; {pct:.1f}%
                        </span>
                    </div>
                    <p style="margin:.5rem 0 0;color:#555;font-size:.88rem">{desc}</p>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PANDUAN
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 📖 Panduan Fitur Input")
        feat_guide = {
            'nps_score'               : ("NPS Score (-100–100)",
                "Net Promoter Score. Skor rendah/negatif = pelanggan tidak puas.", "-100 – 100"),
            'pages_per_session'       : ("Halaman per Sesi",
                "Rata-rata jumlah halaman dikunjungi per sesi.", "1 – 50"),
            'email_open_rate'         : ("Email Open Rate",
                "Proporsi email yang dibuka (0=0%, 1=100%).", "0.0 – 1.0"),
            'email_click_rate'        : ("Email Click Rate",
                "Proporsi klik tautan dalam email.", "0.0 – 1.0"),
            'age'                     : ("Usia Pelanggan",
                "Usia pelanggan dalam tahun.", "18 – 80"),
            'marketing_spend_per_user': ("Biaya Marketing per User (USD)",
                "Total pengeluaran marketing untuk pelanggan ini.", "0 – 500"),
            'avg_session_time'        : ("Waktu Sesi (menit)",
                "Rata-rata durasi setiap sesi penggunaan.", "0 – 200"),
            'total_visits'            : ("Total Kunjungan",
                "Jumlah total kunjungan ke platform.", "1 – 500"),
            'total_spent'             : ("Total Pengeluaran (USD)",
                "Akumulasi total pengeluaran pelanggan.", "0 – 10000"),
            'lifetime_value'          : ("Lifetime Value (USD)",
                "Estimasi total nilai pelanggan selama berlangganan.", "0 – 50000"),
        }
        for feat in features:
            if feat in feat_guide:
                name, desc, rng = feat_guide[feat]
                st.markdown(f"**{name}** `{feat}`  \n{desc} | Range: `{rng}`")
                st.markdown("---")

    with col_r:
        st.markdown("### ℹ️ Metodologi & Info Model")
        st.markdown(f"""
        #### 🤖 Model
        **{model_name}**
        - Algoritma: Random Forest Classifier
        - Feature selection: Top-10 dari 15 fitur (berdasarkan importance)
        - Hyperparameter: `n_estimators=50`, `max_depth=10`, `max_features='sqrt'`

        #### 🔬 Pipeline Pemodelan (9 Model Total)
        | Skenario | Model | Keterangan |
        |----------|-------|-----------|
        | Direct | LR, RF, Voting | Tanpa preprocessing |
        | Preprocessing | LR, RF, Voting | Full cleaning + scaling |
        | **Tuned** | **LR, RF, Voting** | **Feature selection + tuning** |

        #### 📊 Dataset
        - **Sumber:** Sales & Marketing Customer Dataset (Kaggle)
        - **Ukuran:** 15.000 records × 30 kolom
        - **Target:** `churn` (0 = tidak churn, 1 = churn)
        - **Balancing:** SMOTE + Random Undersampling

        #### ⚠️ Catatan Penggunaan
        - Hasil bersifat **probabilistik** — gunakan sebagai alat bantu keputusan.
        - Input `0` untuk kolom yang tidak diketahui nilainya.
        - Semakin tinggi probabilitas churn → semakin perlu tindakan retensi.
        """)

        st.markdown("#### 🎯 Threshold Risiko")
        risk_df = pd.DataFrame({
            'Probabilitas Churn': ['< 40%', '40% – 65%', '> 65%'],
            'Kategori': ['🟢 Risiko Rendah', '🟡 Risiko Sedang', '🔴 Risiko Tinggi'],
            'Rekomendasi': ['Maintenance rutin', 'Monitor aktif', 'Intervensi segera']
        })
        st.dataframe(risk_df.set_index('Probabilitas Churn'), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#888;font-size:.82rem;padding:.4rem'>
    🎓 <strong>UAS Bengkel Koding Data Science</strong>
    | Universitas Dian Nuswantoro Semarang<br>
    Model: Random Forest (Top-10 Features)
    | Dataset: Sales & Marketing Customer (15.000 records)
    | Built with ❤️ Streamlit
</div>
""", unsafe_allow_html=True)
