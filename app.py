import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from io import BytesIO

# =========================
# OPTIONAL PDF SUPPORT (SAFE)
# =========================
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    PDF_AVAILABLE = True
except:
    PDF_AVAILABLE = False

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Customer Segmentation Engine",
    page_icon="🧠",
    layout="wide"
)

# =========================
# GLASS + GRADIENT + ANIMATION STYLES
# =========================
st.markdown("""
<style>
.glass {
    background: rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 20px;
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    animation: fadeUp 0.8s ease-in-out;
}
.gradient {
    background: linear-gradient(90deg,#00f5ff,#7f00ff,#00f5ff);
    background-size: 200% auto;
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    animation: shimmer 6s linear infinite;
}
.kpi {
    font-size:26px;
    font-weight:700;
}
@keyframes fadeUp {
    from {opacity:0; transform:translateY(14px);}
    to {opacity:1; transform:translateY(0);}
}
@keyframes shimmer {
    to {background-position: 200% center;}
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    return (
        joblib.load("scaler.pkl"),
        joblib.load("pca.pkl"),
        joblib.load("kmeans_model.pkl")
    )

scaler, pca, kmeans = load_models()

FEATURES = [
    "Customer_Value",
    "Purchase_Frequency",
    "Campaign_Response",
    "Customer_For_Years"
]

CLUSTER_NAMES = {
    0: "Low Engagement Customers",
    1: "Loyal High-Value Customers",
    2: "Growth Potential Customers",
    3: "At-Risk Customers"
}

CLUSTER_COLORS = {
    "Low Engagement Customers": "#8fd3ff",
    "Loyal High-Value Customers": "#ff4d4d",
    "Growth Potential Customers": "#1f77b4",
    "At-Risk Customers": "#ffb3b3"
}

# =========================
# HEADER
# =========================
st.markdown("""
<h1 class="gradient">🧠 AI Customer Segmentation Engine</h1>
<p style="color:gray">
End-to-End Clustering | EDA | Business Strategy | What-If Analysis
</p>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📂 Control Panel")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home","📊 EDA","🧩 Cluster Insights","📍 PCA Map","🎯 Prediction"]
)

sidebar_file = st.sidebar.file_uploader("Upload CSV (Sidebar)", type="csv")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 What-If Prediction")

val = st.sidebar.number_input("Customer Value",0.0,3000.0,600.0)
freq = st.sidebar.number_input("Purchase Frequency",0.0,50.0,12.0)
camp = st.sidebar.number_input("Campaign Response",0.0,10.0,2.0)
years = st.sidebar.number_input("Customer For Years",0.0,20.0,3.0)

predict_btn = st.sidebar.button("🔮 Predict Segment")

# =========================
# HOME UPLOAD (DUPLICATE FOR UX)
# =========================
st.markdown("## 📂 Upload Customer Dataset")
home_file = st.file_uploader("Upload CSV to begin analysis", type="csv")

uploaded_file = home_file or sidebar_file

# =========================
# MAIN PIPELINE
# =========================
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    before = len(df)
    df = df.drop_duplicates()
    removed_duplicates = before - len(df)
    df = df.dropna()

    df["Customer_Value"] = df.filter(like="Mnt").sum(axis=1)
    df["Purchase_Frequency"] = df.filter(like="Num").sum(axis=1)
    df["Campaign_Response"] = df.filter(like="Accepted").sum(axis=1)
    df["Customer_For_Years"] = (
        pd.to_datetime("today").year -
        pd.to_datetime(df["Dt_Customer"], errors="coerce").dt.year
    ).fillna(0)

    X = scaler.transform(df[FEATURES])
    df["Cluster"] = kmeans.predict(X)
    df["Cluster_Name"] = df["Cluster"].map(CLUSTER_NAMES)

    pca_vals = pca.transform(X)
    df["PCA1"], df["PCA2"] = pca_vals[:,0], pca_vals[:,1]

# =========================
# HOME
# =========================
if page == "🏠 Home":
    if not uploaded_file:
        st.info("Upload dataset from Home or Sidebar to start")
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rows",df.shape[0])
        c2.metric("Columns",df.shape[1])
        c3.metric("Missing Values",0)
        c4.metric("Duplicate Records",0)  # ✔ FIXED (always 0 after cleaning)

        st.dataframe(df.head())
        st.success("Dataset cleaned, engineered & ready")

# =========================
# EDA
# =========================
if page == "📊 EDA" and uploaded_file:
    dist = df["Cluster_Name"].value_counts().reset_index()
    dist.columns=["Cluster","Count"]

    fig = px.bar(
        dist,x="Cluster",y="Count",
        color="Cluster",
        color_discrete_map=CLUSTER_COLORS
    )
    st.plotly_chart(fig,True)

    st.info(
        "This chart shows how customers distribute across behavioral segments. "
        "Larger clusters represent dominant behavior patterns, while smaller clusters "
        "often contribute disproportionately to revenue."
    )

    fig2 = px.pie(
        dist,names="Cluster",values="Count",
        hole=0.55,
        color="Cluster",
        color_discrete_map=CLUSTER_COLORS
    )
    st.plotly_chart(fig2,True)

    st.info(
        "Cluster share helps prioritize business strategy. "
        "Retention, upsell, and loyalty investments should align with these proportions."
    )

# =========================
# CLUSTER INSIGHTS (Animated KPI Cards)
# =========================
if page == "🧩 Cluster Insights" and uploaded_file:
    kpi = df.groupby("Cluster_Name")[FEATURES].mean().round(2)

    cols = st.columns(4)
    for i,(name,row) in enumerate(kpi.iterrows()):
        with cols[i%4]:
            st.markdown(f"""
            <div class="glass">
            <b>{name}</b><br><br>
            💰 Value: {row.Customer_Value}<br>
            🔁 Frequency: {row.Purchase_Frequency}<br>
            📣 Engagement: {row.Campaign_Response}
            </div>
            """,unsafe_allow_html=True)

    st.dataframe(kpi)

# =========================
# PCA MAP
# =========================
if page == "📍 PCA Map" and uploaded_file:
    fig = px.scatter(
        df,x="PCA1",y="PCA2",
        color="Cluster_Name",
        color_discrete_map=CLUSTER_COLORS,
        opacity=0.75
    )
    st.plotly_chart(fig,True)

    st.info(
        "Clear separation in PCA space confirms strong behavioral segmentation. "
        "Overlapping regions represent transition customers — ideal targets for nudging strategies."
    )

# =========================
# PREDICTION
# =========================
if page == "🎯 Prediction" and predict_btn and uploaded_file:
    user = np.array([[val,freq,camp,years]])
    user_scaled = scaler.transform(user)
    pred = kmeans.predict(user_scaled)[0]
    name = CLUSTER_NAMES[pred]

    st.subheader("🎯 Prediction Result")
    st.markdown(
        f"<h2 style='color:#00ff9c'><b>{name}</b></h2>",
        unsafe_allow_html=True
    )

    st.info(
        f"This customer matches **{name}** because:\n"
        f"- Spending & frequency align with cluster centroid\n"
        f"- Engagement behavior mirrors historical members\n\n"
        f"**Recommended Strategy:** Personalized pricing, targeted campaigns, retention nudges."
    )

    user_pca = pca.transform(user_scaled)
    fig = px.scatter(
        df,x="PCA1",y="PCA2",
        color="Cluster_Name",
        color_discrete_map=CLUSTER_COLORS,
        opacity=0.4
    )
    fig.add_scatter(
        x=[user_pca[0,0]],y=[user_pca[0,1]],
        mode="markers",
        marker=dict(size=18,symbol="star",color="yellow"),
        name="User Input"
    )
    st.plotly_chart(fig,True)

# =========================
# PDF DOWNLOAD
# =========================
if uploaded_file and PDF_AVAILABLE:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content=[Paragraph("Customer Segmentation Report",styles["Title"])]
    for k,v in df["Cluster_Name"].value_counts().items():
        content.append(Paragraph(f"{k}: {v}",styles["Normal"]))

    doc.build(content)
    buffer.seek(0)

    st.download_button(
        "📦 Download PDF Report",
        buffer,
        "customer_segmentation_report.pdf"
    )
