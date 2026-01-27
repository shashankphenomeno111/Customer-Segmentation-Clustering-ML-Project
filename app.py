"""
Customer Segmentation Dashboard
A comprehensive Streamlit dashboard for customer segmentation analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
import warnings
warnings.filterwarnings('ignore')

# ===================================
# PAGE CONFIGURATION
# ===================================
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================
# CUSTOM CSS
# ===================================
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(145deg, #1e3a5f, #0d2137);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2e5a8a;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .kpi-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00d4ff;
    }
    .kpi-label {
        font-size: 1rem;
        color: #a0a0a0;
        margin-top: 5px;
    }
    
    /* Cluster Cards */
    .cluster-card {
        background: linear-gradient(145deg, #2d3748, #1a202c);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid;
    }
    .cluster-0 { border-left-color: #32CD32; }  /* Green - Highly Engaged */
    .cluster-1 { border-left-color: #9932CC; }  /* Purple - High Risk */
    .cluster-2 { border-left-color: #4169E1; }  /* Blue - VIP */
    .cluster-3 { border-left-color: #FF4444; }  /* Red - Price Sensitive */
    
    /* Headers */
    .section-header {
        color: #00d4ff;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 20px 0 10px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #2e5a8a;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #1a1a2e;
    }
    
    /* Strategy box */
    .strategy-box {
        background: #1e3a5f;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ===================================
# LOAD DATA & MODEL
# ===================================
@st.cache_data
def load_data():
    df = pd.read_csv("marketing_campaign.csv")
    
    # Handle missing values
    df = df.apply(lambda x: x.fillna(x.median()) if x.dtype != 'object' else x.fillna(x.mode()[0]))
    df = df.drop_duplicates()
    
    # Feature Engineering
    df['Customer_Value'] = df['Income'] + df['MntWines'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
    df['Purchase_Frequency'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases'] + df['NumDealsPurchases']
    df['Campaign_Response'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5'] + df['Response']
    df['Customer_For_Years'] = (pd.to_datetime('2024-01-01') - pd.to_datetime(df['Dt_Customer'], dayfirst=True)).dt.days / 365
    
    return df

@st.cache_resource
def train_model(df):
    FEATURES = ['Customer_Value', 'Purchase_Frequency', 'Campaign_Response', 'Customer_For_Years']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # PCA for visualization
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_scaled)
    
    return kmeans, scaler, labels, X_scaled, X_pca_2d, X_pca_3d, FEATURES

# Load data
df = load_data()
kmeans, scaler, labels, X_scaled, X_pca_2d, X_pca_3d, FEATURES = train_model(df)
df['Cluster'] = labels

# ===================================
# SIDEBAR
# ===================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/customer-insight.png", width=80)
    st.title("🎯 Customer Segmentation")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "📍 Navigation",
        ["📊 Dashboard", "📈 Cluster Analysis", "💼 Business Strategies", "🔮 Predict Segment"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📋 Dataset Info")
    st.info(f"""
    **Total Customers:** {len(df):,}  
    **Features Used:** {len(FEATURES)}  
    **Clusters:** 4
    """)
    
    st.markdown("---")
    st.markdown("### 🎨 Cluster Colors")
    cluster_info = {
        "🟢 Cluster 0": "Highly Engaged",
        "🟣 Cluster 1": "High Risk/At-Risk",
        "🔵 Cluster 2": "VIP Premium",
        "🔴 Cluster 3": "Price-Sensitive"
    }
    for k, v in cluster_info.items():
        st.markdown(f"**{k}**  \n{v}")

# ===================================
# CLUSTER DEFINITIONS (Based on actual data analysis)
# ===================================
cluster_profiles = {
    0: {
        "name": "Highly Engaged",
        "emoji": "🟢",
        "color": "#32CD32",
        "description": "High value customers with frequent purchases. Active buyers who shop regularly but don't respond to campaigns.",
        "characteristics": ["High customer value ($68K)", "High purchase frequency (21)", "Low campaign response", "Long-term customers"],
        "strategies": [
            "Loyalty rewards program",
            "Personalized product recommendations",
            "Exclusive member discounts",
            "Email engagement campaigns",
            "Cross-sell complementary products"
        ]
    },
    1: {
        "name": "High Risk",
        "emoji": "�",
        "color": "#9932CC",
        "description": "Low value customers with infrequent purchases. Risk of churning - need re-engagement strategies.",
        "characteristics": ["Low customer value ($40K)", "Low purchase frequency (8)", "No campaign response", "Shorter tenure"],
        "strategies": [
            "'We miss you' win-back emails",
            "One-time reactivation discount",
            "Survey to understand issues",
            "New product alerts",
            "Cart abandonment reminders"
        ]
    },
    2: {
        "name": "VIP Premium Customers",
        "emoji": "🔵",
        "color": "#4169E1",
        "description": "HIGHEST value customers who respond to campaigns. Most valuable segment - premium buyers.",
        "characteristics": ["Highest customer value ($80K)", "High purchase frequency (21)", "Campaign responsive (2)", "Long-term loyal"],
        "strategies": [
            "VIP membership with exclusive perks",
            "Early access to new products",
            "Personal account manager",
            "Exclusive events and private sales",
            "White-glove premium service"
        ]
    },
    3: {
        "name": "Price-Sensitive Shoppers",
        "emoji": "🔴",
        "color": "#FF4444",
        "description": "LOWEST value customers. Budget-conscious buyers who need value-focused offers.",
        "characteristics": ["Lowest customer value ($33K)", "Low purchase frequency (9)", "No campaign response", "Price-conscious"],
        "strategies": [
            "Volume discounts (Buy 2 Get 1 Free)",
            "Bundle deals combining products",
            "Flash sales and limited-time offers",
            "Free shipping above minimum order",
            "Clearance and sale promotions"
        ]
    }
}

# ===================================
# PAGE: DASHBOARD
# ===================================
if page == "📊 Dashboard":
    st.title("📊 Customer Segmentation Dashboard")
    st.markdown("**Analyze customer segments and drive targeted marketing strategies**")
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{len(df):,}</div>
            <div class="kpi-label">Total Customers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_value = df['Customer_Value'].mean()
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">${avg_value/1000:.1f}K</div>
            <div class="kpi-label">Avg Customer Value</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        vip_count = (labels == 1).sum()
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{vip_count}</div>
            <div class="kpi-label">VIP Customers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        at_risk = (labels == 3).sum()
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{at_risk}</div>
            <div class="kpi-label">At-Risk Customers</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Cluster Distribution
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<p class="section-header">📊 Cluster Distribution</p>', unsafe_allow_html=True)
        
        cluster_counts = df['Cluster'].value_counts().sort_index()
        colors = ['#32CD32', '#9932CC', '#4169E1', '#FF4444']  # Green, Purple, Blue, Red
        
        fig_pie = px.pie(
            values=cluster_counts.values,
            names=[f"Cluster {i}: {cluster_profiles[i]['name']}" for i in cluster_counts.index],
            color_discrete_sequence=colors,
            hole=0.4
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(x=0, y=-0.2, orientation='h')
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown('<p class="section-header">📈 Cluster Size Comparison</p>', unsafe_allow_html=True)
        
        fig_bar = px.bar(
            x=[f"{cluster_profiles[i]['emoji']} {cluster_profiles[i]['name']}" for i in cluster_counts.index],
            y=cluster_counts.values,
            color=[cluster_profiles[i]['name'] for i in cluster_counts.index],
            color_discrete_sequence=colors,
            text=cluster_counts.values
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis_title="",
            yaxis_title="Number of Customers",
            showlegend=False
        )
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # 2D Cluster Visualization
    st.markdown('<p class="section-header">🎯 2D Cluster Visualization</p>', unsafe_allow_html=True)
    
    df_viz = pd.DataFrame({
        'PCA1': X_pca_2d[:, 0],
        'PCA2': X_pca_2d[:, 1],
        'Cluster': [f"{cluster_profiles[c]['emoji']} {cluster_profiles[c]['name']}" for c in labels]
    })
    
    # Create explicit color map for each cluster label
    color_map = {f"{cluster_profiles[i]['emoji']} {cluster_profiles[i]['name']}": cluster_profiles[i]['color'] for i in range(4)}
    
    fig_2d = px.scatter(
        df_viz,
        x='PCA1',
        y='PCA2',
        color='Cluster',
        color_discrete_map=color_map,
        title="K-Means Customer Segments (2D PCA Projection)"
    )
    fig_2d.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.8)',
        font=dict(color='white'),
        xaxis_title="PCA 1 → Customer Value & Spending Power",
        yaxis_title="PCA 2 → Engagement & Loyalty",
        legend=dict(x=0.01, y=0.99)
    )
    fig_2d.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig_2d.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    st.plotly_chart(fig_2d, use_container_width=True)

# ===================================
# PAGE: CLUSTER ANALYSIS
# ===================================
elif page == "📈 Cluster Analysis":
    st.title("📈 Cluster Analysis")
    st.markdown("**Deep dive into each customer segment**")
    
    # 3D Visualization
    st.markdown('<p class="section-header">🌐 3D Interactive Cluster View</p>', unsafe_allow_html=True)
    
    df_3d = pd.DataFrame({
        'PCA1': X_pca_3d[:, 0],
        'PCA2': X_pca_3d[:, 1],
        'PCA3': X_pca_3d[:, 2],
        'Cluster': [f"{cluster_profiles[c]['emoji']} {cluster_profiles[c]['name']}" for c in labels]
    })
    
    # Create explicit color map for each cluster label
    color_map_3d = {f"{cluster_profiles[i]['emoji']} {cluster_profiles[i]['name']}": cluster_profiles[i]['color'] for i in range(4)}
    
    fig_3d = px.scatter_3d(
        df_3d,
        x='PCA1', y='PCA2', z='PCA3',
        color='Cluster',
        color_discrete_map=color_map_3d,
        title="3D Customer Segments Visualization"
    )
    fig_3d.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        scene=dict(
            xaxis=dict(backgroundcolor='rgba(26,26,46,0.8)', gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(backgroundcolor='rgba(26,26,46,0.8)', gridcolor='rgba(255,255,255,0.1)'),
            zaxis=dict(backgroundcolor='rgba(26,26,46,0.8)', gridcolor='rgba(255,255,255,0.1)')
        ),
        height=600
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Cluster Statistics
    st.markdown('<p class="section-header">📊 Cluster Statistics</p>', unsafe_allow_html=True)
    
    cluster_stats = df.groupby('Cluster')[FEATURES].mean().round(2)
    cluster_counts = df['Cluster'].value_counts().sort_index()
    
    for cluster in range(4):
        profile = cluster_profiles[cluster]
        with st.expander(f"{profile['emoji']} {profile['name']} ({cluster_counts[cluster]} customers)", expanded=True):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"**{profile['description']}**")
                st.markdown("**Characteristics:**")
                for char in profile['characteristics']:
                    st.markdown(f"• {char}")
            
            with col2:
                stats = cluster_stats.loc[cluster]
                st.dataframe(pd.DataFrame({
                    'Feature': FEATURES,
                    'Average Value': [f"{stats[f]:.2f}" for f in FEATURES]
                }), hide_index=True)

# ===================================
# PAGE: BUSINESS STRATEGIES
# ===================================
elif page == "💼 Business Strategies":
    st.title("💼 Business Strategies")
    st.markdown("**Actionable recommendations for each customer segment**")
    
    # Define cluster_counts for this page
    cluster_counts = df['Cluster'].value_counts().sort_index()
    
    st.info("""
    📌 **How to Use This Page:**  
    Each customer segment requires different marketing strategies. Below you'll find:
    - **Segment Profile** - Understanding who these customers are
    - **Recommended Strategies** - Specific actions to take
    - **Key Characteristics** - Traits that define this segment
    - **Priority Matrix** - Which segments to focus on first
    """)
    
    for cluster in range(4):
        profile = cluster_profiles[cluster]
        count = cluster_counts[cluster]
        pct = count / len(df) * 100
        
        st.markdown(f"""
        <div class="cluster-card cluster-{cluster}">
            <h3>{profile['emoji']} {profile['name']} ({count} customers - {pct:.1f}%)</h3>
            <p style="color: #a0a0a0;">{profile['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 Recommended Strategies:**")
            for i, strategy in enumerate(profile['strategies'], 1):
                st.markdown(f"{i}. {strategy}")
        
        with col2:
            st.markdown("**🎯 Key Characteristics:**")
            for char in profile['characteristics']:
                st.success(char)
        
        st.markdown("---")
    
    # Priority Matrix
    st.markdown('<p class="section-header">📌 Priority Matrix</p>', unsafe_allow_html=True)
    st.markdown("""
    **How to read this table:**  
    - **Priority** - How urgently this segment needs attention (Critical = highest)
    - **Primary Goal** - Main objective for this segment
    - **Expected ROI** - Return on investment from marketing efforts
    """)
    
    priority_data = pd.DataFrame({
        'Segment': ['Price-Sensitive Loyalists', 'VIP Premium', 'Engaged Regulars', 'At-Risk/Dormant'],
        'Size': [cluster_counts[i] for i in range(4)],
        'Priority': ['⭐⭐⭐ Medium', '⭐⭐⭐⭐⭐ Critical', '⭐⭐⭐⭐ High', '⭐⭐ Low'],
        'Primary Goal': ['Increase Order Value', 'Retention', 'Cross-sell', 'Win-back'],
        'Expected ROI': ['★★★☆☆', '★★★★★', '★★★★☆', '★★☆☆☆']
    })
    
    st.dataframe(priority_data, hide_index=True, use_container_width=True)

# ===================================
# PAGE: PREDICT SEGMENT
# ===================================
elif page == "🔮 Predict Segment":
    st.title("🔮 Predict Customer Segment")
    st.markdown("**Enter customer details to predict their segment**")
    
    # Input guidance
    st.info("""
    📌 **How to Use This Page:**  
    Enter the customer's feature values below to predict which segment they belong to.
    The model will classify the customer into one of 4 segments based on their characteristics.
    """)
    
    # Sample values reference
    with st.expander("📊 View Sample Values & Feature Descriptions", expanded=True):
        st.markdown("""
        | Feature | Description | Typical Range |
        |---------|-------------|---------------|
        | **Customer Value** | Income + Total spending across all categories | $30,000 - $80,000 |
        | **Purchase Frequency** | Total purchases (Web + Catalog + Store + Deals) | 5 - 25 |
        | **Campaign Response** | Number of campaigns accepted (0-6) | 0 - 3 |
        | **Customer Tenure** | Years since customer joined | 10 - 11 years |
        """)
        
        st.markdown("### 🎯 **EXACT VALUES TO GET EACH SEGMENT:**")
        st.warning("⚠️ Use these EXACT values to correctly predict each segment!")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.error("""
            **🔴 Cluster 3 - Price-Sensitive (LOWEST VALUE):**
            - Customer Value: **33,000** ← LOWEST
            - Purchase Frequency: **9** ← LOW  
            - Campaign Response: **0** ← NOT RESPONSIVE
            - Tenure: **11 years**
            
            💡 *Budget-conscious, needs discounts*
            ✅ Try: **33000, 9, 0, 11**
            """)
            
            st.markdown("""
            <div style="background-color: #9932CC22; padding: 15px; border-radius: 10px; border-left: 4px solid #9932CC;">
            
            **🟣 Cluster 1 - High Risk:**
            - Customer Value: **40,000** ← LOW
            - Purchase Frequency: **8** ← LOWEST
            - Campaign Response: **0** ← NOT RESPONSIVE
            - Tenure: **10 years**
            
            💡 *Low engagement, might churn*
            ✅ Try: **40000, 8, 0, 10**
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown("""
            <div style="background-color: #32CD3222; padding: 15px; border-radius: 10px; border-left: 4px solid #32CD32;">
            
            **🟢 Cluster 0 - Highly Engaged:**
            - Customer Value: **68,000** ← HIGH
            - Purchase Frequency: **21** ← HIGH
            - Campaign Response: **0** ← NOT RESPONSIVE
            - Tenure: **10.6 years**
            
            💡 *Active buyers, shop frequently*
            ✅ Try: **68000, 21, 0, 10.6**
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background-color: #4169E122; padding: 15px; border-radius: 10px; border-left: 4px solid #4169E1;">
            
            **🔵 Cluster 2 - VIP Premium (BEST!):**
            - Customer Value: **80,000** ← HIGHEST
            - Purchase Frequency: **21** ← HIGH
            - Campaign Response: **2** ← RESPONSIVE!
            - Tenure: **10.6 years**
            
            💡 *Top spenders, respond to campaigns*
            ✅ Try: **80000, 21, 2, 10.6**
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📝 Enter Customer Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💰 Customer Value ($)**")
        st.caption("Income + Total spending (Wine, Meat, Fish, Sweets, Gold)")
        customer_value = st.number_input(
            "Customer Value", 
            min_value=0, 
            max_value=500000, 
            value=50000,
            help="Sum of customer's income and all product spending",
            label_visibility="collapsed"
        )
        
        st.markdown("**🛒 Purchase Frequency**")
        st.caption("Total number of purchases across all channels")
        purchase_freq = st.number_input(
            "Purchase Frequency", 
            min_value=0, 
            max_value=100, 
            value=15,
            help="Web + Catalog + Store + Deal purchases",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("**📧 Campaign Response (0-6)**")
        st.caption("Number of marketing campaigns accepted")
        campaign_response = st.number_input(
            "Campaign Response", 
            min_value=0, 
            max_value=6, 
            value=2,
            help="How many of 6 campaigns the customer responded to",
            label_visibility="collapsed"
        )
        
        st.markdown("**📅 Customer Tenure (Years)**")
        st.caption("How long they've been a customer")
        years = st.number_input(
            "Years", 
            min_value=0.0, 
            max_value=15.0, 
            value=2.0,
            help="Years since first purchase",
            label_visibility="collapsed"
        )
    
    st.markdown("---")
    
    if st.button("🔮 Predict Segment", type="primary", use_container_width=True):
        # Prepare input
        input_data = np.array([[customer_value, purchase_freq, campaign_response, years]])
        input_scaled = scaler.transform(input_data)
        prediction = kmeans.predict(input_scaled)[0]
        
        profile = cluster_profiles[prediction]
        
        st.balloons()
        
        st.markdown(f"""
        <div class="cluster-card cluster-{prediction}" style="text-align: center;">
            <h2>{profile['emoji']} {profile['name']}</h2>
            <p style="font-size: 1.2rem; color: #a0a0a0;">{profile['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Recommended Actions for This Customer:")
        for strategy in profile['strategies']:
            st.info(f"✅ {strategy}")
        
        st.markdown("### 🎯 Key Characteristics of This Segment:")
        for char in profile['characteristics']:
            st.success(char)

# ===================================
# FOOTER
# ===================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Customer Segmentation Dashboard | Built with Streamlit | © 2024</p>
</div>
""", unsafe_allow_html=True)
