🧠 Customer Segmentation using KMeans
From Raw Customer Data → Business-Driven Decisions
<p align="center"> <img src="assets/customer_segmentation_banner.png" width="100%" /> </p> <p align="center"> <b>End-to-End Machine Learning Project • Clustering • Business Analytics • Deployment</b> </p> <p align="center">










</p>
🚀 Project Overview

This project builds a customer segmentation system using KMeans clustering to group customers based on value, engagement, responsiveness, and loyalty.

The system enables businesses to:

💎 Identify high-value loyal customers

⚠️ Detect at-risk churn segments

📈 Improve engagement for under-utilized customers

🎯 Design targeted marketing & retention strategies

A fully interactive Streamlit web app allows users to simulate customer behavior and instantly visualize cluster insights and business actions.

🎯 Business Problem

Businesses often struggle to understand which customers to retain, upsell, or re-engage.

❌ Challenges

Treating all customers the same

No visibility into behavioral patterns

High churn without early warning signals

✅ Solution

Use unsupervised machine learning (clustering) to segment customers based on real behavioral data, not assumptions.

🧩 Features Used (Behavioral Dimensions)

We engineered 4 core customer behavior features:

Feature	Description
💰 Customer_Value	Total spending across all products
🛒 Purchase_Frequency	Overall buying intensity
📣 Campaign_Response	Marketing responsiveness
⏳ Customer_For_Years	Customer loyalty / tenure

These features form the backbone of business-driven clustering.

🔁 Project Workflow
Raw Data
   ↓
Feature Engineering
   ↓
Standard Scaling
   ↓
KMeans Clustering
   ↓
Cluster Interpretation
   ↓
Business Strategy Mapping
   ↓
Streamlit Deployment

📊 Exploratory Data Analysis (EDA)

EDA was performed to understand customer behavior and prepare data for clustering:

Distribution analysis of customer value & engagement

Outlier detection

Correlation analysis

Feature scaling using StandardScaler

PCA for dimensionality reduction

🧠 PCA Insights

PCA enabled 2D & 3D visualization of clusters, making patterns interpretable for business users.

<p align="center"> <img src="assets/eda_workflow.png" width="90%" /> </p>
🤖 Models Evaluated

We evaluated three clustering algorithms:

Model	Silhouette Score
🥇 KMeans	0.46 (Best)
Agglomerative	0.42
DBSCAN	0.26
✅ Why KMeans?

Better cluster separation

Stable performance

Easier business interpretation

Suitable for production deployment

🧠 Cluster Interpretation (Auto-Generated)

Cluster meanings are automatically inferred from centroid values.

Cluster	Segment Name	Business Meaning
0	💎 High-Value Loyal	Top spenders, highly engaged
1	📈 High Spend – Low Engagement	Churn risk
2	🌱 New / Occasional	Early lifecycle customers
3	🔴 Low-Value	Minimal engagement
📌 Business Strategy per Cluster
Segment	Objective	Recommended Actions
💎 High-Value Loyal	Maximize lifetime value	VIP rewards, premium offers
📈 High Spend – Low Engagement	Prevent churn	Personalized campaigns
🌱 New Customers	Convert to loyal	Onboarding & incentives
🔴 Low-Value	Cost optimization	Generic offers
🖥️ Interactive Web App (Streamlit)
✨ App Capabilities

Slider-based customer simulation

Real-time cluster prediction

Business objective explanation

Risk level visualization

PCA scatter plot with customer position

<p align="center"> <img src="assets/app_preview.png" width="90%" /> </p>
🌍 Live Deployment

🚀 Live Streamlit App
👉 https://customer-segmentation-clustering-4.streamlit.app/

Deployment Highlights

Model artifacts loaded using Joblib

Real-time inference pipeline

Dark-themed, business-friendly UI

Interactive Plotly visualizations

<p align="center"> <img src="assets/deployment_preview.png" width="90%" /> </p>
🛠️ Tech Stack

Python

Pandas / NumPy

Scikit-Learn

Matplotlib / Seaborn / Plotly

Streamlit

Joblib

📁 Project Structure
📦 customer-segmentation
 ┣ 📜 app.py
 ┣ 📜 kmeans_model.pkl
 ┣ 📜 scaler.pkl
 ┣ 📜 pca.pkl
 ┣ 📜 ultra_clustering.ipynb
 ┣ 📁 assets
 ┃   ┣ customer_segmentation_banner.png
 ┃   ┣ eda_workflow.png
 ┃   ┣ app_preview.png
 ┃   ┗ deployment_preview.png
 ┗ 📜 README.md

▶️ How to Run Locally
# Clone repository
git clone https://github.com/shashankphenomeno111/Customer-Segmentation-Clustering-ML-Project.git

# Navigate to project
cd Customer-Segmentation-Clustering-ML-Project

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

📈 Key Takeaways

Clustering enables actionable customer insights

PCA improves model interpretability

Business logic bridges ML → decision-making

Deployment converts analysis into real business value

⭐ Why This Project Stands Out

✔ End-to-end ML pipeline
✔ Business-aligned clustering
✔ Strong visual storytelling
✔ Production-ready deployment
✔ Recruiter & portfolio friendly

🙌 Author

Shashank R
Aspiring Data Scientist | Machine Learning | Business Analytics

📫 Let’s connect on LinkedIn!
🚀 Open to Data Science / ML roles
