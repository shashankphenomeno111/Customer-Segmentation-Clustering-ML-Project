🧠 Customer Segmentation using KMeans
From Raw Customer Data → Business-Driven Decisions
<p align="center"> <img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/da166369-980d-4890-96fd-ebc954db7bc4" />
 </p> <p align="center"> <b>End-to-End Machine Learning Project • Clustering • Business Analytics • Deployment</b> </p>
🚀 Project Overview

This project builds a customer segmentation system using KMeans clustering to group customers based on value, engagement, responsiveness, and loyalty.

The model helps businesses:

Identify high-value customers

Detect at-risk churn segments

Design targeted marketing strategies

Make data-driven retention decisions

A fully interactive Streamlit app is deployed to simulate customer behavior and instantly visualize cluster insights.

🎯 Business Problem

Businesses struggle to understand which customers to retain, upsell, or re-engage.

❌ Challenges

Treating all customers the same

No visibility into behavioral patterns

High churn without early warning

✅ Solution

Use unsupervised machine learning (clustering) to segment customers based on real behavior.

🧩 Features Used

We engineered 4 core behavioral dimensions:

Feature	Description
💰 Customer_Value	Total spending across all products
🛒 Purchase_Frequency	Overall buying intensity
📣 Campaign_Response	Marketing responsiveness
⏳ Customer_For_Years	Customer loyalty (tenure)
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

Distribution plots for customer behavior

Outlier detection

Correlation analysis

PCA for dimensionality reduction

PCA helped visualize clusters in 2D & 3D space for interpretability.

🤖 Models Evaluated

We tested 3 clustering algorithms:

Model	Silhouette Score
🥇 KMeans	0.46 (Best)
Agglomerative	0.42
DBSCAN	0.26

✅ KMeans was selected for deployment due to:

Better cluster separation

Business interpretability

Stable performance

🧠 Cluster Interpretation (Auto-Generated)

Each cluster is automatically interpreted using centroid values:

Cluster	Segment Name	Business Meaning
0	💎 High-Value Loyal	Top spenders, highly engaged
1	📈 High Spend – Low Engagement	Risk of churn
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

<p align="center"> <img src="assets/app_preview.png" width="90%"> </p>
📍 Visualization Highlights

🎯 Cluster decision regions

📊 PCA 2D scatter with convex hulls

⭐ Current customer highlighted

🌈 High-contrast, dark-theme UI

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
 ┃   ┗ app_preview.png
 ┗ 📜 README.md

▶️ How to Run Locally
# Step 1: Clone the repo
git clone https://github.com/your-username/customer-segmentation.git

# Step 2: Navigate to project
cd customer-segmentation

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run app
streamlit run app.py

📈 Key Takeaways

Clustering enables actionable customer insights

PCA improves interpretability

Business logic bridges ML → decision making

Deployment turns analysis into real value

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
🚀 Open to Data Science roles
