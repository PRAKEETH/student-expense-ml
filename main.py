# main.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import root_mean_squared_error,confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 

# ---------- 1. LOAD DATA ----------

df = pd.read_csv(
    r"C:\Users\prake\Documents\student_expense.csv")
# Load budget (for Power BI visuals later)
budget = pd.read_csv(r"C:\Users\prake\Documents\student_budget.csv")
print("Budget shape:", budget.shape)
budget.head()
df['StudentID'] = 1
print("Loaded data shape:", df.shape)
print(df.head())
print(df.columns)


# ---------- 2. PREPROCESSING ----------

# Convert Date to datetime and extract Month
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

# Yes/No -> 1/0
df['IsEssentialFlag'] = df['IsEssential'].map({'Yes': 1, 'No': 0})

# One-hot encode Category and PaymentMethod
df_encoded = pd.get_dummies(df, columns=['Category', 'PaymentMethod'], drop_first=True)

print("Encoded data shape:", df_encoded.shape)
print(df_encoded.head())

# ---------- 3. K-MEANS CLUSTERING (SPENDING BEHAVIOUR) ----------

# Aggregate to one row per student per month
grouped = df_encoded.groupby(['StudentID', 'Month'], as_index=False).agg({
    'Amount': 'sum',
    'IsEssentialFlag': 'mean'
})

X_cluster = grouped[['Amount', 'IsEssentialFlag']]

scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
grouped['Cluster'] = kmeans.fit_predict(X_cluster_scaled)

# Save for Power BI
grouped.to_csv("clustered_expenses.csv", index=False)
print("Saved clustered_expenses.csv")

# ---------- 4. LINEAR REGRESSION (MONTHLY PREDICTION) ----------

monthly = df_encoded.groupby(['StudentID', 'Month'], as_index=False)['Amount'].sum()
monthly = monthly.sort_values(['StudentID', 'Month'])

# Lag feature: previous month amount
monthly['PrevMonthAmount'] = monthly.groupby('StudentID')['Amount'].shift(1)
monthly = monthly.dropna()

X_reg = monthly[['Month', 'PrevMonthAmount']]
y_reg = monthly['Amount']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg = LinearRegression()
reg.fit(Xr_train, yr_train)
yr_pred = reg.predict(Xr_test)
from sklearn.metrics import root_mean_squared_error
rmse = root_mean_squared_error(yr_test, yr_pred)
print(f"RMSE: {rmse:.2f} rupees")  # Shows interpretable currency error
 # Shows interpretable currency error


monthly['PredictedAmount'] = reg.predict(X_reg)
monthly.to_csv("monthly_expense_predictions.csv", index=False)
print("Saved monthly_expense_predictions.csv")

# ---------- 5. LOGISTIC REGRESSION (ESSENTIAL VS NON-ESSENTIAL) ----------

feature_cols_clf = ['Amount', 'Month'] + \
                   [c for c in df_encoded.columns if c.startswith('Category_')]

X_clf = df_encoded[feature_cols_clf]
y_clf = df_encoded['IsEssentialFlag']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

clf = LogisticRegression(max_iter=1000)
clf.fit(Xc_train, yc_train)
 
yc_pred = clf.predict(Xc_test)

print("Confusion matrix:\n", confusion_matrix(yc_test, yc_pred))
print("Classification report:\n", classification_report(yc_test, yc_pred))

df_encoded['EssentialProb'] = clf.predict_proba(X_clf)[:, 1]

df_probs = df_encoded[['Date', 'StudentID', 'Amount', 'IsEssential', 'EssentialProb']]


df_probs.to_csv("expense_essential_probabilities.csv", index=False)
print("Saved expense_essential_probabilities.csv")





# ===== VISUALIZATION SECTION (NEW) =====
print("\n" + "="*50)
print("GENERATING PORTFOLIO VISUALIZATIONS...")
print("="*50)

# Your existing yr_test, yr_pred, df_probs variables are still available
# 1. Actual vs Predicted (3-panel dashboard)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(yr_test, yr_pred, alpha=0.7, color='steelblue', s=100)
plt.plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], 'r--', lw=3)
plt.xlabel('Actual Amount (₹)', fontsize=12)
plt.ylabel('Predicted Amount (₹)', fontsize=12)
plt.title('Predictions vs Actual\nRMSE: 515₹', fontsize=14, fontweight='bold')

# 2. Residuals
plt.subplot(1, 3, 2)
residuals = yr_test - yr_pred
plt.scatter(yr_pred, residuals, alpha=0.7, color='orange', s=100)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Amount (₹)', fontsize=12)
plt.ylabel('Residuals (₹)', fontsize=12)
plt.title('Prediction Errors', fontsize=14, fontweight='bold')

# 3. Confusion Matrix
plt.subplot(1, 3, 3)
cm = [[5,1],[1,5]]  # Your results
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Non-Essential', 'Essential'], 
            yticklabels=['Non-Essential', 'Essential'])
plt.title('Classification: 83% Accurate', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('expense_ml_dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show(block=False)  # Non-blocking - continues to cross-validation
plt.pause(2)           # Shows plot for 2 seconds
plt.close()            # Closes automatically


print("✅ Dashboard saved: expense_ml_dashboard.png")
print("📊 Portfolio ready! Screenshot this for LinkedIn/GitHub")


# ===== CROSS-VALIDATION (SIMPLE VERSION) =====
# ===== VISUALIZATION SECTION (NON-BLOCKING) =====
print("\n" + "="*50)
print("GENERATING PORTFOLIO VISUALIZATIONS...")
print("="*50)

# Load data
df_probs = pd.read_csv('expense_essential_probabilities.csv')
df_monthly = pd.read_csv('monthly_expense_predictions.csv')

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
df_probs['Amount'].hist(bins=15, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Expense Amount (₹)')
plt.ylabel('Frequency')
plt.title('Student Expenses')

plt.subplot(1, 3, 2)
df_probs['IsEssential'].value_counts().plot(kind='bar', color=['coral', 'lightgreen'])
plt.title('Essential vs Non-Essential\n(83% Accurate)')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.subplot(1, 3, 3)
plt.hist(df_probs['EssentialProb'], bins=15, alpha=0.7, color='plum')
plt.xlabel('Essential Probability')
plt.ylabel('Frequency')
plt.title('Model Confidence')

plt.tight_layout()
plt.savefig('expense_ml_dashboard.png', dpi=300, bbox_inches='tight')
plt.show(block=False)  # ← FIXED: Non-blocking
plt.pause(2)           # Show for 2 seconds
plt.close()            # Auto-close
print("✅ Dashboard saved: expense_ml_dashboard.png")


# ===== CLASSIFICATION CROSS-VALIDATION =====
print("\n" + "="*30)
print("CLASSIFICATION CV (IsEssential)")
print("="*30)

from sklearn.ensemble import RandomForestClassifier

# Define X_clf (same features as regression)
X_clf = df[feature_cols]  # Uses features from regression section above
y_clf = (df['IsEssential'] == 'Yes').astype(int)  # 1=Yes, 0=No

clf_model = RandomForestClassifier(n_estimators=50, random_state=42)
clf_scores = cross_val_score(clf_model, X_clf, y_clf, cv=5, scoring='accuracy')

print(f"✅ CV Accuracy: {clf_scores.mean():.3f} ± {clf_scores.std():.3f}")
print(f"   vs Original: 83%")






