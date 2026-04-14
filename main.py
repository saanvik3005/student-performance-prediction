

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
import warnings
warnings.filterwarnings("ignore")


# Load Dataset

df = pd.read_csv("StudentPerformanceFactors.csv")

df = df[['Hours_Studied', 'Attendance', 'Parental_Involvement',
         'Access_to_Resources', 'Extracurricular_Activities', 'Exam_Score']]

print(" Full Dataset Shape ")
print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")


# Create Target Column

df["final_result"] = df["Exam_Score"].apply(
    lambda score: "Pass" if score >= 70 else "Fail"
)

print("\n Pass/Fail Distribution ")
print(df["final_result"].value_counts())


# Check for Missing Values

print("\n Missing Values ")
print(df.isnull().sum())
df = df.dropna()
print(f"Dataset shape after cleaning: {df.shape}")


# Basic Statistics

print("\n Basic Statistics ")
print(df.describe())


# EDA Plot 1 - Pass vs Fail Count

plt.figure(figsize=(6, 4))
sns.countplot(x="final_result", data=df, palette=["#2563a8", "#f0a500"])
plt.title("Pass vs Fail Count")
plt.xlabel("Result")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("plot1_pass_fail.png")
plt.show()
print("Plot 1 saved: Pass vs Fail Count")


# EDA Plot 2 - Exam Score Distribution

plt.figure(figsize=(7, 4))
sns.histplot(df["Exam_Score"], bins=20, kde=True, color="#2563a8")
plt.title("Distribution of Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("plot2_exam_score_distribution.png")
plt.show()
print("Plot 2 saved: Exam Score Distribution")


# EDA Plot 3 - Hours Studied vs Exam Score

plt.figure(figsize=(7, 4))
sns.scatterplot(x="Hours_Studied", y="Exam_Score", data=df,
                hue="final_result", palette=["#f0a500", "#2563a8"])
plt.title("Hours Studied vs Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.tight_layout()
plt.savefig("plot3_hours_vs_score.png")
plt.show()
print("Plot 3 saved: Hours Studied vs Exam Score")


# EDA Plot 4 - Attendance vs Exam Score

plt.figure(figsize=(7, 4))
sns.scatterplot(x="Attendance", y="Exam_Score", data=df,
                hue="final_result", palette=["#f0a500", "#2563a8"])
plt.title("Attendance vs Exam Score")
plt.xlabel("Attendance (%)")
plt.ylabel("Exam Score")
plt.tight_layout()
plt.savefig("plot4_attendance_vs_score.png")
plt.show()
print("Plot 4 saved: Attendance vs Exam Score")


# EDA Plot 5 - Parental Involvement vs Avg Exam Score

plt.figure(figsize=(6, 4))
sns.barplot(x="Parental_Involvement", y="Exam_Score", data=df,
            palette="Blues_d", order=["Low", "Medium", "High"])
plt.title("Parental Involvement vs Average Exam Score")
plt.xlabel("Parental Involvement Level")
plt.ylabel("Average Exam Score")
plt.tight_layout()
plt.savefig("plot5_parental_involvement.png")
plt.show()
print("Plot 5 saved: Parental Involvement vs Exam Score")


# EDA Plot 6 - Access to Resources vs Avg Exam Score

plt.figure(figsize=(6, 4))
sns.barplot(x="Access_to_Resources", y="Exam_Score", data=df,
            palette="Oranges_d", order=["Low", "Medium", "High"])
plt.title("Access to Resources vs Average Exam Score")
plt.xlabel("Access to Resources Level")
plt.ylabel("Average Exam Score")
plt.tight_layout()
plt.savefig("plot6_access_to_resources.png")
plt.show()
print("Plot 6 saved: Access to Resources vs Exam Score")

print("\n EDA Complete")


# Label Encoding - Convert Text to Numbers

le = LabelEncoder()

df["Parental_Involvement"]       = le.fit_transform(df["Parental_Involvement"])
df["Access_to_Resources"]        = le.fit_transform(df["Access_to_Resources"])
df["Extracurricular_Activities"] = le.fit_transform(df["Extracurricular_Activities"])
df["final_result"]               = le.fit_transform(df["final_result"])

print("\n Dataset After Encoding ")
print(df.head())


# Separate Features and Target

X = df[['Hours_Studied', 'Attendance', 'Parental_Involvement',
        'Access_to_Resources', 'Extracurricular_Activities', 'Exam_Score']]
y = df['final_result']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape:   {y.shape}")


# Feature Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Train Test Split 

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTraining Rows: {X_train.shape[0]}")
print(f"Testing Rows:  {X_test.shape[0]}")


# Helper Function to Evaluate Each Model

def evaluate_model(name, y_test, y_pred):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"  Precision : {precision_score(y_test, y_pred):.2f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.2f}")
    print(f"  F1 Score  : {f1_score(y_test, y_pred):.2f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Fail", "Pass"]))


# Model 1 - Logistic Regression

print("\n Training Logistic Regression")
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
evaluate_model("Logistic Regression", y_test, y_pred_lr)


# Model 2 - KNN

print("\n Training KNN")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
evaluate_model("KNN", y_test, y_pred_knn)


# Model 3 - SVM

print("\n Training SVM")
svm = SVC(kernel="rbf", random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
evaluate_model("SVM", y_test, y_pred_svm)


# Model 4 - Naive Bayes

print("\n Training Naive Bayes")
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
evaluate_model("Naive Bayes", y_test, y_pred_nb)


# Model 5 - Decision Tree

print("\nTraining Decision Tree")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
evaluate_model("Decision Tree", y_test, y_pred_dt)


#  Model 6 - Random Forest

print("\n Training Random Forest")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
evaluate_model("Random Forest", y_test, y_pred_rf)


#  Model 7 - PCA + Random Forest

print("\n Applying PCA + Random Forest")
pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)

print("\n PCA Explained Variance")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  Component {i+1}: {var*100:.2f}%")
print(f"  Total Variance Retained: {sum(pca.explained_variance_ratio_)*100:.2f}%")

rf_pca = RandomForestClassifier(random_state=42)
rf_pca.fit(X_train_pca, y_train)
y_pred_rf_pca = rf_pca.predict(X_test_pca)
evaluate_model("PCA + Random Forest", y_test, y_pred_rf_pca)


# Final Comparison Table

results = {
    "Model": [
        "Logistic Regression",
        "KNN",
        "SVM",
        "Naive Bayes",
        "Decision Tree",
        "Random Forest",
        "PCA + Random Forest",
    ],
    "Accuracy": [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_svm),
        accuracy_score(y_test, y_pred_nb),
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_rf_pca),
    ],
    "Precision": [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_knn),
        precision_score(y_test, y_pred_svm),
        precision_score(y_test, y_pred_nb),
        precision_score(y_test, y_pred_dt),
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_rf_pca),
    ],
    "Recall": [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_knn),
        recall_score(y_test, y_pred_svm),
        recall_score(y_test, y_pred_nb),
        recall_score(y_test, y_pred_dt),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_rf_pca),
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_knn),
        f1_score(y_test, y_pred_svm),
        f1_score(y_test, y_pred_nb),
        f1_score(y_test, y_pred_dt),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_rf_pca),
    ]
}

results_df = pd.DataFrame(results)
results_df["Accuracy"]  = (results_df["Accuracy"] * 100).round(2)
results_df["Precision"] = results_df["Precision"].round(2)
results_df["Recall"]    = results_df["Recall"].round(2)
results_df["F1 Score"]  = results_df["F1 Score"].round(2)

print("\n FINAL MODEL COMPARISON TABLE ")
print(results_df.to_string(index=False))


# Plot - Accuracy Comparison Bar Chart

plt.figure(figsize=(11, 5))
sns.barplot(x="Model", y="Accuracy", data=results_df, palette="Blues_d")
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("plot7_model_accuracy_comparison.png")
plt.show()
print("Plot 7 saved: Model Accuracy Comparison")


# Confusion Matrices for All 7 Models

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Confusion Matrices - All Models", fontsize=14)

models_info = [
    ("Logistic Regression", y_pred_lr),
    ("KNN",                 y_pred_knn),
    ("SVM",                 y_pred_svm),
    ("Naive Bayes",         y_pred_nb),
    ("Decision Tree",       y_pred_dt),
    ("Random Forest",       y_pred_rf),
    ("PCA + Random Forest", y_pred_rf_pca),
]

for ax, (name, y_pred) in zip(axes.flatten(), models_info):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Fail", "Pass"],
                yticklabels=["Fail", "Pass"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

# Hide the empty 8th subplot
axes[1][3].set_visible(False)

plt.tight_layout()
plt.savefig("plot8_confusion_matrices.png")
plt.show()
print("Plot 8 saved: Confusion Matrices for All Models")

print("\nPROJECT COMPLETE")
