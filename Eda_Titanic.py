import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create screenshots folder
import os
os.makedirs("screenshots", exist_ok=True)

# Load Dataset
df = pd.read_csv(r"C:\Users\MAHALINGAM R\PycharmProjects\Task-5-EDA\train.csv")

# -----------------------
# Basic Dataset Analysis
# -----------------------

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

# -----------------------
# Histograms
# -----------------------

plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.savefig("screenshots/age_histogram.png")
plt.show()

# -----------------------
# Boxplot
# -----------------------

plt.figure(figsize=(8,5))
sns.boxplot(x=df["Fare"])
plt.title("Fare Boxplot")
plt.savefig("screenshots/fare_boxplot.png")
plt.show()

# -----------------------
# Survival Count
# -----------------------

plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.savefig("screenshots/survival_count.png")
plt.show()

# -----------------------
# Survival by Gender
# -----------------------

plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.savefig("screenshots/gender_survival.png")
plt.show()

# -----------------------
# Passenger Class vs Survival
# -----------------------

plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Passenger Class vs Survival")
plt.savefig("screenshots/class_survival.png")
plt.show()

# -----------------------
# Correlation Heatmap
# -----------------------

df_corr = df.copy()

df_corr["Sex"] = df_corr["Sex"].map({
    "male": 0,
    "female": 1
})

plt.figure(figsize=(10,6))
sns.heatmap(
    df_corr.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.savefig("screenshots/heatmap.png")
plt.show()

# -----------------------
# Pairplot
# -----------------------

sns.pairplot(
    df[["Age", "Fare", "Pclass", "Survived"]]
)

plt.savefig("screenshots/pairplot.png")
plt.show()

# -----------------------
# Scatterplot
# -----------------------

plt.figure(figsize=(8,5))
sns.scatterplot(
    x="Age",
    y="Fare",
    hue="Survived",
    data=df
)

plt.title("Age vs Fare")
plt.savefig("screenshots/scatterplot.png")
plt.show()

print("\nEDA Completed Successfully.")