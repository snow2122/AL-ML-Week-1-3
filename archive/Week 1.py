import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#file path
disease_filepath = r"C:\Users\Meerab\Downloads\internsip\heart+disease\processed.cleveland.data"

#assigning headers to the columns from the UCI description

column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]
#converting '?' with NaN so that Pandas can understand it
df = pd.read_csv(disease_filepath, header = None, na_values= '?', names = column_names)
#print(df.isnull().sum())

#make target binary BEFORE scaling
df["target"] = df["target"].astype(float)
df["target"] = (df["target"] > 0).astype(int)

#filling numeric values with median
df.fillna(df.median(numeric_only=True),inplace=True)

#filling non-numeric (categorical) columns with mode
for col in df.select_dtypes(include='object'):
    df[col].fillna(df[col].mode()[0],inplace=True)
#print(df.isnull().sum())

#normalizing data between 0 and 1
#creating scaler
scaler = MinMaxScaler()
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

#applying scaling
df[numeric_cols] =  scaler.fit_transform(df[numeric_cols])

#creating encoder
le= LabelEncoder()
#looks for categorical columns and only applies label encoding to them
for col in df.columns:
    if  df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

#stats
# Show basic statistics of dataset
#print(df.describe())

#correlation matrix
corr = df.corr()
#making heatmap
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, cmap = "coolwarm", fmt= ".2f")
plt.title("Correlation Heatmap")
plt.show()

#training models
X = df.drop(columns=["target"])
y = df["target"].apply(lambda v: 1 if float(v) > 0 else 0)  # make sure it's 0/1

#split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#defining models
lr = LogisticRegression(max_iter=1000)                 # Logistic Regression
rf = RandomForestClassifier(n_estimators=200, random_state=42)  # Random Forest

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

#print("Trained Logistic Regression and Random Forest.")

# Logistic Regression predictions
y_pred_lr = lr.predict(X_test)

# Random Forest predictions
y_pred_rf = rf.predict(X_test)

# Calculating accuracy
acc_lr = accuracy_score(y_test, y_pred_lr)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("Logistic Regression Accuracy:", acc_lr)
print("Random Forest Accuracy:", acc_rf)

#comparison
if acc_lr > acc_rf:
    print("Logistic Regression performed better :)")
else:
    print("Random Forest performed better :)")
