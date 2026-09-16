import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score , classification_report

df=pd.read_csv('mushrooms.csv')
df.head()



df.info()

df.shape

df.isnull().sum()/3276

df.duplicated().sum()

for i in df:
  print(i)
  print(df[i].unique())

encoders = {}

for col in df:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder
    import joblib


joblib.dump(encoders, 'encoders.pkl')
df.head()


for col in df:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper= Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"Outliers in {col}:")
    print("Numbre of Outliers :", outliers.shape[0])
    print("_-"*30)

for col in df.drop('class', axis=1):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper= Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower,  df[col])
    df[col] = np.where(df[col] > upper, upper,  df[col])

for col in df:
  plt.figure(figsize=(9, 6))
  sns.boxplot(x=df[col])
  plt.title(col)
  plt.show()


from matplotlib import figure
plt.figure(figsize=(18, 12))
sns.heatmap(df.corr(), annot=True, cmap=("coolwarm"))
plt.show()

X=df.drop('class',axis=1)
y=df['class']
X_train,X_test,y_train,y_test=train_test_split(X , y ,random_state=42,test_size=0.2 )

scale = StandardScaler()
X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)

lg_model=LogisticRegression()
lg_model.fit(X_train_scaled,y_train)
y_pred_lg=lg_model.predict(X_test_scaled)

lg_acc = accuracy_score(y_test, y_pred_lg)
print("Accuracy:", lg_acc)

dt=DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train_scaled,y_train)
y_pred_dt=dt.predict(X_test_scaled)

dt_acc=accuracy_score(y_test,y_pred_dt)
print("Accuracy:", dt_acc)
print(classification_report(y_test,y_pred_dt))

rf=RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    min_samples_split=10,

)
rf.fit(X_train_scaled,y_train)
y_pred_rf=rf.predict(X_test_scaled)

rf_acc=accuracy_score(y_test,y_pred_rf)
print("Accuracy:", rf_acc)
print(classification_report(y_test,y_pred_rf))


ab=AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=70,
    learning_rate=0.05,
    random_state=42
)
ab.fit(X_train_scaled,y_train)
y_pred_ab=ab.predict(X_test_scaled)

ab_acc=accuracy_score(y_test,y_pred_ab)
print("Accuracy:", ab_acc)
print(classification_report(y_test,y_pred_ab))


gb=GradientBoostingClassifier(
    n_estimators=70,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
gb.fit(X_train_scaled,y_train)
y_pred_gb=gb.predict(X_test_scaled)

gb_acc=accuracy_score(y_test,y_pred_gb)
print("Accuracy:", gb_acc)
print(classification_report(y_test,y_pred_gb))

xgb=XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.6,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(X_train_scaled,y_train)
y_pred_xgb=xgb.predict(X_test_scaled)

xgb_acc=accuracy_score(y_test,y_pred_xgb)


print("Accuracy: ",accuracy_score(y_test,y_pred_xgb))
print(classification_report(y_test,y_pred_xgb))

plt.figure(figsize=(10, 6))
plt.bar(['Logistic Regression', 'Decision Tree', 'Random Forest', 'AdaBoost', 'Gradient Boosting', 'XGBoost'],
       [lg_acc, dt_acc, rf_acc, ab_acc, gb_acc, xgb_acc])
plt.ylabel('Accuracy')
plt.title('Comparison of Models')
plt.show()

import joblib

joblib.dump(xgb, 'xgb_model.pkl')
joblib.dump(scale, 'scaler.pkl')