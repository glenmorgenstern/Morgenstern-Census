# %%
# %pip install ucimlrepo
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
us_census_data_1990 = fetch_ucirepo(id=116) 

# This project uses the 1990 U.S. Census dataset from the UCI Machine Learning Repository to model marital status (iMarital). The dataset was chosen because it is large, diverse, and rich in demographic and socioeconomic variables, making it ideal for supervised learning. The goal is to identify which attributes (e.g., age, income, education, occupation) are predictive of marital status and to compare the performance of several classical machine learning models. This analysis provides insight into social patterns while demonstrating applied ML methodology.


# %%  
# Load data (as pandas dataframes) 
X = us_census_data_1990.data.features 
y = us_census_data_1990.data.targets

# Combine features and targets (if targets exist)
df = us_census_data_1990.data.original

print(df.shape)        # rows, columns
print(df.head())       # peek at first 5 rows

# %%
# Exploratory Data Analysis (EDA)
# Check dimensions
print(X.shape)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Step 1: Separate features and target ---
X = df.drop(columns=['dTravtime'])
y = df['dTravtime'].astype('category')   # categorical outcome

# --- Step 2: Mark categorical predictors ---
categorical_vars = [
    "iSex", "dOccup", "dIndustry", "dAncstry1", "dAncstry2",
    "dHispanic", "dPOB", "iCitizen", "iEnglish", "iLang1",
    "iMilitary", "iSchool", "iWorklwk", "iDisabl1",
    "iDisabl2", "iMobillim", "iVietnam", "iWWII"
]
numeric_vars = [col for col in X.columns if col not in categorical_vars]

# --- Step 3: EDA ---
# Distribution of commute time categories
sns.countplot(x=y)
plt.xlabel("Commute time category (0–6)")
plt.ylabel("Count")
plt.title("Distribution of Commute Time Categories")
plt.show()

# Compare commute time categories vs age and earnings
sns.boxplot(x=y, y=df['dAge'])
plt.title("Commute Time Category vs Age")
plt.show()

sns.boxplot(x=y, y=df['dRearning'])
plt.title("Commute Time Category vs Earnings")
plt.show()

# %% 
import warnings
warnings.filterwarnings("ignore")  # suppress warnings globally

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import f_oneway, chi2_contingency

# --- Step 1: Sample 10,000 rows ---
df_sample = df.sample(n=10000, random_state=42)

# --- Step 2: Collapse high-cardinality categorical variables ---
def collapse_categories(series, top_n=4):
    top_cats = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_cats), other='Other')

for col in ['dIndustry','dOccup','dAncstry1','dAncstry2','iLang1']:
    if col in df_sample.columns:
        df_sample[col] = collapse_categories(df_sample[col], top_n=4)

# --- Step 3: Target and predictors ---
y = df_sample['dTravtime'].astype('category')
predictors = [col for col in df_sample.columns if col not in ['dTravtime','caseid']]

# --- Step 4: Pre-screen predictors ---
scores = {}
for col in predictors:
    if pd.api.types.is_numeric_dtype(df_sample[col]):
        groups = [df_sample[col][y==cat] for cat in y.cat.categories]
        try:
            _, pval = f_oneway(*groups)
            scores[col] = pval
        except Exception:
            scores[col] = 1.0
    else:
        table = pd.crosstab(df_sample[col].astype(str), y.astype(str))
        try:
            _, pval, _, _ = chi2_contingency(table)
            scores[col] = pval
        except Exception:
            scores[col] = 1.0

# --- Step 5: Select top 20 predictors ---
top_predictors = sorted(scores, key=scores.get)[:20]
print("Top 20 predictors:", top_predictors)

# --- Step 6: One-hot encode ---
X = pd.get_dummies(df_sample[top_predictors], drop_first=True)
y_codes = y.cat.codes

# --- Step 7: Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_codes, test_size=0.2, random_state=42, stratify=y_codes
)

# --- Step 8: Initial model ---
clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", penalty="l2", max_iter=500)
clf.fit(X_train, y_train)
init_acc = accuracy_score(y_test, clf.predict(X_test))
print("Initial accuracy with 20 predictors:", init_acc)

# --- Step 9: Backward stepwise with accuracy tracking ---
def backward_stepwise(X_train, y_train, X_test, y_test, threshold=0.001):
    included = list(X_train.columns)
    acc_history = []
    best_acc = accuracy_score(y_test, clf.predict(X_test))
    acc_history.append((len(included), best_acc, included.copy()))

    improved = True
    while improved and len(included) > 1:
        improved = False
        for col in included:
            trial = [c for c in included if c != col]
            model = LogisticRegression(
                multi_class="multinomial", solver="lbfgs", penalty="l2", max_iter=500
            ).fit(X_train[trial], y_train)
            acc = accuracy_score(y_test, model.predict(X_test[trial]))
            if acc >= best_acc - threshold:
                included.remove(col)
                best_acc = acc
                acc_history.append((len(included), acc, included.copy()))
                print(f"Dropped {col}, accuracy {acc}")
                improved = True
                break
    return acc_history

acc_history = backward_stepwise(X_train, y_train, X_test, y_test)

# --- Step 10: Plot accuracy vs. predictors retained ---
steps, accs, _ = zip(*acc_history)
plt.figure(figsize=(8,5))
plt.plot(steps, accs, marker='o')
plt.gca().invert_xaxis()
plt.xlabel("Number of Predictors Retained")
plt.ylabel("Test Accuracy")
plt.title("Backward Stepwise Selection: Accuracy vs. Predictors")
plt.grid(True)
plt.show()

# --- Step 11: Extract best model (highest accuracy) ---
best_step = max(acc_history, key=lambda x: x[1])
best_vars = best_step[2]
print("Best model predictors:", best_vars)
print("Best model accuracy:", best_step[1])

best_model = LogisticRegression(
    multi_class="multinomial", solver="lbfgs", penalty="l2", max_iter=500
).fit(X_train[best_vars], y_train)

# --- Step 12: Coefficient table for best model ---
coef_df = pd.DataFrame(best_model.coef_, columns=X_train[best_vars].columns)
coef_df.index = [f"Class {c}" for c in best_model.classes_]
print("Coefficient table for best model (highest accuracy):")
print(coef_df)

# We get about 68.7% accuracy with 12 predictors retained, compared to 68.5% with all 20 predictors.

# %%
