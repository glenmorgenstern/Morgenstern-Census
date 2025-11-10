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

#  Step 1: Separate features and target 
X = df.drop(columns=['dTravtime'])
y = df['dTravtime'].astype('category')   # categorical outcome

#  Step 2: Mark categorical predictors 
categorical_vars = [
    "iSex", "dOccup", "dIndustry", "dAncstry1", "dAncstry2",
    "dHispanic", "dPOB", "iCitizen", "iEnglish", "iLang1",
    "iMilitary", "iSchool", "iWorklwk", "iDisabl1",
    "iDisabl2", "iMobillim", "iVietnam", "iWWII"
]
numeric_vars = [col for col in X.columns if col not in categorical_vars]

#  Step 3: EDA 
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
# Multinomial Logistic Regression with Backward Stepwise Selection
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import f_oneway, chi2_contingency
from collections import Counter

#  Step 1: Sample 50,000 rows 
df_sample = df.sample(n=50000, random_state=42)

#  Step 2: Collapse high-cardinality categorical variables 
def collapse_categories(series, top_n=4):
    top_cats = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_cats), other='Other')

for col in ['dIndustry','dOccup','dAncstry1','dAncstry2','iLang1']:
    if col in df_sample.columns:
        df_sample[col] = collapse_categories(df_sample[col], top_n=4)

#  Step 3: Target and predictors 
y = df_sample['dTravtime'].astype('category')
predictors = [col for col in df_sample.columns if col not in ['dTravtime','caseid']]

#  Step 4: Pre-screen predictors 
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

#  Step 5: Select top 20 predictors 
top_predictors = sorted(scores, key=scores.get)[:20]
print("Top 20 predictors:", top_predictors)

#  Step 6: One-hot encode 
X = pd.get_dummies(df_sample[top_predictors], drop_first=True)
y_codes = y.cat.codes

#  Step 7: Balance classes (undersample majority) 
counts = Counter(y_codes)
min_count = min(counts.values())
balanced_idx = np.hstack([
    np.random.choice(np.where(y_codes==cls)[0], min_count, replace=False)
    for cls in counts.keys()
])
X_balanced = X.iloc[balanced_idx]
y_balanced = y_codes.iloc[balanced_idx]

print("Balanced class counts:", Counter(y_balanced))

#  Step 8: Train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

#  Step 9: Initial model 
clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", penalty="l2", max_iter=500)
clf.fit(X_train, y_train)
init_acc = accuracy_score(y_test, clf.predict(X_test))
print("Initial accuracy with 20 predictors (balanced):", init_acc)

#  Step 10: Backward stepwise selection (down to 2 predictors) 
def backward_stepwise(X_train, y_train, X_test, y_test, threshold=0.001, min_vars=2):
    included = list(X_train.columns)
    acc_history = []
    best_acc = accuracy_score(y_test, clf.predict(X_test))
    acc_history.append((len(included), best_acc, included.copy()))

    improved = True
    while improved and len(included) > min_vars:
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

acc_history = backward_stepwise(X_train, y_train, X_test, y_test, min_vars=2)

#  Step 11: Plot accuracy vs. predictors retained 
steps, accs, _ = zip(*acc_history)
plt.figure(figsize=(8,5))
plt.plot(steps, accs, marker='o')
plt.gca().invert_xaxis()
plt.xlabel("Number of Predictors Retained")
plt.ylabel("Test Accuracy")
plt.title("Backward Stepwise Selection (Balanced, 50k): Accuracy vs. Predictors")
plt.grid(True)
plt.show()

#  Step 12: Extract best model 
best_step = max(acc_history, key=lambda x: x[1])
best_vars = best_step[2]
print("Best model predictors:", best_vars)
print("Best model accuracy:", best_step[1])

best_model = LogisticRegression(
    multi_class="multinomial", solver="lbfgs", penalty="l2", max_iter=500
).fit(X_train[best_vars], y_train)

coef_df = pd.DataFrame(best_model.coef_, columns=X_train[best_vars].columns)
coef_df.index = [f"Class {c}" for c in best_model.classes_]
print("Coefficient table for best model (balanced, 50k):")
print(coef_df)

# ROC curves for best model
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

#  Step 1: Binarize labels for OvR 
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

#  Step 2: Get predicted probabilities from best model 
y_score = best_model.predict_proba(X_test[best_vars])

#  Step 3: Compute ROC curves for each class 
fpr, tpr, roc_auc = {}, {}, {}
plt.figure(figsize=(8,6))

for i in range(y_test_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

#  Step 4: Plot baseline and finalize 
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC (OvR) for Best Multinomial Logistic Regression")
plt.legend()
plt.show()

# We get between 32 and 39% accuracy with 12 predictors retained depending on the sampling, compared to a lower accuracy with all 20 predictors. We have not done PCA at this stage so we get the advantage of interpretability that multinomial logistic regression provides. We can see that the final multinomial model is very good at predicing the shortest commute time category (0) with AUC of 1.0, while performance for other categories is more modest (AUCs around 0.6-0.75). This suggests that while the model captures some patterns in the data, there is a lot of room for improvement to capture patterns in the other classes.

# Multinomial logistic regression has served us well as a transparent baseline. It forces careful preprocessing, highlights which predictors carry signal, and produces interpretable coefficients that map cleanly to commute time categories. The backward stepwise selection and regularization helped tame collinearity and dimensionality, giving us a leaner model without sacrificing accuracy.
# That said, the approach has limitations. It assumes linear relationships in log‑odds space, struggles with very high‑cardinality categorical variables even after collapsing, and can plateau in predictive power once the most obvious predictors are included. Runtime and convergence issues also become more pronounced as the feature space grows.
# In short, multinomial logit is a strong pedagogical and diagnostic tool for understanding variable importance, but as we move forward, more flexible methods (KNN, gradient boosting, etc.) will capture nonlinearities and complex interactions that this model can’t.

# %%
# K-Nearest Neighbors (KNN) with PCA
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from collections import Counter

# Step 1: Sample 50,000 rows
df_sample = df.sample(n=50000, random_state=42)

# Step 2: Collapse high-cardinality categorical variables
def collapse_categories(series, top_n=4):
    top_cats = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_cats), other='Other')

for col in ['dIndustry','dOccup','dAncstry1','dAncstry2','iLang1']:
    if col in df_sample.columns:
        df_sample[col] = collapse_categories(df_sample[col], top_n=4)

# Step 3: Target and predictors
y = df_sample['dTravtime'].astype('category')
predictors = [col for col in df_sample.columns if col not in ['dTravtime','caseid']]

# Step 4: One-hot encode
X = pd.get_dummies(df_sample[predictors], drop_first=True)
y_codes = y.cat.codes

# Step 5: Balance classes (undersample majority)
counts = Counter(y_codes)
min_count = min(counts.values())
balanced_idx = np.hstack([
    np.random.choice(np.where(y_codes==cls)[0], min_count, replace=False)
    for cls in counts.keys()
])
X_balanced = X.iloc[balanced_idx]
y_balanced = y_codes.iloc[balanced_idx]

print("Balanced class counts:", Counter(y_balanced))

# Step 6: Scale predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)

# Step 7: PCA (retain 95% variance)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("Number of PCA components retained:", X_pca.shape[1])

# --- PCA Visualization 1: Cumulative explained variance ---
cum_var = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(8,5))
plt.plot(range(1, len(cum_var)+1), cum_var, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance by PCA Components")
plt.legend()
plt.grid(True)
plt.show()

# --- PCA Visualization 2: Scatterplot of first two PCs ---
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y_balanced, cmap='tab10', alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Scatterplot of First Two Principal Components")
plt.colorbar(scatter, label="Commute Category")
plt.show()

# Step 8: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# Step 9: Hyperparameter tuning for n_neighbors
results = []
best_model = None
best_f1 = -1

for k in [3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    results.append((k, acc, macro_f1))
    print(f"\nKNN (k={k})")
    print("Test accuracy:", acc)
    print("Macro F1:", macro_f1)
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_model = knn

# Step 10: ROC curves for best model
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
y_score = best_model.predict_proba(X_test)

fpr, tpr, roc_auc = {}, {}, {}
plt.figure(figsize=(8,6))
for i in range(y_test_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC (OvR) for Best Balanced KNN Model")
plt.legend()
plt.show()

# Our PCA cumulative explained variance plot shows that nearly 50 components are needed to capture 95% of the variance, indicating a complex feature space. That is, any one feature does not explain a bulk of the variance (the first one only explains around 10%). The scatterplot of the first two PCs reveals some clustering by commute time category, but with significant overlap, suggesting that linear separability is limited.
# We see that k=5 gives the best test accuracy of around 30% after PCA dimensionality reduction, which is comparable to and even slightly worse than the multinomial logistic regression model earlier. The ROC curves show moderate AUC values around 0.5-0.6 for most classes, indicating the model captures some patterns but isn't really an improvement on multinomial regression. KNN benefits from PCA by reducing noise and focusing on key variance directions, but it still struggles with the complexity of predicting multiple commute time categories.

# %%
# Single Tree, pruned
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from collections import Counter

#  Step 1: Sample 100,000 rows 
df_sample = df.sample(n=100000, random_state=42)

#  Step 2: Collapse high-cardinality categorical variables 
def collapse_categories(series, top_n=4):
    top_cats = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_cats), other='Other')

for col in ['dIndustry','dOccup','dAncstry1','dAncstry2','iLang1']:
    if col in df_sample.columns:
        df_sample[col] = collapse_categories(df_sample[col], top_n=4)

#  Step 3: Target and predictors 
y = df_sample['dTravtime'].astype('category')
predictors = [col for col in df_sample.columns if col not in ['dTravtime','caseid']]

#  Step 4: One-hot encode 
X = pd.get_dummies(df_sample[predictors], drop_first=True)
y_codes = y.cat.codes

#  Step 5: Balance classes (undersample majority) 
counts = Counter(y_codes)
min_count = min(counts.values())
balanced_idx = np.hstack([
    np.random.choice(np.where(y_codes==cls)[0], min_count, replace=False)
    for cls in counts.keys()
])
X_balanced = X.iloc[balanced_idx]
y_balanced = y_codes.iloc[balanced_idx]

print("Balanced class counts:", Counter(y_balanced))

#  Step 6: Train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

#  Step 7: Grid search for pruning 
param_grid = {
    "max_depth": [5, 10, 15],
    "min_samples_leaf": [5, 10, 20],
    "min_samples_split": [10, 20]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="f1_macro"
)
grid.fit(X_train, y_train)

best_tree = grid.best_estimator_
print("Best parameters:", grid.best_params_)
print("Best macro F1 (CV):", grid.best_score_)

#  Step 8: Evaluate best tree 
y_pred = best_tree.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Decision Tree test accuracy:", acc)
print("Classification report:")
print(classification_report(y_test, y_pred))

#  Step 9: ROC curves (OvR) 
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
y_score = best_tree.predict_proba(X_test)

fpr, tpr, roc_auc = {}, {}, {}
plt.figure(figsize=(8,6))
for i in range(y_test_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC (OvR) for Tuned Decision Tree")
plt.legend()
plt.show()

# %%
# Random Forest
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from collections import Counter

#  Step 1: Sample 100,000 rows 
df_sample = df.sample(n=100000, random_state=42)

#  Step 2: Collapse high-cardinality categorical variables 
def collapse_categories(series, top_n=4):
    top_cats = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_cats), other='Other')

for col in ['dIndustry','dOccup','dAncstry1','dAncstry2','iLang1']:
    if col in df_sample.columns:
        df_sample[col] = collapse_categories(df_sample[col], top_n=4)

#  Step 3: Target and predictors 
y = df_sample['dTravtime'].astype('category')
predictors = [col for col in df_sample.columns if col not in ['dTravtime','caseid']]

#  Step 4: One-hot encode 
X = pd.get_dummies(df_sample[predictors], drop_first=True)
y_codes = y.cat.codes

#  Step 5: Balance classes (undersample majority) 
counts = Counter(y_codes)
min_count = min(counts.values())
balanced_idx = np.hstack([
    np.random.choice(np.where(y_codes==cls)[0], min_count, replace=False)
    for cls in counts.keys()
])
X_balanced = X.iloc[balanced_idx]
y_balanced = y_codes.iloc[balanced_idx]

print("Balanced class counts:", Counter(y_balanced))

#  Step 6: Train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

#  Step 7: Hyperparameter tuning with CV 
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_leaf": [5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=3,
    scoring="f1_macro"
)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
print("Best parameters:", grid.best_params_)
print("Best macro F1 (CV):", grid.best_score_)

#  Step 8: Evaluate best model 
y_pred = best_rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Random Forest test accuracy:", acc)
print("Classification report:")
print(classification_report(y_test, y_pred))

#  Step 9: ROC curves (OvR) 
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
y_score = best_rf.predict_proba(X_test)

fpr, tpr, roc_auc = {}, {}, {}
plt.figure(figsize=(8,6))
for i in range(y_test_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC (OvR) for Random Forest")
plt.legend()
plt.show()

#  Step 10: Feature importance plot 
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1][:20]  # top 20 features

plt.figure(figsize=(10,6))
plt.bar(range(len(indices)), importances[indices], align="center")
plt.xticks(range(len(indices)), [X_train.columns[i] for i in indices], rotation=90)
plt.title("Top 20 Feature Importances (Random Forest)")
plt.show()

# %%
# Gradient Boosting
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from collections import Counter
from xgboost import XGBClassifier

#  Step 1: Sample 100,000 rows 
df_sample = df.sample(n=100000, random_state=42)

#  Step 2: Collapse high-cardinality categorical variables 
def collapse_categories(series, top_n=4):
    top_cats = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_cats), other='Other')

for col in ['dIndustry','dOccup','dAncstry1','dAncstry2','iLang1']:
    if col in df_sample.columns:
        df_sample[col] = collapse_categories(df_sample[col], top_n=4)

#  Step 3: Target and predictors 
y = df_sample['dTravtime'].astype('category')
predictors = [c for c in df_sample.columns if c not in ['dTravtime','caseid']]

#  Step 4: One-hot encode 
X = pd.get_dummies(df_sample[predictors], drop_first=True)
y_codes = y.cat.codes

#  Step 5: Balance classes (undersample majority) 
counts = Counter(y_codes)
min_count = min(counts.values())
balanced_idx = np.hstack([
    np.random.choice(np.where(y_codes==cls)[0], min_count, replace=False)
    for cls in counts.keys()
])
X_balanced = X.iloc[balanced_idx]
y_balanced = y_codes.iloc[balanced_idx]

print("Balanced class counts:", Counter(y_balanced))

#  Step 6: Train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

#  Step 7: Hyperparameter tuning with CV 
param_grid = {
    "n_estimators": [300, 500],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.7, 1.0]
}

xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=len(np.unique(y_balanced)),
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)

grid = GridSearchCV(
    xgb,
    param_grid,
    cv=3,
    scoring="f1_macro",
    n_jobs=-1
)
grid.fit(X_train, y_train)

best_xgb = grid.best_estimator_
print("Best parameters:", grid.best_params_)
print("Best macro F1 (CV):", grid.best_score_)

#  Step 8: Evaluate best model 
y_pred = best_xgb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("XGBoost test accuracy:", acc)
print("Classification report:")
print(classification_report(y_test, y_pred))

#  Step 9: ROC curves (OvR) 
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
y_score = best_xgb.predict_proba(X_test)

fpr, tpr, roc_auc = {}, {}, {}
plt.figure(figsize=(8,6))
for i in range(y_test_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC (OvR) for XGBoost")
plt.legend()
plt.show()

import matplotlib.pyplot as plt
from xgboost import plot_importance

# Plot top 20 features ranked by gain
plt.figure(figsize=(10,8))
plot_importance(best_xgb, importance_type='gain', max_num_features=20, height=0.6)
plt.title("Top 20 Feature Importances by Gain (XGBoost)")
plt.tight_layout()
plt.show()

# %%
# SVM with PCA
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, f1_score
import matplotlib.pyplot as plt
from collections import Counter

# --- Step 1: Sample 100,000 rows ---
df_sample = df.sample(n=100000, random_state=42)

# --- Step 2: Collapse high-cardinality categorical variables ---
def collapse_categories(series, top_n=4):
    top_cats = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_cats), other='Other')

for col in ['dIndustry','dOccup','dAncstry1','dAncstry2','iLang1']:
    if col in df_sample.columns:
        df_sample[col] = collapse_categories(df_sample[col], top_n=4)

# --- Step 3: Target and predictors ---
y = df_sample['dTravtime'].astype('category')
predictors = [c for c in df_sample.columns if c not in ['dTravtime','caseid']]

# --- Step 4: One-hot encode ---
X = pd.get_dummies(df_sample[predictors], drop_first=True)
y_codes = y.cat.codes

# --- Step 5: Balance classes (undersample majority) ---
counts = Counter(y_codes)
min_count = min(counts.values())
balanced_idx = np.hstack([
    np.random.choice(np.where(y_codes==cls)[0], min_count, replace=False)
    for cls in counts.keys()
])
X_balanced = X.iloc[balanced_idx]
y_balanced = y_codes.iloc[balanced_idx]

print("Balanced class counts:", Counter(y_balanced))

# --- Step 6: Scale predictors ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)

# --- Step 7: PCA (retain 95% variance) ---
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("Number of PCA components retained:", X_pca.shape[1])

# --- Step 8: Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# --- Step 9: Expanded hyperparameter grid for SVM (RBF kernel) ---
param_grid = {
    "C": [0.1, 0.5, 1, 2, 5, 10, 20, 50],
    "gamma": [0.001, 0.005, 0.01, 0.05, 0.1, "scale"]
}

svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)

grid = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=3,
    scoring="f1_macro",
    n_jobs=-1
)
grid.fit(X_train, y_train)

best_svm = grid.best_estimator_
print("Best parameters:", grid.best_params_)
print("Best macro F1 (CV):", grid.best_score_)

# --- Step 10: Evaluate best model ---
y_pred = best_svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("SVM (RBF) test accuracy:", acc)
print("Classification report:")
print(classification_report(y_test, y_pred))

# --- Step 11: ROC curves (OvR) ---
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
y_score = best_svm.predict_proba(X_test)

fpr, tpr, roc_auc = {}, {}, {}
plt.figure(figsize=(8,6))
for i in range(y_test_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC (OvR) for Tuned SVM (RBF + PCA)")
plt.legend()
plt.show()

# %%

