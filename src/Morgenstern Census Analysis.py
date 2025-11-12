# %%
# %pip install ucimlrepo
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
us_census_data_1990 = fetch_ucirepo(id=116) 

# This project uses the 1990 U.S. Census dataset from the UCI Machine Learning Repository to model commute time to work (dTravtime). The dataset was chosen because it is large, diverse, and rich in demographic and socioeconomic variables, making it ideal for supervised learning. The goal is to identify which attributes (e.g., age, income, education, occupation) are predictive of commute time and to compare the performance of several classical machine learning models. This analysis provides insight into social patterns while demonstrating applied ML methodology.


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

# Define numeric and categorical variables from top features
numeric_features = [
    "dDepart", "iMeans", "dHours", "iClass", "dHour89", "iRiders",
    "dWeek89", "dRearning", "iYearsch", "iYearwrk", "dIncome1",
    "iLooking", "dAge", "dPwgt1", "iRPOB", "iTmpabsnt"
]

categorical_features = [
    "dOccup", "dIndustry", "iWorklwk"
]

# Function to plot numeric features
def plot_numeric_univariate(df, features, bins=30):
    for col in features:
        if col in df.columns:
            plt.figure(figsize=(6,4))
            sns.histplot(df[col], bins=bins, kde=False, color="steelblue")
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.grid(True, alpha=0.3)
            plt.show()

# Function to plot categorical features
def plot_categorical_univariate(df, features, top_n=10):
    for col in features:
        if col in df.columns:
            plt.figure(figsize=(8,4))
            counts = df[col].value_counts().nlargest(top_n)
            sns.barplot(x=counts.index.astype(str), y=counts.values, palette="viridis")
            plt.title(f"Top {top_n} Categories of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.show()

plt.figure(figsize=(8,5))
sns.countplot(x=y, palette="viridis")
plt.xlabel("Commute Time Category (0–6)")
plt.ylabel("Count")
plt.title("Distribution of Commute Time Categories (dTravtime)")
plt.grid(axis="y", alpha=0.3)
plt.show()

# Run univariate plots on your sample dataframe
plot_numeric_univariate(df, numeric_features)
plot_categorical_univariate(df, categorical_features)


# Bivariate EDA 
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

from scipy.stats import f_oneway, chi2_contingency

# Target variable
y = df['dTravtime'].astype('category')

# Separate numeric and categorical predictors
numeric_features = [
    "dDepart","iMeans","dHours","iClass","dHour89","iRiders",
    "dWeek89","dRearning","iYearsch","iYearwrk","dIncome1",
    "iLooking","dAge","dPwgt1","iRPOB","iTmpabsnt"
]

categorical_features = ["dOccup","dIndustry","iWorklwk"]

# ANOVA for numeric predictors vs. commute time
anova_results = {}
for col in numeric_features:
    if col in df.columns:
        groups = [df[col][y==cat] for cat in y.cat.categories]
        try:
            fstat, pval = f_oneway(*groups)
            anova_results[col] = pval
        except Exception:
            anova_results[col] = np.nan

print("ANOVA p-values (numeric predictors vs. commute time):")
print(pd.Series(anova_results).sort_values())

# Chi-square for categorical predictors vs. commute time
chi2_results = {}
for col in categorical_features:
    if col in df.columns:
        table = pd.crosstab(df[col].astype(str), y.astype(str))
        try:
            chi2, pval, _, _ = chi2_contingency(table)
            chi2_results[col] = pval
        except Exception:
            chi2_results[col] = np.nan

print("\nChi-square p-values (categorical predictors vs. commute time):")
print(pd.Series(chi2_results).sort_values())

import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix for numeric features
numeric_df = df[numeric_features].dropna()
corr_matrix = numeric_df.corr(method="pearson")

# Plot heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Pearson Correlation Heatmap (Numeric Predictors)")
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

# Step 1: Sample 50,000 rows
df_sample = df.sample(n=50000, random_state=42)

# Step 2: Collapse high-cardinality categorical variables
def collapse_categories(series, top_n=4):
    top_cats = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_cats), other='Other')

for col in ['dIndustry','dOccup','dAncstry1','dAncstry2','iLang1']:
    if col in df_sample.columns:
        df_sample[col] = collapse_categories(df_sample[col], top_n=4)

# Ensure categorical columns are all strings
for col in ['dIndustry','dOccup','dAncstry1','dAncstry2','iLang1']:
    if col in df_sample.columns:
        df_sample[col] = collapse_categories(df_sample[col], top_n=4).astype(str)

# Also cast other categorical predictors to string
categorical_vars = [
    "iSex", "dOccup", "dIndustry", "dAncstry1", "dAncstry2",
    "dHispanic", "dPOB", "iCitizen", "iEnglish", "iLang1",
    "iMilitary", "iSchool", "iWorklwk", "iDisabl1",
    "iDisabl2", "iMobillim", "iVietnam", "iWWII"
]
for col in categorical_vars:
    if col in df_sample.columns:
        df_sample[col] = df_sample[col].astype(str)

# Step 3: Target and predictors
y = df_sample['dTravtime'].astype('category')
predictors = [col for col in df_sample.columns if col not in ['dTravtime','caseid']]

# Step 4: Pre-screen predictors
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

# Step 5: Select top 20 predictors
top_predictors = sorted(scores, key=scores.get)[:20]
print("Top 20 predictors:", top_predictors)

# Step 6: Build design matrix using top predictors
X = df_sample[top_predictors]
y_codes = y.cat.codes

# Step 7: Balance classes (undersample majority)
counts = Counter(y_codes)
min_count = min(counts.values())
balanced_idx = np.hstack([
    np.random.choice(np.where(y_codes == cls)[0], min_count, replace=False)
    for cls in counts.keys()
])
X_balanced = X.iloc[balanced_idx]
y_balanced = y_codes.iloc[balanced_idx]

print("Balanced class counts:", Counter(y_balanced))

# Step 8: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# Step 9: Build preprocessing pipeline (restricted to available predictors)
def make_pipeline(predictor_list):
    categorical_vars = [col for col in predictor_list if col in [
        "iSex", "dOccup", "dIndustry", "dAncstry1", "dAncstry2",
        "dHispanic", "dPOB", "iCitizen", "iEnglish", "iLang1",
        "iMilitary", "iSchool", "iWorklwk", "iDisabl1",
        "iDisabl2", "iMobillim", "iVietnam", "iWWII"
    ]]
    numeric_vars = [col for col in predictor_list if col not in categorical_vars]

    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_vars),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_vars)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(
            multi_class="multinomial", solver="lbfgs", penalty="l2", max_iter=500))
    ])
    return pipeline

# Initial model
logit_pipeline = make_pipeline(list(X_train.columns))
logit_pipeline.fit(X_train, y_train)
init_acc = accuracy_score(y_test, logit_pipeline.predict(X_test))
print("Initial accuracy with 20 predictors (balanced):", init_acc)

# Step 10: Backward stepwise selection
def backward_stepwise(X_train, y_train, X_test, y_test, threshold=0.001, min_vars=2):
    included = list(X_train.columns)
    acc_history = []
    pipeline = make_pipeline(included)
    pipeline.fit(X_train, y_train)
    best_acc = accuracy_score(y_test, pipeline.predict(X_test))
    acc_history.append((len(included), best_acc, included.copy()))

    improved = True
    while improved and len(included) > min_vars:
        improved = False
        for col in included:
            trial = [c for c in included if c != col]
            trial_pipeline = make_pipeline(trial)
            trial_pipeline.fit(X_train[trial], y_train)
            acc = accuracy_score(y_test, trial_pipeline.predict(X_test[trial]))
            if acc >= best_acc - threshold:
                included.remove(col)
                best_acc = acc
                acc_history.append((len(included), acc, included.copy()))
                print(f"Dropped {col}, accuracy {acc}")
                improved = True
                break
    return acc_history

acc_history = backward_stepwise(X_train, y_train, X_test, y_test, min_vars=2)

# Step 11: Plot accuracy vs. predictors retained
steps, accs, _ = zip(*acc_history)
plt.figure(figsize=(8,5))
plt.plot(steps, accs, marker='o')
plt.gca().invert_xaxis()
plt.xlabel("Number of Predictors Retained")
plt.ylabel("Test Accuracy")
plt.title("Backward Stepwise Selection (Balanced, 50k): Accuracy vs. Predictors")
plt.grid(True)
plt.show()

# Step 12: Extract best model
best_step = max(acc_history, key=lambda x: x[1])
best_vars = best_step[2]
print("Best model predictors:", best_vars)
print("Best model accuracy:", best_step[1])

best_pipeline = make_pipeline(best_vars)
best_pipeline.fit(X_train[best_vars], y_train)

# Coefficient table
best_model = best_pipeline.named_steps["model"]
coef_df = pd.DataFrame(best_model.coef_, columns=best_pipeline.named_steps["preprocess"].get_feature_names_out())
coef_df.index = [f"Class {c}" for c in best_model.classes_]
print("Coefficient table for best model (balanced, 50k):")
print(coef_df)

# ROC curves for best model
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
y_score = best_pipeline.predict_proba(X_test[best_vars])

fpr, tpr, roc_auc = {}, {}, {}
plt.figure(figsize=(8,6))
for i in range(y_test_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

# Step 1: Sample 50,000 rows
df_sample = df.sample(n=50000, random_state=42)

# Step 2: Collapse high-cardinality categorical variables
def collapse_categories(series, top_n=4):
    top_cats = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_cats), other='Other')

for col in ['dIndustry','dOccup','dAncstry1','dAncstry2','iLang1']:
    if col in df_sample.columns:
        df_sample[col] = collapse_categories(df_sample[col], top_n=4).astype(str)

# Cast other categorical predictors to string
categorical_vars_master = [
    "iSex", "dOccup", "dIndustry", "dAncstry1", "dAncstry2",
    "dHispanic", "dPOB", "iCitizen", "iEnglish", "iLang1",
    "iMilitary", "iSchool", "iWorklwk", "iDisabl1",
    "iDisabl2", "iMobillim", "iVietnam", "iWWII"
]
for col in categorical_vars_master:
    if col in df_sample.columns:
        df_sample[col] = df_sample[col].astype(str)

# Step 3: Target and predictors
y = df_sample['dTravtime'].astype('category')
predictors = [col for col in df_sample.columns if col not in ['dTravtime','caseid']]
X = df_sample[predictors]
y_codes = y.cat.codes

# Step 4: Balance classes (undersample majority)
counts = Counter(y_codes)
min_count = min(counts.values())
balanced_idx = np.hstack([
    np.random.choice(np.where(y_codes == cls)[0], min_count, replace=False)
    for cls in counts.keys()
])
X_balanced = X.iloc[balanced_idx]
y_balanced = y_codes.iloc[balanced_idx]

print("Balanced class counts:", Counter(y_balanced))

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# Step 6: Build preprocessing pipeline
def make_pipeline_knn(predictor_list, n_neighbors=5):
    categorical_vars = [col for col in predictor_list if col in categorical_vars_master]
    numeric_vars = [col for col in predictor_list if col not in categorical_vars]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_vars),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_vars)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("pca", PCA(n_components=0.95, random_state=42)),  # retain 95% variance
        ("model", KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance"))
    ])
    return pipeline

# Step 7: Hyperparameter tuning for n_neighbors
results = []
best_model = None
best_f1 = -1

for k in [3, 5, 7, 9]:
    knn_pipeline = make_pipeline_knn(list(X_train.columns), n_neighbors=k)
    knn_pipeline.fit(X_train, y_train)
    y_pred = knn_pipeline.predict(X_test)
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
        best_model = knn_pipeline

# Step 8: ROC curves for best model
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
# Random Forest with standardized preprocessing
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

# Step 1: Sample 100,000 rows (RF can handle larger sample sizes)
df_sample = df.sample(n=100000, random_state=42)

# Step 2: Collapse high-cardinality categorical variables
def collapse_categories(series, top_n=4):
    top_cats = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_cats), other='Other')

for col in ['dIndustry','dOccup','dAncstry1','dAncstry2','iLang1']:
    if col in df_sample.columns:
        df_sample[col] = collapse_categories(df_sample[col], top_n=4).astype(str)

# Cast other categorical predictors to string
categorical_vars_master = [
    "iSex", "dOccup", "dIndustry", "dAncstry1", "dAncstry2",
    "dHispanic", "dPOB", "iCitizen", "iEnglish", "iLang1",
    "iMilitary", "iSchool", "iWorklwk", "iDisabl1",
    "iDisabl2", "iMobillim", "iVietnam", "iWWII"
]
for col in categorical_vars_master:
    if col in df_sample.columns:
        df_sample[col] = df_sample[col].astype(str)

# Step 3: Target and predictors
y = df_sample['dTravtime'].astype('category')
predictors = [col for col in df_sample.columns if col not in ['dTravtime','caseid']]
X = df_sample[predictors]
y_codes = y.cat.codes

# Step 4: Balance classes (undersample majority)
counts = Counter(y_codes)
min_count = min(counts.values())
balanced_idx = np.hstack([
    np.random.choice(np.where(y_codes == cls)[0], min_count, replace=False)
    for cls in counts.keys()
])
X_balanced = X.iloc[balanced_idx]
y_balanced = y_codes.iloc[balanced_idx]

print("Balanced class counts:", Counter(y_balanced))

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# Step 6: Build preprocessing + RF pipeline
def make_pipeline_rf(predictor_list, n_estimators=200, max_depth=15):
    categorical_vars = [col for col in predictor_list if col in categorical_vars_master]
    numeric_vars = [col for col in predictor_list if col not in categorical_vars]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_vars),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_vars)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        ))
    ])
    return pipeline

# Step 7: Fit and evaluate
rf_pipeline = make_pipeline_rf(list(X_train.columns))
rf_pipeline.fit(X_train, y_train)
y_pred = rf_pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
print("\nRandom Forest Results")
print("Test accuracy:", acc)
print("Macro F1:", macro_f1)
print("Classification report:")
print(classification_report(y_test, y_pred))

# Step 8: ROC curves
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
y_score = rf_pipeline.predict_proba(X_test)

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

# Step 9: Feature importance
model = rf_pipeline.named_steps["model"]
importances = model.feature_importances_
feature_names = rf_pipeline.named_steps["preprocess"].get_feature_names_out()

feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
feat_imp = feat_imp.sort_values("importance", ascending=False).head(20)
print("\nTop 20 Feature Importances:")
print(feat_imp)

# %%
# Gradient Boosting
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from collections import Counter

# Step 1: Sample 100,000 rows (XGB can handle larger sample sizes)
df_sample = df.sample(n=100000, random_state=42)

# Step 2: Collapse high-cardinality categorical variables
def collapse_categories(series, top_n=4):
    top_cats = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_cats), other='Other')

for col in ['dIndustry','dOccup','dAncstry1','dAncstry2','iLang1']:
    if col in df_sample.columns:
        df_sample[col] = collapse_categories(df_sample[col], top_n=4).astype(str)

# Cast other categorical predictors to string
categorical_vars_master = [
    "iSex", "dOccup", "dIndustry", "dAncstry1", "dAncstry2",
    "dHispanic", "dPOB", "iCitizen", "iEnglish", "iLang1",
    "iMilitary", "iSchool", "iWorklwk", "iDisabl1",
    "iDisabl2", "iMobillim", "iVietnam", "iWWII"
]
for col in categorical_vars_master:
    if col in df_sample.columns:
        df_sample[col] = df_sample[col].astype(str)

# Step 3: Target and predictors
y = df_sample['dTravtime'].astype('category')
predictors = [col for col in df_sample.columns if col not in ['dTravtime','caseid']]
X = df_sample[predictors]
y_codes = y.cat.codes

# Step 4: Balance classes (undersample majority)
counts = Counter(y_codes)
min_count = min(counts.values())
balanced_idx = np.hstack([
    np.random.choice(np.where(y_codes == cls)[0], min_count, replace=False)
    for cls in counts.keys()
])
X_balanced = X.iloc[balanced_idx]
y_balanced = y_codes.iloc[balanced_idx]

print("Balanced class counts:", Counter(y_balanced))

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# Step 6: Build preprocessing + XGB pipeline
def make_pipeline_xgb(predictor_list, n_estimators=300, learning_rate=0.1, max_depth=6):
    categorical_vars = [col for col in predictor_list if col in categorical_vars_master]
    numeric_vars = [col for col in predictor_list if col not in categorical_vars]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_vars),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_vars)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss"
        ))
    ])
    return pipeline

# Step 7: Fit and evaluate
xgb_pipeline = make_pipeline_xgb(list(X_train.columns))
xgb_pipeline.fit(X_train, y_train)
y_pred = xgb_pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
print("\nXGBoost Results")
print("Test accuracy:", acc)
print("Macro F1:", macro_f1)
print("Classification report:")
print(classification_report(y_test, y_pred))

# Step 8: ROC curves
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
y_score = xgb_pipeline.predict_proba(X_test)

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

# Step 9: Feature importance
model = xgb_pipeline.named_steps["model"]
importances = model.feature_importances_
feature_names = xgb_pipeline.named_steps["preprocess"].get_feature_names_out()

feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
feat_imp = feat_imp.sort_values("importance", ascending=False).head(20)
print("\nTop 20 Feature Importances:")
print(feat_imp)

# %%
# SVM with PCA
# Support Vector Machine (SVM) with standardized preprocessing
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from collections import Counter

# Step 1: Sample 50,000 rows (SVM is computationally heavier)
df_sample = df.sample(n=50000, random_state=42)

# Step 2: Collapse high-cardinality categorical variables
def collapse_categories(series, top_n=4):
    top_cats = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_cats), other='Other')

for col in ['dIndustry','dOccup','dAncstry1','dAncstry2','iLang1']:
    if col in df_sample.columns:
        df_sample[col] = collapse_categories(df_sample[col], top_n=4).astype(str)

# Cast other categorical predictors to string
categorical_vars_master = [
    "iSex", "dOccup", "dIndustry", "dAncstry1", "dAncstry2",
    "dHispanic", "dPOB", "iCitizen", "iEnglish", "iLang1",
    "iMilitary", "iSchool", "iWorklwk", "iDisabl1",
    "iDisabl2", "iMobillim", "iVietnam", "iWWII"
]
for col in categorical_vars_master:
    if col in df_sample.columns:
        df_sample[col] = df_sample[col].astype(str)

# Step 3: Target and predictors
y = df_sample['dTravtime'].astype('category')
predictors = [col for col in df_sample.columns if col not in ['dTravtime','caseid']]
X = df_sample[predictors]
y_codes = y.cat.codes

# Step 4: Balance classes (undersample majority)
counts = Counter(y_codes)
min_count = min(counts.values())
balanced_idx = np.hstack([
    np.random.choice(np.where(y_codes == cls)[0], min_count, replace=False)
    for cls in counts.keys()
])
X_balanced = X.iloc[balanced_idx]
y_balanced = y_codes.iloc[balanced_idx]

print("Balanced class counts:", Counter(y_balanced))

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# Step 6: Build preprocessing + PCA + SVM pipeline
def make_pipeline_svm(predictor_list, C=1.0, gamma="scale"):
    categorical_vars = [col for col in predictor_list if col in categorical_vars_master]
    numeric_vars = [col for col in predictor_list if col not in categorical_vars]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_vars),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_vars)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("pca", PCA(n_components=0.95, random_state=42)),  # retain 95% variance
        ("model", SVC(kernel="rbf", C=C, gamma=gamma, probability=True, random_state=42))
    ])
    return pipeline

# Step 7: Fit and evaluate
svm_pipeline = make_pipeline_svm(list(X_train.columns))
svm_pipeline.fit(X_train, y_train)
y_pred = svm_pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
print("\nSVM Results")
print("Test accuracy:", acc)
print("Macro F1:", macro_f1)
print("Classification report:")
print(classification_report(y_test, y_pred))

# Step 8: ROC curves
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
y_score = svm_pipeline.predict_proba(X_test)

fpr, tpr, roc_auc = {}, {}, {}
plt.figure(figsize=(8,6))
for i in range(y_test_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC (OvR) for SVM")
plt.legend()
plt.show()

# %%
# Compare all models
from sklearn.metrics import accuracy_score, f1_score

def compare_models(X_train, X_test, y_train, y_test):
    results = []

    #  Logistic Regression with backward stepwise selection 
    acc_history = backward_stepwise(X_train, y_train, X_test, y_test, min_vars=2)
    best_step = max(acc_history, key=lambda x: x[1])
    best_vars = best_step[2]
    best_pipeline = make_pipeline(best_vars)
    best_pipeline.fit(X_train[best_vars], y_train)
    y_pred = best_pipeline.predict(X_test[best_vars])
    results.append({
        "Model": "Multinomial Logistic Regression (Stepwise)",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro F1": f1_score(y_test, y_pred, average="macro")
    })

    #  KNN 
    knn_pipeline = make_pipeline_knn(list(X_train.columns), n_neighbors=5)
    knn_pipeline.fit(X_train, y_train)
    y_pred = knn_pipeline.predict(X_test)
    results.append({
        "Model": "KNN (k=5)",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro F1": f1_score(y_test, y_pred, average="macro")
    })

    #  Random Forest 
    rf_pipeline = make_pipeline_rf(list(X_train.columns))
    rf_pipeline.fit(X_train, y_train)
    y_pred = rf_pipeline.predict(X_test)
    results.append({
        "Model": "Random Forest",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro F1": f1_score(y_test, y_pred, average="macro")
    })

    #  XGBoost 
    xgb_pipeline = make_pipeline_xgb(list(X_train.columns))
    xgb_pipeline.fit(X_train, y_train)
    y_pred = xgb_pipeline.predict(X_test)
    results.append({
        "Model": "XGBoost",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro F1": f1_score(y_test, y_pred, average="macro")
    })

    #  SVM 
    svm_pipeline = make_pipeline_svm(list(X_train.columns))
    svm_pipeline.fit(X_train, y_train)
    y_pred = svm_pipeline.predict(X_test)
    results.append({
        "Model": "SVM (RBF)",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro F1": f1_score(y_test, y_pred, average="macro")
    })

    # Convert to DataFrame for readability
    results_df = pd.DataFrame(results)
    return results_df

results_df = compare_models(X_train, X_test, y_train, y_test)
print(results_df)
# %%
