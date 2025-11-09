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

# Inspect marital status distribution
print(X['iMarital'].value_counts())

# 0 (Married): 1,095,567 records
# 4 (Never married): 1,022,432 records
# 2 (Divorced): 153,745 records
# 1 (Widowed): 145,497 records
# 3 (Separated): 41,044 records

# Summary statistics
print(X.describe(include='all'))

# Missing values
print(X.isnull().sum().sort_values(ascending=False))

# Visualize marital status vs. age, income, education
import seaborn as sns
sns.countplot(x='iMarital', data=X)

sns.boxplot(x='iMarital', y='dAge', data=X)

sns.boxplot(x='iMarital', y='dRearning', data=X)

sns.violinplot(x='iMarital', y='dRearning', data=X, scale='width')

X.groupby('iMarital')['dRearning'].median().plot(kind='bar')

# %%
import pandas as pd
X_sample = X.sample(n=50000, random_state=42)
y_sample = X_sample['iMarital']
X_sample = X_sample.drop(columns=['iMarital'])

# ANOVA F-test for numeric predictors
from sklearn.feature_selection import f_classif

numeric_cols = X_sample.select_dtypes(include='number').columns
f_scores, p_values = f_classif(X_sample[numeric_cols], y_sample)

f_df = pd.Series(f_scores, index=numeric_cols).sort_values(ascending=False)
print(f_df.head(10))  # Top 10 numeric predictors

# Rank predictors by chi-squared test for categorical predictors
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest

categorical_cols = X_sample.select_dtypes(exclude='number').columns
# Treat these as categorical
categorical_vars = ['iSex', 'dIndustry', 'dOccup', 'dHispanic', 'dPOB']

# Convert to string or category dtype
X_sample[categorical_vars] = X_sample[categorical_vars].astype('category')
X_cat = pd.get_dummies(X_sample[categorical_vars], drop_first=True)

chi_scores, p_values = chi2(X_cat, y_sample)
chi_df = pd.Series(chi_scores, index=X_cat.columns).sort_values(ascending=False)
print(chi_df.head(10))  # Top 10 categorical predictors

# Numeric variables (ANOVA F-scores)
# - dAge (19,254) → Age is by far the strongest predictor of marital status. This makes sense: younger individuals are more likely never married, older individuals more likely married or widowed.
# - iRrelchld (11,631) → Relationship to children in household. Strongly tied to marital status (married people more likely to have children present).
# - iRownchld (9,720) → Own children indicator. Again, highly predictive of being married vs. never married.
# - iRemplpar (8,836) → Employment of parents. Likely correlated with marital status through household structure.
# - Disability indicators (iDisabl1, iDisabl2, iMobillim) → Moderate association. Disability status may correlate with marital status through socioeconomic factors.
# - iWork89 (6,566) → Employment in 1989. Employment history is moderately predictive.
# - iMilitary (5,855) → Military service status. Predictive of marital status (historically, veterans had higher marriage rates).
# Categorical variables (Chi-square scores)
# - dOccup_1 (2,064) → Occupation category 1 strongly associated with marital status.
# - dIndustry_4, dIndustry_9, etc. → Certain industries show strong associations, likely reflecting socioeconomic differences.
# - iSex_1 (709) → Gender is predictive, though less than age or children indicators. Widowed status is more common among women, for example.

# %%
