# %%
# %pip install ucimlrepo
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
us_census_data_1990 = fetch_ucirepo(id=116) 

# %%  
# data (as pandas dataframes) 
X = us_census_data_1990.data.features 
y = us_census_data_1990.data.targets

# Combine features and targets (if targets exist)
df = us_census_data_1990.data.original

print(df.shape)        # rows, columns
print(df.head())       # peek at first 5 rows