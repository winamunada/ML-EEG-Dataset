# Import Library

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 99)

import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
!pip install catboost
from catboost import CatBoostClassifier
sns.set()

df = pd.read_csv('Dataset/EEG.machinelearing_data_BRMH.csv')
df.head()
df["main.disorder"].unique()
df["specific.disorder"].unique()

#function to rename
def reformat_name(name):
    '''
    reformat from XX.X.band.x.channel to band.channel or
    COH.X.band.x.channel1.x.channel2 to COH.band.channel1.channel2
    '''
    splitted = name.split(sep='.')
    if len(splitted) < 5:
        return name
    if splitted[0] != 'COH':
        result = f'{splitted[2]}.{splitted[4]}'
    else:
        result = f'{splitted[0]}.{splitted[2]}.{splitted[4]}.{splitted[6]}'
    return result
# rename columns
df.rename(reformat_name, axis=1, inplace=True)
df
typo_ind = df[df['specific.disorder'] == 'Obsessive compulsitve disorder'].index
df.loc[typo_ind, 'specific.disorder'] = 'Obsessive compulsive disorder'

# Missing Data
missing = df.isna().sum()
sep_col = missing[missing == df.shape[0]].index[0]
sep_col

educ_na = df[df['education'].isna()]
iq_na = df[df['IQ'].isna()]
educ_iq_na = pd.concat([educ_na, iq_na]).drop_duplicates()
educ_iq_na

drop_md = educ_iq_na['specific.disorder'].value_counts().sort_index()
all_md = df['specific.disorder'].value_counts().sort_index()[drop_md.index]
pd.concat([all_md, drop_md/all_md * 100], axis=1).set_axis(['all_data', 'na_percentage'], axis=1).sort_values('na_percentage', ascending=False)

check_missing = df.isnull().sum() * 100 / df.shape[0]
check_missing[check_missing > 0].sort_values(ascending=False)

# Gets the data type of each column
data_type_column = df.dtypes

# Count the number of missing values ​​in each column
missing_values = df.isnull().sum()

# Calculate the percentage of missing values ​​in each column
percentage_missing_values = (missing_values / len(df)) * 100

# Combine these results in a DataFrame
summary = pd.DataFrame({'Data Type': data_type_column, 'Total Missing Value': missing_values, 'Percentage of Missing Value': percentage_missing_values})

# Filter only columns that have missing values
column_with_missing_value = summary[summary['Total Missing Value'] > 0]

print(column_with_missing_value)

for kolom in column_with_missing_value.index:
  df[kolom].fillna(df[kolom].median(), inplace=True)
df
df.groupby('specific.disorder').size()
df.drop_duplicates(inplace=True)
df.shape
df.select_dtypes(include='object').nunique()
df_fix=df.copy()
df_fix.head()
df_fix=df_fix.drop([sep_col, 'no.', 'eeg.date','main.disorder'], axis=1).copy(deep=True)
df_fix.dtypes

# Feature Scaling and Transformation
# One Hot Encoding
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
df_fix[['specific.disorder']] = enc.fit_transform(df_fix[['specific.disorder']])
df_fix
df_fix.groupby('specific.disorder').size()
Integer = ['age','education','IQ','specific.disorder']
for item in Integer :
    df_fix[item] = df_fix[item].astype(int)
categorical_cols = [col for col in df_fix.select_dtypes(include='object').columns.tolist()]
onehot = pd.get_dummies(df_fix[categorical_cols], drop_first=True)
onehot
float_cols = [col for col in df_fix.select_dtypes(include='float').columns.tolist()]
float_cols

# Standardization
# All columns of type 'float' undergo a standardization process with StandardScaler.
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
std = pd.DataFrame(ss.fit_transform(df_fix[float_cols]), columns=float_cols)

std.head()

# Combine the data
data_model = pd.concat([onehot, std, df_fix.select_dtypes(include='int')], axis=1)
data_model.head()

# Modelling
from imblearn.over_sampling import SMOTE
import pandas as pd

# Using raw_data
X = data_model.drop(columns=['specific.disorder'])  # Fitur-fitur
y = data_model['specific.disorder']  # Target

# Create a SMOTE object
smote = SMOTE(random_state=42)

# Resample data
X_resampled, y_resampled = smote.fit_resample(X, y)

# Recombine the resampled features and targets
data_resampled = pd.concat([X_resampled, y_resampled], axis=1)

X_resampled.count()
data_resampled
data_resampled.groupby('specific.disorder').size()

from sklearn.model_selection import train_test_split

# Splitting resampled data into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
X_train.shape, X_test.shape

# CATBOOST
categorical_features = np.where(X.dtypes != float)[0]

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    eval_metric='Accuracy',
    random_seed=42,
    logging_level='Verbose',
    cat_features=categorical_features,
    early_stopping_rounds=50  # Stop if no improvement after 50 rounds
)

model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=True)

y_predcat = model.predict(X_test)
accuracy=accuracy_score(y_predcat, y_test)
print('CatBoost Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_predcat)))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predcat))

y,levels = pd.factorize(df['specific.disorder'])
y_predcat_re=y_predcat.reshape(-1)
cnb = pd.crosstab(levels[y_test],levels[y_predcat_re])
fig, ax = plt.subplots(figsize=(15, 5))
nb = sns.heatmap(cnb, annot=True, fmt=".0f", ax=ax, cmap="Oranges")
nb.set(xlabel ="Prediction", ylabel = "Actual")
nb.set_title("CatBoost", fontsize =15, weight='bold')

# TUNING
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Define the parameter grid
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5, 7],
    'iterations': [500, 1000, 1500],
    'early_stopping_rounds': [50]  # Early stopping rounds
}

# Initialize CatBoostClassifier
model = CatBoostClassifier(eval_metric='Accuracy', random_seed=42, logging_level='Verbose', cat_features=categorical_features)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=3, n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=True)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Train CatBoostClassifier with the best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=True)

# Predict on the test set
y_pred_best = best_model.predict(X_test)


accuracy=accuracy_score(y_pred_best, y_test)
print('CatBoost Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred_best)))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_best))

y,levels = pd.factorize(df['specific.disorder'])
y_predcat_re=y_predcat.reshape(-1)
cnb = pd.crosstab(levels[y_test],levels[y_pred_best])
fig, ax = plt.subplots(figsize=(15, 5))
nb = sns.heatmap(cnb, annot=True, fmt=".0f", ax=ax, cmap="Oranges")
nb.set(xlabel ="Prediction", ylabel = "Actual")
nb.set_title("CatBoost", fontsize =15, weight='bold')