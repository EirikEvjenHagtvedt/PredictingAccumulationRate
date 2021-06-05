import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 


data = pd.read_excel(r'C:\Patch where file is stored\ Name of file.xlsx')

# For further analysis we only take the datapoints, where we have information on the filling
idx = data['Filling'].dropna().index
df = data.loc[idx]

df = df.drop(['Filling/time', 'Precip/time', 'U1', 'U2', 'Direction', 'Routine'], axis=1)

df[['Time', 'Filling', 'Precipitation']].sort_values(by='Time')

df['dt_prec'] = df['Precipitation']/df['Time']
df['dt_fill'] = df['Filling']/df['Time']

df = df.drop(['Filling', 'Time', 'Precipitation'], axis=1)

# Just rearranging the dataframe so that filling over time is the first element

df = df.iloc[:, ::-1]

sns.set_theme(style="white")

# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});

print(df.dt_prec.describe())

df.dt_prec.hist(bins=50)

# I am getting rid of the extreme values

df = df[df['dt_prec']>1.25]
df = df[df['dt_prec']<3.5]

df.dt_prec.hist(bins=50)
len(df)

from statsmodels.regression.linear_model import OLS
# Here I standardise the data

s_df = (df - df.mean())/df.std()

# I was getting rid of Routine before, since it is not a numerical variable (keep in mind that this might be useful to tell the utility, which routine is the best). Furthermore, I will get rid of construction year, since it is highly correlated to the width.

s_df = s_df.drop(['Construction_year'], axis=1)

# Furthermore, I am getting rid of all the NANs in the data set

s_df = s_df.dropna()

s_df.corr()

s_df = s_df.drop(['Contributing area'], axis=1)

# Let's try to find the most influencing variables and their influence with a Linear Regression Model using ordinary least squares (OLS)

X = s_df.drop(['SID', 'dt_fill'], axis=1)
y = s_df['dt_fill']

from sklearn.linear_model import Ridge
clf = Ridge(alpha=10.0)
result = clf.fit(X, y)

print(pd.Series(result.coef_, index=X.columns).abs().sort_values(ascending=False))

y_fitted = result.predict(X)

plt.plot(y, y_fitted, 'ko')

np.corrcoef(y, y_fitted)[0, 1]

from sklearn.linear_model import Lasso
clf = Lasso(alpha=0.09)
result = clf.fit(X, y)

print(pd.Series(result.coef_, index=X.columns).abs().sort_values(ascending=False))

y_fitted = result.predict(X)

plt.plot(y, y_fitted, 'ko')

np.corrcoef(y, y_fitted)[0, 1]

X = s_df[['East', 'Slope', 'Width [mm]']]
y = s_df['dt_fill']

model = OLS(y, X)
results = model.fit()
results.summary2()

import graphviz
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn import tree

clf = DecisionTreeRegressor(random_state=0, max_depth=3)

d = df.drop(['SID', 'dt_prec', 'Flow accumulation', 'North', 'Construction_year', 'masl GIS', 'Contributing area'], axis=1)
d = d.dropna()
Xf = d.drop('dt_fill', axis=1)
yf = d['dt_fill']
clf.fit(Xf, yf)
# Visualize the tree
from IPython.display import display
display(graphviz.Source(export_graphviz(clf, feature_names=Xf.columns, filled=True, rounded=True, special_characters=False, impurity=False,)))
