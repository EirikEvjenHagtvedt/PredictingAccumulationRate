# import libraries
import requests
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import datetime
import math

# import data
data_field = pd.read_excel(r'Path where the Excel file is stored\File name.xlsx') #field measurements
data_GIS = pd.read_excel(r'Path where the Excel file is stored\File name.xlsx') #data from GIS
data_Horten = pd.read_excel(r'Path where the Excel file is stored\File name.xlsx') #data from Gemini
data_Horten.drop(data_Horten.loc[data_Horten['Filling percentage'] == 1.0].index, inplace=True)
data_field.drop(data_field.loc[data_field['Total depth']/data_field['Depth filling'] == 1.0].index, inplace=True)

# filling percentage for field measurements
data_field['Filling percentage'] = data_field['Depth filling']/data_field['Total depth'] #filling in percent (to exclude all with 100%)
                       
# joining files
data = [data_field, data_Horten]
data = pd.concat(data)

# merging files with ArcMap data
data = pd.merge(data, data_GIS, on="SID", how="inner")

# dropping rows not containing filling information
idx = data['Maintenance'].dropna().index
df = data.loc[idx]
df = df.reset_index(drop=True)

# calculating inital parameters
df['dt_time'] = df['Timestamp']-df['Maintenance'] #time interval
df['dt_time'] = df['dt_time'].dt.days #timedelta to days

df['dt_fill'] = df['Depth filling']/df['dt_time'] #filling over time [cm/time]
df['dt_filling'] = df['Filling percentage']/df['dt_time']*100 # filling over time [%/time]

date_limit = '2015-06-02'
mask = (df['Maintenance'] < date_limit) & df['Place'].isin(['Asgardstrand','Skoppum']) 
df.loc[mask, 'Place'] = 'Horten'

from datetime import date
client_id = 'INSERT CLIENT ID' #id
start_date = '2015-06-02' #start date climate data
end_date = date.today() # end date climate data
end_date = end_date.strftime("%Y-%m-%d")  
date = "{}/{}".format(start_date,end_date) #settinng format

# climate datachoose station name, type of data and reference time 
# Horten = SN27160, Åsgårdstrand = SN27162, Nykirke = SN27120, Skoppum = SN27161
endpoint = 'https://frost.met.no/observations/v0.jsonld'
parameters = {
    'sources': 'SN27160, SN27162, SN27120, SN27161',
    'elements': 'sum(precipitation_amount P1D)',
    'referencetime': date,
}
# issue a HTTP GET request
r = requests.get(endpoint, parameters, auth=(client_id,''))
# extract JSON data
json = r.json()

# check if the request worked, print errors if any
if r.status_code == 200:
    data = json['data']
    print('Data retrieved from frost.met.no!')
else:
    print('Error! Returned status code %s' % r.status_code)
    print('Message: %s' % json['error']['message'])
    print('Reason: %s' % json['error']['reason'])
    
# dataframe with all of the observations in a table format
precip = pd.DataFrame()
for i in range(len(data)):
    row = pd.DataFrame(data[i]['observations'])
    row['referenceTime'] = data[i]['referenceTime']
    row['sourceId'] = data[i]['sourceId']
    precip = precip.append(row)

precip = precip.reset_index()

# these additional columns will be kept
columns = ['sourceId','referenceTime','elementId','value','unit','timeOffset']
precip_data = precip[columns].copy()

# convert the time value to something Python understands
precip_data['referenceTime'] = pd.to_datetime(precip_data['referenceTime'])

# set to data and replace station number with names
precip_data['referenceTime'] = precip_data['referenceTime'].dt.date

#just changing the apperance of the station names
precip_data['sourceId'] = precip_data['sourceId'].replace(['SN27160:0','SN27162:0','SN27120:0','SN27161:0'],['Horten','Asgardstrand','Nykirke','Skoppum'])

# merging precipitation and dataframe 
df['key'] = df['Place'].str.lower()
precip_data['key'] = precip_data['sourceId'].str.lower()
m = df.merge(precip_data, on='key', how='left')
m['value'] = m['value'].mask(~m['referenceTime'].between(m['Maintenance'], m['Timestamp']))
out = m.groupby(['Maintenance', 'Timestamp', 'SID'], as_index=False, sort=False)['value'].mean()

# adding precipitation data
df = pd.merge(df,out)

# renaming column
df.rename(columns = {'value' : 'Precipitation'}, inplace = True)

# delete precipitation data from dates older then treshold
date_min = '2015-06-02'
mask_min = (df['Maintenance'] < date_min)
df.loc[mask_min, 'Precipitation'] = np.nan

# separating files
# files will be evaluated separately
field = df.dropna( how='any', subset=['dt_fill']) # field data
horten = df[df['dt_fill'].isna()] # data from Horten

# binary 
df_bin = field[['dt_fill','Drain type','Curb','Ditch','Trees','Bushes','Sediment','Stopper','Pavement']] 
# field measurements
df_field = field[['dt_fill','Precipitation','SID','Longitude','Latitude','Width','Slope','Total depth','MASL','Flow accumulation']] 
# Hortens data
df_horten = horten[['dt_filling','Precipitation','SID','Longitude','Latitude','Width','Slope','MASL','Flow accumulation']] 
# data combined
df2 = df[['dt_filling', 'Precipitation','Slope','Longitude','Latitude','MASL','Place','Width','SID' ,'Flow accumulation']] 

# Evaluating first data set
df2 = df2.dropna()

# function for removing outliers
def outlier_treatment(datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn , [25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range,upper_range

# calcualting outliers
l,u = outlier_treatment(df2.dt_filling)
#inpecting outliers
remove = df2[ (df2.dt_filling > u) | (df2.dt_filling < l) ]
# remove outliers
df2 = df2[ ~df2['dt_filling'].isin(remove['dt_filling']) ]

# I am getting rid of the extreme values

df2 = df2[df2['Precipitation']>1.5]

df2.Precipitation.hist(bins=50)
len(df2)

sns.set_theme(style="white")

s_df = (df2 - df2.mean())/df2.std()

# Compute the correlation matrix
corr = df2.corr('pearson')

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 15, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
            square=True, linewidths=1.5, cbar_kws={"shrink": 0.6}, annot = True, fmt = '.1g');
plt.savefig('corrHeatmap.png')

#Correlation with output variable
cor_target = abs(corr['dt_filling'])

#Selecting highly correlated features setting a correlation treshold as > 0.1
relevant_features = cor_target[cor_target>0.1]
not_relevant_features = cor_target[cor_target<0.1]
print("Most correlating features in respect to dt_filling are: \n{}".format(relevant_features))
print("\n Least correlating features in respect to dt_filling are: \n{}".format(not_relevant_features))

#Deleting features that are not relevant
drop_list = list(relevant_features.index.values.tolist())
#Dropping features that are not relevant
df2 = df2.drop(df2.columns.difference(drop_list), axis=1)

import statsmodels.formula.api as smf
# defining function to retrieve vif
def get_vif(exogs, data):
    # initialize dictionaries
    vif_dict, tolerance_dict = {}, {}

    # create formula for each exogenous variable
    for exog in exogs:
        not_exog = [i for i in exogs if i != exog]
        formula = f"{exog} ~ {' + '.join(not_exog)}"

        # extract r-squared from the fit
        r_squared = smf.ols(formula, data=data).fit().rsquared

        # calculate VIF
        vif = 1/(1 - r_squared)
        vif_dict[exog] = vif

        # calculate tolerance
        tolerance = 1 - r_squared
        tolerance_dict[exog] = tolerance

    # return VIF DataFrame
    df_vif = pd.DataFrame({'VIF': vif_dict, 'Tolerance': tolerance_dict})

    return df_vif
    
s_df = (df2 - df2.mean())/df2.std()
X = s_df.drop('dt_filling', axis=1)
y = s_df['dt_filling']

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state = 1)

# OLS
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = model.predict(X_test)

print('Kept for testing', X_test.shape)
print('Kept for training', X_train.shape)
print('Score:', model.score(X_test, y_test))
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, predictions))
print('Mean absolute error: %.2f'
    % mean_absolute_error(y_test, predictions))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, predictions))
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
# The coefficients
print('Coefficients: \n', coeff_df)

## The line / model
plt.scatter(y_test, predictions, c='green')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('Ordinary least squares')
plt.savefig('OLSPlot.png')

# plotting residuals
plt.hist(y_test - predictions, color='green')
plt.xlabel('y test - predictions')
plt.ylabel('Frequency')
plt.title('Ordinary least squares')
plt.savefig('OLSPlot2.png')

from sklearn.linear_model import Ridge
alphas = 10**np.linspace(10,-2,100)*0.25
ridge = Ridge(normalize = False)
coefs = []

for a in alphas:
    ridge.set_params(alpha = a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
    
np.shape(coefs)

from sklearn.linear_model import Ridge, RidgeCV
alphas = 10**np.linspace(10,-2,100)*0.5
ridge = Ridge(normalize = False)
coefs = []

for a in alphas:
    ridge.set_params(alpha = a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
    
np.shape(coefs)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

# Ridge

ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = False)
ridgecv.fit(X_train, y_train)
ridgecv.alpha_

ridge_all = Ridge(alpha = ridgecv.alpha_, normalize = False)
ridge_all.fit(X_train, y_train)
predictions_ridge = ridge_all.predict(X_test)
print('Kept for testing', X_test.shape)
print('Kept for training', X_train.shape)
print('Score:', ridge_all.score(X_test, y_test))
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, predictions_ridge))
print('Mean absolute error: %.2f'
    % mean_absolute_error(y_test, predictions))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, predictions_ridge))
coeff_df = pd.DataFrame(ridge_all.coef_, X.columns, columns=['Coefficient'])
# The coefficients
print('Coefficients: \n', coeff_df)

## The line / model
plt.scatter(y_test, predictions_ridge, c='blue')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('Ridge')
plt.savefig('RidgePlot.png')

# plotting residuals
plt.hist(y_test - predictions_ridge, color='blue')
plt.xlabel('y test - predictions')
plt.ylabel('Frequency')
plt.title('Ridge')
plt.savefig('RidgePlot2.png')

# Lasso
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import scale
lasso = Lasso(max_iter = 10000, normalize = False)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X_train), y_train)
    coefs.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import scale
lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = False)
lassocv.fit(X_train, y_train)

lasso.set_params(alpha=lassocv.alpha_)
model_lasso = lasso.fit(X_train, y_train)
predictions_lasso = model_lasso.predict(X_test)

print('Kept for testing', X_test.shape)
print('Kept for training', X_train.shape)
print('Score:', model_lasso.score(X_test, y_test))
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, predictions_lasso))
print('Mean absolute error: %.2f'
    % mean_absolute_error(y_test, predictions))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, predictions_lasso))
coeff_df = pd.DataFrame(model_lasso.coef_, X.columns, columns=['Coefficient'])
# The coefficients
print('Coefficients: \n', coeff_df)

## The line / model
plt.scatter(y_test, predictions_lasso, c='red')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('Lasso')
plt.savefig('LassoPlot.png')

# plotting residuals
plt.hist(y_test - predictions_lasso, color='red')
plt.xlabel('y test - predictions')
plt.ylabel('Frequency')
plt.title('Lasso')
plt.savefig('LassoPlot2.png')

# Decision tree
import graphviz
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn import tree
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
d = df2[['dt_filling','Precipitation', 'Slope']]
d = d.dropna()
Xf = d.drop('dt_filling', axis=1)
yf = d['dt_filling']

X_train, X_test, y_train, y_test = train_test_split(Xf, yf, test_size=0.10,random_state=0,shuffle=True)

from sklearn.metrics import mean_squared_error as mse
max_depths = range(1, 20)
training_error = []
for max_depth in max_depths:
    model_1 = DecisionTreeRegressor(max_depth=max_depth)
    model_1.fit(Xf, yf)
    training_error.append(mse(yf, model_1.predict(Xf)))
    
testing_error = []
for max_depth in max_depths:
    model_2 = DecisionTreeRegressor(max_depth=max_depth)
    model_2.fit(X_train, y_train)
    testing_error.append(mse(y_test, model_2.predict(X_test)))
    
plt.plot(max_depths, training_error, color='blue', label='Training error')
plt.plot(max_depths, testing_error, color='green', label='Testing error')
plt.xlabel('Tree depth')
plt.ylabel('Mean squared error')
plt.title('Hyperparameter Tuning', pad=15, size=15)
plt.legend()
plt.savefig('error.png')

from sklearn.model_selection import GridSearchCV

model = DecisionTreeRegressor()

gs = GridSearchCV(model,
                  param_grid = {'max_depth': range(1, 20),
                                'min_samples_split': range(10, 60, 5)},
                  cv=10,
                  n_jobs=3,
                  scoring='neg_mean_squared_error')

gs.fit(X_train, y_train)

print(gs.best_params_)
print(-gs.best_score_)

from sklearn.tree import DecisionTreeRegressor
new_model = DecisionTreeRegressor(max_depth=3, min_samples_split=50)
#or new_model = gs.best_estimator_
new_model.fit(X_train, y_train)
y_pred = new_model.predict(X_test)

from sklearn.tree import export_graphviz
import graphviz


dot_data = tree.export_graphviz(new_model, feature_names=list(Xf), class_names=sorted(yf.unique()), filled=True)
graphviz.Source(dot_data)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R Squared Score is:', r2_score(y_test, y_pred))

# Evaluating Horten's data
df_horten = df_horten.dropna()
df_horten.describe().T

# calling outlier function
# calcualting outliers
l,u = outlier_treatment(df_horten.dt_filling)
# inpecting outliers
remove = df_horten[ (df_horten.dt_filling > u) | (df_horten.dt_filling < l) ]
# remove outliers
df_horten = df_horten[ ~df_horten['dt_filling'].isin(remove['dt_filling']) ]

s_horten = (df_horten-df_horten.mean())/df_horten.std()
corr_horten = df_horten.corr('pearson')
mask = np.triu(np.ones_like(corr_horten, dtype=bool))
sns.heatmap(corr_horten, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot =True, fmt ='.1g');
plt.savefig('HortenDataCorr.png')

print(df_horten.Precipitation.describe())

df_horten.Precipitation.hist(bins=50)

# I am getting rid of the extreme values

df_horten = df_horten[df_horten['Precipitation']>1.5]

#Correlation with output variable
target_horten = abs(corr_horten['dt_filling'])

#Selecting highly correlated features setting a correlation treshold as > 0.1
relevant_features = target_horten[target_horten>0.1]
not_relevant_features = target_horten[target_horten<0.1]
print("Most correlating features in respect to dt_filling are: \n{}".format(relevant_features))
print("\n Least correlating features in respect to dt_filling are: \n{}".format(not_relevant_features))

#Deleting features that are not relevant
drop_list = list(relevant_features.index.values.tolist())
#Dropping features that are not relevant
df_horten = df_horten.drop(df_horten.columns.difference(drop_list), axis=1)
df_horten

# calling vif function
exogs = df_horten
get_vif(exogs=exogs, data = df_horten)

# OLS
s_df_horten = (df_horten - df_horten.mean())/df_horten.std()
X_horten = s_df_horten.drop('dt_filling', axis=1)
y_horten = s_df_horten['dt_filling']

# create training and testing vars
X_train_horten, X_test_horten, y_train_horten, y_test_horten = train_test_split(X_horten, y_horten, test_size=0.10, random_state = 1)

# fit a model
lm_horten = linear_model.LinearRegression()
model_horten = lm_horten.fit(X_train_horten, y_train_horten)
predictions_horten = model_horten.predict(X_test_horten)

print('Kept for testing', X_test_horten.shape)
print('Kept for training', X_train_horten.shape)
print('Score:', model_horten.score(X_test_horten, y_test_horten))
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test_horten, predictions_horten))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test_horten, predictions_horten))
coeff_df_horten = pd.DataFrame(model_horten.coef_, X_horten.columns, columns=['Coefficient'])
# The coefficients
print('Coefficients: \n', coeff_df_horten)

## The line / model
plt.scatter(y_test_horten, predictions_horten)
plt.xlabel('True Values')
plt.ylabel('Predictions')

plt.hist(y_test_horten - predictions_horten)

coefs = []

for a in alphas:
    ridge.set_params(alpha = a)
    ridge.fit(X_horten, y_horten)
    coefs.append(ridge.coef_)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

ridgecv_horten = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = False)
ridgecv_horten.fit(X_train_horten, y_train_horten)
ridgecv_horten.alpha_

ridge_horten = Ridge(alpha = ridgecv_horten.alpha_, normalize = False)
ridge_horten.fit(X_train_horten, y_train_horten)
predictions_ridge_horten = ridge_horten.predict(X_test_horten)
print('Kept for testing', X_test_horten.shape)
print('Kept for training', X_train_horten.shape)
print('Score:', ridge_horten.score(X_test_horten, y_test_horten))
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test_horten, predictions_ridge_horten))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test_horten, predictions_ridge_horten))
coeff_df_horten = pd.DataFrame(ridge_horten.coef_, X_horten.columns, columns=['Coefficient'])
# The coefficients
print('Coefficients: \n', coeff_df_horten)

## The line / model
plt.scatter(y_test_horten, predictions_ridge_horten)
plt.xlabel('True Values')
plt.ylabel('Predictions')

plt.hist(y_test_horten - predictions_ridge_horten)

lasso = Lasso(max_iter = 10000, normalize = False)
coefs_horten = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X_train_horten), y_train_horten)
    coefs_horten.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(alphas*2, coefs_horten)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

lassocv_horten = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = False)
lassocv_horten.fit(X_train_horten, y_train_horten)

lasso.set_params(alpha=lassocv_horten.alpha_)
model_lasso_horten = lasso.fit(X_train_horten, y_train_horten)
predictions_lasso_horten = model_lasso_horten.predict(X_test_horten)

print('Kept for testing', X_test_horten.shape)
print('Kept for training', X_train_horten.shape)
print('Score:', model_lasso_horten.score(X_test_horten, y_test_horten))
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test_horten, predictions_lasso_horten))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test_horten, predictions_lasso_horten))
coeff_df_horten = pd.DataFrame(model_lasso_horten.coef_, X_horten.columns, columns=['Coefficient'])
# The coefficients
print('Coefficients: \n', coeff_df_horten)

## The line / model
plt.scatter(y_test_horten, predictions_lasso_horten)
plt.xlabel('True Values')
plt.ylabel('Predictions')

lassocv_horten.alpha_
plt.hist(y_test_horten - predictions_lasso_horten)

d_horten = df_horten[['dt_filling','Longitude','MASL','Slope','Latitude']]
d_horten = d_horten.dropna()
Xf_horten = d_horten.drop('dt_filling', axis=1)
yf_horten = d_horten['dt_filling']

X_train_horten, X_test_horten, y_train_horten, y_test_horten = train_test_split(Xf_horten, yf_horten, test_size=0.10,random_state=0, shuffle = True)

max_depths = range(1, 20)
training_error_horten = []
for max_depth in max_depths:
    model_1_horten = DecisionTreeRegressor(max_depth=max_depth)
    model_1_horten.fit(Xf_horten, yf_horten)
    training_error_horten.append(mse(yf_horten, model_1_horten.predict(Xf_horten)))
    
testing_error_horten = []
for max_depth in max_depths:
    model_2_horten = DecisionTreeRegressor(max_depth=max_depth)
    model_2_horten.fit(X_train_horten, y_train_horten)
    testing_error_horten.append(mse(y_test_horten, model_2_horten.predict(X_test_horten)))
    
plt.plot(max_depths, training_error_horten, color='blue', label='Training error')
plt.plot(max_depths, testing_error_horten, color='green', label='Testing error')
plt.xlabel('Tree depth')
plt.ylabel('Mean squared error')
plt.title('Hyperparameter Tuning', pad=15, size=15)
plt.legend()
plt.savefig('error.png')

model_horten_DT = DecisionTreeRegressor()

gs_horten = GridSearchCV(model_horten_DT, param_grid = {'max_depth': range(1, 20),'min_samples_split': range(10, 60, 10)},cv=10, 
                         n_jobs=3,scoring='neg_mean_squared_error')

gs_horten.fit(X_train_horten, y_train_horten)

print(gs_horten.best_params_)
print(-gs_horten.best_score_)

new_model_horten = DecisionTreeRegressor(max_depth=5, min_samples_split=10)
#or new_model = gs.best_estimator_
new_model_horten.fit(X_train_horten, y_train_horten)
y_pred_horten = new_model_horten.predict(X_test_horten)

dot_data_horten = tree.export_graphviz(new_model_horten, feature_names=list(Xf_horten), class_names=sorted(yf_horten.unique()), filled=True)
graphviz.Source(dot_data_horten)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_horten, y_pred_horten))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_horten, y_pred_horten))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_horten, y_pred_horten)))
print('R Squared Score is:', r2_score(y_test_horten, y_pred_horten))

df_field = df_field.dropna()
df_field.describe().T

# calling outlier function
# calcualting outliers
l,u = outlier_treatment(df_field.dt_fill)
#inpecting outliers
remove = df_field[ (df_field.dt_fill > u) | (df_field.dt_fill < l) ]
# remove outliers
df_field = df_field[ ~df_field['dt_fill'].isin(remove['dt_fill']) ]

s_field = (df_field-df_field.mean())/df_field.std()
corr_field = df_field.corr('pearson')
mask = np.triu(np.ones_like(corr_field, dtype=bool))
sns.heatmap(corr_field, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot= True, fmt='.1g');
plt.savefig('FieldMeasuremntCorrelation.png')

#Correlation with output variable
target_field = abs(corr_field['dt_fill'])

#Selecting highly correlated features setting a correlation treshold as > 0.1
relevant_features = target_field[target_field>0.1]
not_relevant_features = target_field[target_field<0.1]
print("Most correlating features in respect to dt_filling are: \n{}".format(relevant_features))
print("\n Least correlating features in respect to dt_filling are: \n{}".format(not_relevant_features))

#Deleting features that are not relevant
drop_list = list(relevant_features.index.values.tolist())
#Dropping features that are not relevant
df_field = df_field.drop(df_field.columns.difference(drop_list), axis=1)

s_df_field = (df_field - df_field.mean())/df_field.std()
X_field = s_df_field.drop('dt_fill', axis=1)
y_field = s_df_field['dt_fill']

# create training and testing vars
X_train_field, X_test_field, y_train_field, y_test_field = train_test_split(X_field, y_field, test_size=0.05, random_state = 1)

# fit a model
lm_field = linear_model.LinearRegression()
model_field = lm_field.fit(X_train_field, y_train_field)
predictions_field = model_field.predict(X_test_field)

print('Kept for testing', X_test_field.shape)
print('Kept for training', X_train_field.shape)
print('Score:', model_field.score(X_test_field, y_test_field))
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test_field, predictions_field))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test_field, predictions_field))
coeff_df_field = pd.DataFrame(model_field.coef_, X_field.columns, columns=['Coefficient'])
# The coefficients
print('Coefficients: \n', coeff_df_field)

## The line / model
plt.scatter(y_test_field, predictions_field)
plt.xlabel('True Values')
plt.ylabel('Predictions')

plt.hist(y_test_field - predictions_field)

coefs_field = []

for a in alphas:
    ridge.set_params(alpha = a)
    ridge.fit(X_field, y_field)
    coefs_field.append(ridge.coef_)

ax = plt.gca()
ax.plot(alphas, coefs_field)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

ridgecv_field = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = False)
ridgecv_field.fit(X_train_field, y_train_field)
ridgecv_field.alpha_

ridge_field = Ridge(alpha = ridgecv_field.alpha_, normalize = False)
ridge_field.fit(X_train_field, y_train_field)
predictions_ridge_field = ridge_field.predict(X_test_field)
print('Kept for testing', X_test_field.shape)
print('Kept for training', X_train_field.shape)
print('Score:', ridge_field.score(X_test_field, y_test_field))
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test_field, predictions_ridge_field))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test_field, predictions_ridge_field))
coeff_df_field = pd.DataFrame(ridge_field.coef_, X_field.columns, columns=['Coefficient'])
# The coefficients
print('Coefficients: \n', coeff_df_field)

## The line / model
plt.scatter(y_test_field, predictions_ridge_field)
plt.xlabel('True Values')
plt.ylabel('Predictions')

plt.hist(y_test_field - predictions_ridge_field)

lasso = Lasso(max_iter = 10000, normalize = False)
coefs_lasso_field = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X_train_field), y_train_field)
    coefs_lasso_field.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(alphas*2, coefs_lasso_field)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

lassocv_field = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = False)
lassocv_field.fit(X_train_field, y_train_field)

lasso.set_params(alpha=lassocv_field.alpha_)
model_lasso_field = lasso.fit(X_train_field, y_train_field)
predictions_lasso_field = model_lasso_field.predict(X_test_field)

print('Kept for testing', X_test_field.shape)
print('Kept for training', X_train_field.shape)
print('Score:', model_lasso_field.score(X_test_field, y_test_field))
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test_field, predictions_lasso_field))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test_field, predictions_lasso_field))
coeff_df_field = pd.DataFrame(model_lasso_field.coef_, X_field.columns, columns=['Coefficient'])
# The coefficients
print('Coefficients: \n', coeff_df_field)

## The line / model
plt.scatter(y_test_field, predictions_lasso_field)
plt.xlabel('True Values')
plt.ylabel('Predictions')

lassocv_field.alpha_
plt.hist(y_test_field - predictions_lasso_field)

d_field = df_field[['dt_fill', 'Slope',]]
d_field = d_field.dropna()
Xf_field = d_field.drop('dt_fill', axis=1)
yf_field = d_field['dt_fill']

X_train_field, X_test_field, y_train_field, y_test_field = train_test_split(Xf_field, yf_field, test_size=0.05,random_state=0, shuffle = True)

max_depths = range(1, 20)
training_error_field = []
for max_depth in max_depths:
    model_1_field = DecisionTreeRegressor(max_depth=max_depth)
    model_1_field.fit(Xf_field, yf_field)
    training_error_field.append(mse(yf_field, model_1_field.predict(Xf_field)))
    
testing_error_field = []
for max_depth in max_depths:
    model_2_field = DecisionTreeRegressor(max_depth=max_depth)
    model_2_field.fit(X_train_field, y_train_field)
    testing_error_field.append(mse(y_test_field, model_2_field.predict(X_test_field)))
    
plt.plot(max_depths, training_error_field, color='blue', label='Training error')
plt.plot(max_depths, testing_error_field, color='green', label='Testing error')
plt.xlabel('Tree depth')
plt.ylabel('Mean squared error')
plt.title('Hyperparameter Tuning', pad=15, size=15)
plt.legend()
plt.savefig('error.png')

model_field_DT = DecisionTreeRegressor()

gs_field = GridSearchCV(model_field_DT, param_grid = {'max_depth': range(1, 20),'min_samples_split': range(10, 60, 10)},cv=10, 
                         n_jobs=3,scoring='neg_mean_squared_error')

gs_field.fit(X_train_field, y_train_field)

print(gs_field.best_params_)
print(-gs_field.best_score_)

new_model_field = DecisionTreeRegressor(max_depth=2, min_samples_split=50)
#or new_model = gs.best_estimator_
new_model_field.fit(X_train_field, y_train_field)
y_pred_field = new_model_field.predict(X_test_field)

dot_data_field = tree.export_graphviz(new_model_field, feature_names=list(Xf_field), class_names=sorted(yf_field.unique()), filled=True)
graphviz.Source(dot_data_field)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_field, y_pred_field))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_field, y_pred_field))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_field, y_pred_field)))
print('R Squared Score is:', r2_score(y_test_field, y_pred_field))

df_bin = field[['dt_fill','Drain type','Curb','Ditch','Trees','Bushes','Sediment','Stopper','Pavement']] 
df_bin = df_bin.dropna()
df_bin

# calcualting outliers
l,u = outlier_treatment(df_bin.dt_fill)
#inpecting outliers
remove = df_bin[ (df_bin.dt_fill > u) | (df_bin.dt_fill < l) ]
# remove outliers
df_bin = df_bin[ ~df_bin['dt_fill'].isin(remove['dt_fill']) ]
df_bin

#To change variables from string to int. 
Mapping = {'Yes': 1, 'No': 0}
Mapping2 = {'Grated': 1, 'Domed': 0}
Mapping3 = {'Top': 1, 'Combi': 0}

df_bin['Bushes'] = df_bin['Bushes'].map(Mapping)
df_bin['Trees'] = df_bin['Trees'].map(Mapping)
df_bin['Curb'] = df_bin['Curb'].map(Mapping)
df_bin['Stopper'] = df_bin['Stopper'].map(Mapping)
df_bin['Ditch'] = df_bin['Ditch'].map(Mapping)
df_bin['Drain type'] = df_bin['Drain type'].map(Mapping2)

#Categorical data transfer
hot_Pavement = pd.get_dummies(df['Pavement'])
#hot_Commercial = pd.get_dummies(df['Commercial area'])
hot_Sediment = pd.get_dummies(df['Sediment'])

#Adding the dummies to a new dataframe to have a look on importance. If not important they will be removed
df_bin = df_bin.join(hot_Pavement)
df_bin = df_bin.join(hot_Sediment)

#dropping original columns
df_bin = df_bin.drop(['Pavement', 'Sediment'], axis = 1)
df_bin.sum(axis=0)

from scipy import stats
corr_list = []
y_bin = df_bin['dt_fill'].astype(float)


for column in df_bin:
    x_bin=df_bin[column]
    corr = stats.pointbiserialr(list(x_bin), list(y_bin))
    corr_list.append(corr[0])
    print(column)
    print(corr)
