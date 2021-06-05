# PredictingAccumulationRate


NB! The following code is owned by Eirik Evjen Hagtvedt and may not be shared without permission. 

The model take part in the master thesis submitted by Eirik Evjen Hagtvedt: "Identifying an optimal maintenance schedule for sand traps - A Norwegian case study in Horten". 

There are two separate models provided
1. FinalDevelopedModel.py - is used to predict the accumulation rate in Horten's sandtraps. It includes the regression models; OLS, Ridge, Lasso, decision tree regressor and evaluation of categorical parameters. The code is developed by Eirik Evjen Hagtvedt.
2. ModelForStratifying.py - is the initial model used for stratifying sampling before field measurements in Horten municipality spring 2021. The code is developed by David Steffelbauer. Thank you David for showing me the fun and usefullness of python! :) 

The model is python-based and aims to predict the accumulation rate in sand traps located in the municipality of Horten, Norway. 

There is  a wish to provide the code as it may be useful for other municipalities, and especially the municipality of Horten. The model may treat data from a Gemini portal format with preliminary steps. Gemini portal is used by almost every municipality in Norway. 

Feel free to contact Eirik Evjen Hagtvedt; e-mail: eirikehag@gmail.com if there are questions. 

The model is last updated: [05/06/2021]

# FinalDevelopedModel.py description
The model input the data sets;  Horten's data, field measurement data and GIS data from ArcMap.

Horten's data contain the columns; SID, width, accumulation rate measured, and maintenance dates. 
Field measurement data contain all the columns above plus one column for each measured parameter during field measurements. GIS data contain: slope, flow accumulation, contributing area, construction year and masl. 

Running the model require a CLIENT-ID from frost.met to extract the json-file containing precipitation data.

### Syntax
A generalized syntax of the model follow:
1. Import data sets.
2. Merge data sets based on SID number. 
3. Choose to evaluate with or without 100\% measurements included.
4. Calculate initial parameter.
5. Retrieve all precipitation data from all stations up until today (updates automatically).
6. Calculate and append mean precipitation for all sand traps based on time interval. In this thesis the first maintenance event to the last maintenance event.
7. If precipitation values are not valid. Append to different stations. 
8. Split data set into four different data sets evaluated separately. 
9. Treat outliers with IQR. 
10. Delete uncorrelated variables.
11. Check VIF value.  
12. Train and test OLS model. 
13. Train and test Ridge model.
14. Train and test Lasso model.
15. Train and test Regression tree.
16. Convert categorical variables to dummy variables. 
17. Append p-value and PBC for categorical variables. 

### Known syntax errors
1. If flow accumulation is input the VIF will not calculate. Reason is not known. Has not been a problem during the thesis as flow accumulation is removed in the preliminary step due to low correlation. 
2. Warning when mapping dummy variables due to calling the function .loc. May be avoided without problems. 

### Citations
Parts of the code is inspired from the following links:
1. https://medium.com/@prashant.nair2050/hands-on-outlier-detection-and-treatment-in-python-using-1-5-iqr-rule-f9ff1961a414
2. https://stackoverflow.com/questions/67545499/clean-code-that-runs-getting-the-mean-of-values-in-one-dataframe-based-on-date
3. https://www.reddit.com/r/learnpython/comments/n6h12o/how_to_sum_up_rows_given_start_and_end_condition/gx7pq1n/?context=3
4. https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python
5. https://towardsdatascience.com/train-a-regression-model-using-a-decision-tree-70012c22bcc1
6. https://frost.met.no/python_example.html
7. http://www.science.smith.edu/~jcrouser/SDS293/labs/lab10-py.html
8. https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
9. https://stackoverflow.com/questions/67106853/how-to-do-point-biserial-correlation-for-multiple-columns-in-one-iteration 

NB! I had never used python before starting the thesis. Resulting in sources such as stack overflow actively being used when building the model. The broad spectre of different literature may suggest that snippets of code are not cited.  
