# PredictingAccumulationRate
The following model take part in the master thesis submitted by Eirik Evjen Hagtvedt: "Identifying an optimal maintenance schedule for sand traps - A Norwegian case study in Horten".

The model is python-based and aims to predict the accumulation rate in sand traps located in the municipality of Horten, Norway. 

Last updated: [05/06/2021]

The fully developed model is presented below. The model input the data sets;  Horten's data, field measurement data and GIS data from ArcMap.

Horten's data contain the columns; SID, width, accumulation rate measured, and maintenance dates. 
Field measurement data contain all the columns above plus one column for each measured parameter during field measurements. GIS data contain: slope, flow accumulation, contributing area, construction year and masl. 

These may be adjusted as a generalized syntax of the model follow:
\begin{enumerate}
    \item Import data sets.
    \item Merge data sets based on SID number. 
    \item Choose to evaluate with or without 100\% measurements included.
    \item Calculate initial parameter.
    \item Retrieve all precipitation data from all stations up until today (updates automatically).
    \item Calculate and append mean precipitation for all sand traps based on time interval. In this thesis the first maintenance event to the last maintenance event.
    \item If precipitation values are not valid. Append to different stations. 
    \item Split data set into four different data sets evaluated separately. 
    \item Treat outliers with IQR. 
    \item Delete uncorrelated variables.
    \item Check VIF value.  
    \item Train and test OLS model. 
    \item Train and test Ridge model.
    \item Train and test Lasso model.
    \item Train and test Regression tree.
    \item Convert categorical variables to dummy variables. 
    \item Append p-value and PBC for categorical variables. 
\end{enumerate}

Parts of the code is inspired from the following links:
\newline
[0] \url{https://medium.com/@prashant.nair2050/hands-on-outlier-detection-and-treatment-in-python-using-1-5-iqr-rule-f9ff1961a414}
\newline
[1] \url{https://stackoverflow.com/questions/67545499/clean-code-that-runs-getting-the-mean-of-values-in-one-dataframe-based-on-date}
\newline
[2] \url{https://www.reddit.com/r/learnpython/comments/n6h12o/how_to_sum_up_rows_given_start_and_end_condition/gx7pq1n/?context=3}
\newline
[2] \url{https://stackoverflow.com/questions/67106853/how-to-do-point-biserial-correlation-for-multiple-columns-in-one-iteration}
\newline
[3] \url{https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python}
\newline
[4] \url{https://towardsdatascience.com/train-a-regression-model-using-a-decision-tree-70012c22bcc1}
[5] \url{https://frost.met.no/python_example.html} \newline
[6] \url{http://www.science.smith.edu/~jcrouser/SDS293/labs/lab10-py.html} \newline
[7] \url{https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b}

NB!I had never used python before starting the thesis. Hence, stack overflow is used a lot as inspiration when building the model. Hence, more snippets of code may be similar to sources on. It is with best efforts tried to cite the code from the main contributors. 

There is  a wish to provide the code as it may be useful for other municipalities, and especially the municipality or anyone else doing similar work. The model treats data from a Gemini portal format. Gemini portal is used by almost every municipality in Norway. 

If there is something wrong with citations, running the code or anything else. Feel free to contact Eirik Evjen Hagtvedt; e-mail: eirikehag@gmail.com. 

NB2! Known issues with the model: \newline
[1] If flow accumulation is input the VIF will not calculate. Reason is not known. Has not been a problem during the thesis as flow accumulation is removed in the preliminary step due to low correlation. \newline
[2] Warning when mapping dummy variables due to calling the function .loc. Reason is not known but is not a problem. 
