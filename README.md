# Machine Learning for Hotel Inspection Outcome Prediction

This project delves into the Chicago Food Inspection dataset to explore the relationship between the attributes of food establishments and the outcomes of their inspections.
The hypothesis is that the type of facility, its location, and the nature of the inspection significantly influence the inspection outcomes.

## Introduction of Dataset
This information is derived from inspections of restaurants and other food establishments in Chicago from January 1, 2010 to the present. Inspections are performed by staff from the Chicago Department of Public Health’s Food Protection Program. Inspections are done using a standardized procedure.

## Data Preparation and Cleaning (Libraries used in the project)

**NumPy and Pandas:** Utilized for core data manipulation and numerical computing in Python.
**Matplotlib and Plotly:** used for data exploration and to enhance the interactivity of visualizations.
**Fuzzywuzzy:** consolidated text data by matching and correcting similar strings.
**Folium: ** Used for geospatial data visualization and interactive mapping
**Regular Expressions (re):** searching and parsing complex string patterns
**Scikit-learn:** used for building ML models
**Facebook's "Prophet":** used for forecasting future hotel inspection results.

## Descriptive Analytics Findings :

●	Synthesized inspection data into a clustered bar chart to elucidate trends in pass/fail rates across varying risk levels of food establishments.
●	Illustrated inspection result disparities across restaurants, grocery stores, and schools, revealing the highest compliance in schools with a 67.5% pass rate.
●	Analyzed inspection results by type, revealing that routine canvass checks show a 50.7% pass rate and complaints lead to a higher failure rate
●	Mapped the trend of the five most common violation codes over several years, identifying code 34 as the most recurrent issue in food establishments.

<img width="541" alt="image" src="https://github.com/PHANINDRA25/Hotel-inspection-result-prediction/assets/136892334/c2cbb3c4-f8b7-4448-954e-1dbd4d08207c">

<img width="448" alt="image" src="https://github.com/PHANINDRA25/Hotel-inspection-result-prediction/assets/136892334/07e235e3-184c-4cbf-9989-d16905fba0a3">


## Predictive Modelling :
●	Achieved a 78% accuracy in predicting hotel inspection outcomes by employing Decision Tree, Random Forest and Logistic rEGRESSION.
●	performed hyper parametering tuning using Gridsearch CV and implemented cross validation to assess the performance of ML models 
●	Optimized training dataset by applying One Hot Encoding to categorical features and addressing imbalanced classes (80:20 ratio) using SMOTE.

![image](https://github.com/PHANINDRA25/Hotel-inspection-result-prediction/assets/136892334/ea8d92dd-9e00-44d8-9cb1-ba35cdfc7c7d)
