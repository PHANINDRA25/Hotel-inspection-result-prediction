#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Load the dataset
file_path = 'inspections.csv'  # Replace with your file path
df = pd.read_csv(file_path)
data = df.copy()


# In[3]:


data.head(5)


# In[4]:


data['results'] = data['results'].str.replace('Pass w/ Conditions', 'Pass').str.strip()


# In[5]:


data['results'].value_counts()


# In[6]:


data.columns


# In[7]:


# Assuming df is your DataFrame
selected_columns = ['facility_cleaned', 'risk', 'inspectiondate','inspectiontype','results','num_violations','cited']

# Create a new DataFrame with selected columns
filtered_df = data[selected_columns]

# Filter rows where 'results' is either 'Pass' or 'Fail'
filtered_df = filtered_df[filtered_df['results'].isin(['Pass', 'Fail'])]




# In[8]:


filtered_df.head(5)


# In[9]:


# one hot encoding
df = pd.get_dummies(filtered_df, columns=['risk'], prefix='risk')

# Display the resulting DataFrame
df.head()


# In[10]:


df.dtypes


# In[11]:


df['results'].value_counts()


# In[12]:


df['results'].value_counts()


# In[13]:


# Binary encoding of the target variable 'results'
df['results'] = df['results'].apply(lambda x: 1 if x == 'Pass' else 0)


# In[14]:


df.dtypes


# In[15]:


# Encoding categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])





# In[16]:


label_encoders


# In[17]:


df.dtypes


# In[18]:


df['results'].value_counts()


# In[19]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


# In[20]:


# Assuming df is your DataFrame and 'target' is the name of your target column
X = df.drop('results', axis=1)
y = df['results'].astype('int')


# In[21]:


# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


# In[ ]:





# In[24]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

# Initializing the Logistic Regression model
log_reg_model = LogisticRegression()

# Cross-validation for Logistic Regression
cv_scores_log_reg = cross_val_score(log_reg_model, X_scaled, y, cv=5)
log_reg_model.fit(X_train_smote, y_train_smote)
y_pred_log_reg = log_reg_model.predict(X_test)
report_log_reg = classification_report(y_test, y_pred_log_reg)

cv_accuracy_log_reg = cv_scores_log_reg.mean()
print("Logistic Regression - Cross-Validation Accuracy:", cv_accuracy_log_reg)
print("Logistic Regression - Classification Report:\n", report_log_reg)


# In[25]:


from sklearn.metrics import confusion_matrix, roc_curve, auc

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_log_reg)

# ROC curve data
fpr, tpr, thresholds = roc_curve(y_test, log_reg_model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)


# In[26]:


# Calculate the percentage for each cell in the confusion matrix
conf_matrix_log_reg_percent = conf_matrix / np.sum(conf_matrix) * 100

# Confusion Matrix plot with percentages
fig = go.Figure(data=go.Heatmap(
                    z=conf_matrix_log_reg_percent,
                    x=['Predicted Negative', 'Predicted Positive'],
                    y=['Actual Negative', 'Actual Positive'],
                    hoverongaps=False,
                    text=np.around(conf_matrix_log_reg_percent, 2),  # round to 2 decimal places
                    colorscale='blues',
                    ))

# Adding annotations for confusion matrix (percentages)
annotations_log_reg_percent = []
for i, row in enumerate(conf_matrix_log_reg_percent):
    for j, value in enumerate(row):
        annotations_log_reg_percent.append(
            dict(
                x=j, y=i,
                xref='x1', yref='y1',
                text=f"{value:.2f}%",  # format to 2 decimal places
                showarrow=False,
                font=dict(
                    family="Arial",
                    size=20,
                    color="black"
                )
            )
        )

fig.update_layout(
    title='Confusion Matrix (Percentages) - Logistic Regression',
    xaxis_title='Predicted Labels',
    yaxis_title='True Labels',
    annotations=annotations_log_reg_percent, 
    width=800, 
    height=500,
    title_x=0.5
)

fig.show()


# In[27]:


# Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (area = %0.2f)' % roc_auc))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Chance'))
fig.update_layout(title='Receiver Operating Characteristic',
                  xaxis_title='False Positive Rate',
                  yaxis_title='True Positive Rate',
                  width=800,
                  height=500,
                  title_x=0.5)
fig.show()


# In[ ]:





# In[28]:


# Decision Tree Classifier
decision_tree_model = DecisionTreeClassifier()

# Cross-validation for Decision Tree
cv_scores_dt = cross_val_score(decision_tree_model, X_scaled, y, cv=5)
decision_tree_model.fit(X_train_smote, y_train_smote)
y_pred_dt = decision_tree_model.predict(X_test)
report_dt = classification_report(y_test, y_pred_dt)

cv_accuracy_dt = cv_scores_dt.mean()
print("Decision Tree - Cross-Validation Accuracy:", cv_accuracy_dt)
print("Decision Tree - Classification Report:\n", report_dt)


# In[29]:


# Confusion matrix
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)

# ROC curve data - note: this might not be as useful for decision trees
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, decision_tree_model.predict_proba(X_test)[:,1])
roc_auc_dt = auc(fpr_dt, tpr_dt)


# In[30]:


# Calculate the percentage for each cell in the confusion matrix
conf_matrix_dt_percent = conf_matrix_dt / np.sum(conf_matrix_dt) * 100

# Confusion Matrix plot with percentages
fig = go.Figure(data=go.Heatmap(
                    z=conf_matrix_dt_percent,
                    x=['Predicted Negative', 'Predicted Positive'],
                    y=['Actual Negative', 'Actual Positive'],
                    hoverongaps=False,
                    text=np.around(conf_matrix_dt_percent, 2),  # round to 2 decimal places
                    colorscale='blues',
                    ))

# Adding annotations for confusion matrix (percentages)
annotations_dt_percent = []
for i, row in enumerate(conf_matrix_dt_percent):
    for j, value in enumerate(row):
        annotations_dt_percent.append(
            dict(
                x=j, y=i,
                xref='x1', yref='y1',
                text=f"{value:.2f}%",  # format to 2 decimal places
                showarrow=False,
                font=dict(
                    family="Arial",
                    size=20,
                    color="black"
                )
            )
        )

fig.update_layout(
    title='Confusion Matrix (Percentages) - Decision Tree',
    xaxis_title='Predicted Labels',
    yaxis_title='True Labels',
    annotations=annotations_dt_percent, 
    width=800, 
    height=500,
    title_x=0.5
)

fig.show()


# In[31]:


# ROC Curve plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr_dt, y=tpr_dt, mode='lines', name='ROC curve (area = %0.2f)' % roc_auc_dt))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Chance'))
fig.update_layout(title='Receiver Operating Characteristic - Decision Tree',
                  xaxis_title='False Positive Rate',
                  yaxis_title='True Positive Rate',
                  width=800,
                  height=500,
                  title_x=0.5)
fig.show()


# In[ ]:





# In[32]:


from sklearn.ensemble import RandomForestClassifier

# Random Forest Classifier
random_forest_model = RandomForestClassifier()

# Cross-validation for Random Forest
cv_scores_rf = cross_val_score(random_forest_model, X_scaled, y, cv=5)
random_forest_model.fit(X_train_smote, y_train_smote)
y_pred_rf = random_forest_model.predict(X_test)
report_rf = classification_report(y_test, y_pred_rf)

cv_accuracy_rf = cv_scores_rf.mean()
print("Random Forest - Cross-Validation Accuracy:", cv_accuracy_rf)
print("Random Forest - Classification Report:\n", report_rf)


# In[33]:


# Confusion matrix
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# ROC curve data
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, random_forest_model.predict_proba(X_test)[:,1])
roc_auc_rf = auc(fpr_rf, tpr_rf)


# In[34]:


# Calculate the percentage for each cell in the confusion matrix
conf_matrix_rf_percent = conf_matrix_rf / np.sum(conf_matrix_rf) * 100

# Confusion Matrix plot with percentages
fig = go.Figure(data=go.Heatmap(
                    z=conf_matrix_rf_percent,
                    x=['Predicted Negative', 'Predicted Positive'],
                    y=['Actual Negative', 'Actual Positive'],
                    hoverongaps=False,
                    text=np.around(conf_matrix_rf_percent, 2),  # round to 2 decimal places
                    colorscale='blues',
                    ))

# Adding annotations for confusion matrix (percentages)
annotations_rf_percent = []
for i, row in enumerate(conf_matrix_rf_percent):
    for j, value in enumerate(row):
        annotations_rf_percent.append(
            dict(
                x=j, y=i,
                xref='x1', yref='y1',
                text=f"{value:.2f}%",  # format to 2 decimal places
                showarrow=False,
                font=dict(
                    family="Arial",
                    size=20,
                    color="black"
                )
            )
        )

fig.update_layout(
    title='Confusion Matrix (Percentages) - Random Forest',
    xaxis_title='Predicted Labels',
    yaxis_title='True Labels',
    annotations=annotations_rf_percent, 
    width=800, 
    height=500,
    title_x=0.5
)

fig.show()


# In[35]:


# ROC Curve plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode='lines', name='ROC curve (area = %0.2f)' % roc_auc_rf))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Chance'))
fig.update_layout(title='Receiver Operating Characteristic - Random Forest',
                  xaxis_title='False Positive Rate',
                  yaxis_title='True Positive Rate',
                  width=800,
                  height=500,
                  title_x=0.5)
fig.show()


# In[ ]:





# In[36]:


# Generate probability predictions for each model
y_pred_proba_log_reg = log_reg_model.predict_proba(X_test)[:, 1]
y_pred_proba_dt = decision_tree_model.predict_proba(X_test)[:, 1]
y_pred_proba_rf = random_forest_model.predict_proba(X_test)[:, 1]

# Compute ROC curve and ROC area for each model
fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, y_pred_proba_log_reg)
roc_auc_log_reg = auc(fpr_log_reg, tpr_log_reg)

fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plot all ROC curves
fig = go.Figure()

# Logistic Regression ROC
fig.add_trace(go.Scatter(x=fpr_log_reg, y=tpr_log_reg, mode='lines', 
                         name=f'Logistic Regression (area = {roc_auc_log_reg:.2f})'))

# Decision Tree ROC
fig.add_trace(go.Scatter(x=fpr_dt, y=tpr_dt, mode='lines', 
                         name=f'Decision Tree (area = {roc_auc_dt:.2f})'))

# Random Forest ROC
fig.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode='lines', 
                         name=f'Random Forest (area = {roc_auc_rf:.2f})'))

# Chance Line (Diagonal)
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Chance'))

fig.update_layout(title='Receiver Operating Characteristic - All Models',
                  xaxis_title='False Positive Rate',
                  yaxis_title='True Positive Rate',
                  width=800,
                  height=500,
                  title_x=0.5)

fig.show()


# In[ ]:




