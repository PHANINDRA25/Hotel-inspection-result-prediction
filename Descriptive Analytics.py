#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import fuzz
from fuzzywuzzy import process 
import requests
from bs4 import BeautifulSoup
from itertools import chain
from collections import Counter
import re


# In[2]:


# reading data 
data = pd.read_csv(r"Food_Inspections.csv")


# In[3]:


# making a copy of the dataframe, to preserve the original
df = data.copy()


# In[4]:


# Initial inspection of the data
df.info()


# In[5]:


# function to clean columns to remove spaces, making it lowercase
def clean_columnNames(x):
    x = x.str.lower()
    x = x.str.replace(' ', '')
    return x


# In[6]:


# cleaning columns
df.columns = clean_columnNames(df.columns)


# In[7]:


df.columns


# In[8]:


# Convert 'inspectionid' to strings and handle NaN values
df['inspectionid'] = df['inspectionid'].apply(lambda x: str(int(x)) if not pd.isnull(x) else str(x))

# Convert 'licenseid' (formerly 'license#') to strings and handle NaN values
df['licenseid'] = df['license#'].apply(lambda x: str(int(x)) if not pd.isnull(x) else str(x))
df.drop(['license#'], axis=1, inplace=True)

# Convert 'zip' to strings and handle NaN values
df['zip'] = df['zip'].apply(lambda x: str(int(x)) if not pd.isnull(x) else str(x))

# Convert 'inspectiondate' to datetime and handle NaN values by coercing them to NaT
df['inspectiondate'] = pd.to_datetime(df['inspectiondate'], errors='coerce')


# In[9]:


df.info()


# In[10]:


df.head()


# In[11]:


# Function to strip the data frame values of extra spaces as appropriate

def strip_spaces(x):
    if isinstance(x, str):
        return x.strip()
    else:
        return x


#cleaning spaces
    
df = df.applymap(strip_spaces)


# In[12]:


# dataframe info after changes

df.info()


# In[13]:


# inspecting null values

df.isna().sum()


# # Data Cleaning

# #### DBA NAME

# In[14]:


df[df.dbaname.isna()].address.iloc[0]


# In[15]:


df[df.dbaname.isna()]


# In[16]:


# checking if there is another establishment with the same address

df[df.address == df[df.dbaname.isna()].address.iloc[0]].head(3)


# In[17]:


# filling in the NaN value with the latest name of the restaurant, as the name seems to change with time only ove the years

df.dbaname.fillna('CHIKIS N GRILL', inplace = True)
df['akaname'].fillna(df['dbaname'], inplace=True)
df.loc[df['inspectionid'].isin(['2510397', '2501193']), 'facilitytype'] = 'Restaurant'


# In[18]:


df[df.inspectionid.isin(['2510397','2501193'])]


# #### Facility Type

# In[19]:


# Define the business types
business_types = [
    'bakery', 'banquet hall', 'candy store', 'caterer', 'coffee shop',
    'day care center less than two', 'day care center between two and six',
    'day care center combined', 'gas station',
    'Golden Diner', 'grocery store', 'hospital', 'long term care center nursing home',
    'liquor store', 'mobile food dispenser', 'restaurant', 'paleteria', 'school',
    'shelter', 'tavern', 'social club', 'wholesaler', 'Wrigley Field Rooftop'
]

#finding the closest match to a value, so that we could change the passed item to the nearest match (as required)
def find_closest_match(value):
    if pd.notna(value):  # Check for NaN values
        return process.extractOne(str(value), business_types, scorer=fuzz.partial_ratio)[0]
    else:
        return None  # Return None for NaN values

# Apply the function to create the 'facility_cleaned' column with closest matches and original values, 
# all present in the mentioned business_types list 

df['facility_cleaned'] = df['facilitytype'].apply(find_closest_match)


# In[20]:


print("Percentage Nulls: ", 100*df.facility_cleaned.isna().sum()/df.facility_cleaned.count())


# In[21]:


df[df.facility_cleaned.isna() & df.inspectiontype.isna()]


# In[22]:


# replacing the unknown facility types with "Unknown" value, as there is 2.26 percentage of nulls

df.facility_cleaned.fillna('Unkown', inplace = True)


# In[23]:


df.facility_cleaned.isna().sum()


# #### Risk

# In[24]:


df.risk.describe(include = 'category')


# In[25]:


df.risk.unique()


# In[26]:


# changing "All" risk to the mode value as, it is not defined and invalid value
df.loc[df['risk'] == 'All', 'risk'] = df.risk.mode()[0]


# In[27]:


print("Percentage Nulls: ", 100*df.risk.isna().sum()/df.risk.count())


# In[28]:


df.risk.mode()[0]


# In[29]:


# replacing null values with the mode value, as the null value percentage is really low (0.03%)
df.risk.fillna(df.risk.mode()[0], inplace = True)


# In[30]:


print("Percentage Nulls: ", 100*df.risk.isna().sum()/df.risk.count())


# In[ ]:





# #### Geographics

# In[31]:


df.address.describe()


# In[32]:


df.address.isna().sum()


# In[33]:


df.city.describe()


# In[34]:


df.state.describe()


# In[35]:


df.state.unique()


# In[36]:


# filtering the data for only the restaurants in Illiinois (IL)
df = df[~ df.state.isin(['WI', 'IN', 'NY'])]


# In[37]:


df.state.unique()


# In[38]:


print("Percentage Nulls: ", 100*df.state.isna().sum()/df.state.count())


# In[39]:


mode_value = df['state'].mode()[0]

# replacing the null values with mode value "IL"
df.state.fillna(mode_value, inplace = True)


# In[40]:


df.state.unique()


# In[41]:


# inspecting the unique values of city : looks like there are near close values to "chicago" and then other areas possibly present in chicago

df.city.unique()


# In[42]:


# for consistency
df.city = df.city.str.lower()


# In[43]:


df.city.unique()


# In[44]:


# checking the fuzzy matching score to see how many are closely matching or spelling mistakes of "chicago"
for l in df.city.unique().tolist():
    print(str(l) + ":" + str(fuzz.ratio(str(l), 'chicago')))


# In[45]:


# Function to replace words closest to "Chicago"
def replace_closest_to_chicago(city):
    if fuzz.ratio(str(city), 'chicago') > 50: # threshold set as per the above step examination
        return 'chicago'
    else:
        return city

# Apply the function to the 'city' column
df['city'] = df['city'].apply(replace_closest_to_chicago)


# In[46]:


df.city.nunique()


# In[47]:


'''
Using the Wikipedia list of chicago suburbs to check the presence of the rest of the areas mentioned in the city column 

Using BeautifulSoup4 for web scraping the city names

'''
# URL of the Wikipedia page : the source listing Chicago suburbs
url = 'https://simple.wikipedia.org/wiki/Category:Suburbs_of_Chicago'

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content of the page
soup = BeautifulSoup(response.content, 'html.parser')

# Find the appropriate list in the parsed HTML
suburbs_list = soup.find_all('div', class_='mw-category-group')

my_list = []

for item in suburbs_list:
    my_list.append(item.text.strip('\n'))
    
print(my_list)


# In[48]:


# Iterate through the list and extract the values

sub_list = []
for item in my_list:
    # Split each item by ', Illinois\n'
    parts = item.split(', Illinois\n')
    for part in parts:
        # Split each part by '\n'
        splits = part.split('\n')
        sub_list.append(splits[-1])

final_list = []

for item in sub_list:
    # making sure to remove the ", Illinois" part from the last element of each "ul" element as it doesnt have the same delimiting value
    splits = item.split(', Illinois')
    final_list.append(splits[0])
    
sub_list = [sub.lower() for sub in final_list]


# In[49]:


# replacing the city value with the "chicago" value, if the value is in the above list

def replace_city(city):
    if city in sub_list:
        return 'chicago'
    else:
        return city

df.city = df.city.apply(replace_city)


# In[50]:


df.city.nunique()


# In[51]:


df.city.unique()


# In[52]:


df[df.city == 'charles a hayes']


# In[53]:


# manual checking of the above 8-10 places showed chicago city, so direct imputation
def change_city_name(x):
    return 'chicago'

df.city = df.city.apply(change_city_name)


# In[54]:


print(df.city.nunique())

print(df.city.unique())


# #### Date & Inspection Type

# In[55]:


df.inspectiondate.isna().sum()


# In[56]:


# defined list of inspection types
my_list = ["Canvass", "Consultation", "Complaint", "License", "Suspect Food Poisoning", "Task-Force Inspection"]

#finding the closest match in the inspection types list and assigning it
def find_closest_match(value):
    if pd.notna(value):  # Check for NaN values
        return process.extractOne(str(value), my_list , scorer=fuzz.partial_ratio)[0]
    else:
        return None  # Return None for NaN values

# Apply the function to create the 'facility_cleaned' column
df['inspectiontype'] = df['inspectiontype'].apply(find_closest_match)


# In[57]:


df.inspectiontype.isna().sum()


# In[58]:


df[df.inspectiontype.isna()]


# In[59]:


# replacing just the one missing value with the mode value
df.inspectiontype.fillna(df.inspectiontype.mode(), inplace = True)


# #### Violations

# In[60]:


# inspecting the indices of the rows where there are violations mentioned
print(df[~df.violations.isna()].index)


# In[61]:


# examining the first indexed violation
df.loc[15, 'violations']


# In[62]:


len(df.loc[15, 'violations'].split('|'))


# In[63]:


# function to calculate the number of violations based on the delimiting character
def calculate_num_violations(violation):
    if pd.notna(violation):
        return len(violation.split('|'))
    else:
        return 0

df['num_violations'] = df.violations.apply(calculate_num_violations)

df.num_violations.head()


# In[64]:


# number of rows with violations
len(df[df.num_violations!=0])


# In[65]:


#function to see if ther is the word "CITATION" in the violations mentioned, essentially telling its a bad standards by the restaurant
def find_citation(violation):
    if pd.notna(violation):
        violation = violation.replace('NO CITATION', 'NO_CITATION') # replacing the "NO CITATION" to avoid the overlap
        if ('CITATION' in violation.split(' ')):
            return 1
        else:
            return 0
    else:
        return 0

df['cited'] = df.violations.apply(find_citation)
df.cited.sum()


# In[66]:


# function to get the violation codes from the text 
def extract_numbers(record):
 
  if isinstance(record, str):
 
   
    pattern = r'(?:^|\|)\s*(\d+)\.\s'
    return [int(num) for num in re.findall(pattern, record)]


# In[67]:


df['violation_ids'] = df['violations'].apply(extract_numbers)


# In[68]:


# previously calculated number of violations using the delimiter
df[~df.violations.isna()][[ 'violations' , 'violation_ids', 'num_violations']].loc[15][2]


# In[69]:


# number of violations obtained using the violation codes list obtained
len(df[~df.violations.isna()][[ 'violations' , 'violation_ids', 'num_violations']].loc[15][1])


# In[70]:


# corresponding violation text
df[~df.violations.isna()][[ 'violations' , 'violation_ids', 'num_violations']].loc[15][0]


# In[71]:


# counting the number of elements in the lists for each row
df['count_violation_ids'] = df.violation_ids.apply(lambda x: len(x) if isinstance(x, list) and len(x) > 0 else 0)


# In[72]:


# checking if all the previous outputs line up with the new calculations
df[df.count_violation_ids != df.num_violations]


# In[73]:


# dropping the count_violation_ids column as we dont need it anymore, testing done
df.drop('count_violation_ids', axis = 1, inplace = True)


# In[74]:


df[~df.violation_ids.isna()].violation_ids.loc[15]


# In[75]:


# combining the all the lists obtained for all the rows, to get the top 5 occurences of violations over the years in the whole data
combined_list = list(chain.from_iterable(x if x is not None else [] for x in df['violation_ids']))

# making a dictionary for each element using the collections library
element_count = Counter(combined_list)

print(element_count)


# ### Top 5 Violation Codes
# 
# * 32. FOOD AND NON-FOOD CONTACT SURFACES PROPERLY DESIGNED, CONSTRUCTED AND MAINTAINED
#  
# * 33. FOOD AND NON-FOOD CONTACT EQUIPMENT UTENSILS CLEAN, FREE OF ABRASIVE DETERGENTS
#  
# * 34. FLOORS: CONSTRUCTED PER CODE, CLEANED, GOOD REPAIR, COVING INSTALLED, DUST-LESS CLEANING METHODS USED
#  
# * 35. WALLS, CEILINGS, ATTACHED EQUIPMENT CONSTRUCTED PER CODE: GOOD REPAIR, SURFACES CLEAN AND DUST-LESS CLEANING METHODS
#  
# * 38. VENTILATION: ROOMS AND EQUIPMENT VENTED AS REQUIRED: PLUMBING: INSTALLED AND MAINTAINED

# In[76]:


import pandas as pd
import plotly.express as px

# Convert element_count (Counter) to a DataFrame
df_violations = pd.DataFrame.from_dict(element_count, orient='index', columns=['Count'])
df_violations.reset_index(inplace=True)
df_violations.rename(columns={'index': 'ViolationNumber'}, inplace=True)

# Sort the DataFrame by Count in descending order
df_violations = df_violations.sort_values(by='Count', ascending=False)
df_violations.ViolationNumber = df_violations.ViolationNumber.astype(str)
print(df_violations)

# Get the top 5 violations
top_5_violations = df_violations.head(5)

# Create a bar plot using Plotly with the "index" column as a category
fig = px.bar(
    top_5_violations,
    x='ViolationNumber',
    y='Count',
    labels={'ViolationNumber': 'Violation Code', 'Count': 'Count'},
    title='Top 5 Violations Over the Years',
)

# # Customize the y-axis with tick values every 15,000 units
# y_tick_values = list(range(0, 75001, 15000))  # Generate tick values from 0 to 75,000 with a step of 15,000
# fig.update_yaxes(tickmode='array', tickvals=y_tick_values)

fig.update_layout(width=600, height=500)
fig.update_layout(title_x=0.5)
fig.update_traces(marker_line_color='black', marker_line_width=2, opacity=0.8)

# Show the plot
fig.show()


# In[77]:


# writing to an excel file for further analysis
df.to_csv("inspections.csv")


# In[78]:


import plotly.express as px

# Convert to datetime if not already
df['inspectiondate'] = pd.to_datetime(df['inspectiondate'])

# Trend analysis
monthly_counts = df.resample('M', on='inspectiondate')['licenseid'].count().reset_index()

# Plotting with Plotly
fig = px.line(monthly_counts, x='inspectiondate', y='licenseid', title='Monthly Inspection Counts Over Time')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Number of Inspections')
fig.show()


# In[79]:


df.columns


# In[80]:


map3 = folium.Map(location=[41.798029497076946, -87.60246286753599], zoom_start=10)
map3.add_child(FastMarkerCluster(df[~(df.latitude.isna() | df.longitude.isna())][['latitude', 'longitude']].values.tolist()))
map3.save("save_file.html")

map3


# ## Risk Analysis

# In[ ]:


import pandas as pd
import plotly.express as px

# Assuming you have a DataFrame named df
data = df.groupby('risk')['inspectionid'].count().reset_index()

# Calculate the percentage values
total_inspections = data['inspectionid'].sum()
data['percentage'] = (data['inspectionid'] / total_inspections) * 100

fig = px.bar(data, x='risk', y='percentage', text='percentage',
             title='% of inspections across risk levels', labels={'percentage': 'Percentage'})
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig.update_yaxes(title_text='% of inspections')
fig.update_layout(width=500, height=500,title_x=0.5)

fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))  # Set bar border color
fig.update_traces(marker_line_color='black')  # Set bar border color
fig.update_traces(opacity=0.8) 

fig.show()


# In[ ]:


filtered_df = df[df['results'].isin(['Pass', 'Fail', 'Pass w/ Conditions'])]

riskdf = filtered_df.groupby(['risk', 'results'])['inspectionid'].count().reset_index()

# Calculate the sum of inspection IDs for each risk category
riskdf['sum_inspectionid'] = riskdf.groupby('risk')['inspectionid'].transform('sum')


riskdf['perc'] = (riskdf['inspectionid'] / riskdf['sum_inspectionid']) * 100


riskdf['perc'] = round(riskdf['perc'], 1)

riskdf = riskdf.sort_values(by=['risk', 'perc'], ascending=[True, False])


print(riskdf)


# In[ ]:


fig = px.bar(riskdf, x='risk', y='perc', color='results', barmode='group')

# Customize the chart layout (optional)
fig.update_layout(
    title='Clustered Bar Chart of Risks & its results',
    xaxis_title='Risk',
    yaxis_title='Percentage',
    legend_title='Result',
    height=500,  # Adjust the height of the chart
    width=700,
    title_x=0.42,
)

fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))  # Set bar border color
fig.update_traces(marker_line_color='black')  # Set bar border color
fig.update_traces(opacity=0.8) 
#fig.update_traces(marker_color='blue')

# Add data labels
fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')

# Show the chart
fig.show()


# ## Analysis of Facility types

# In[ ]:


facility=pd.DataFrame(df.facility_cleaned.value_counts())
facility['percentage']=facility['count']/(facility['count'].sum())*100
facility['cumulative']=facility['percentage'].cumsum()*100
facility.reset_index(inplace=True)
facility['percentage']=round(facility['percentage'],1)
facility.rename(columns={'facility_cleaned':'facility'},inplace=True)
facility


# In[ ]:


import pandas as pd
import plotly.express as px

fig = px.bar(facility[0:3], x='facility', y='percentage', text='percentage',
             title='Top 3 inspected facilities', labels={'percentage': 'Percentage'})
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig.update_yaxes(title_text='% of inspections')
fig.update_layout(width=500, height=500,title_x=0.5)

fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))  # Set bar border color
fig.update_traces(marker_line_color='black')  # Set bar border color
fig.update_traces(opacity=0.8) 

fig.show()


# In[ ]:


rest_3=df[df['facility_cleaned'].isin(['restaurant','grocery store','school'])][['facility_cleaned','results']]
rest_3_results=pd.DataFrame(rest_3.groupby('facility_cleaned')['results'].value_counts()).reset_index()

rest = rest_3_results[rest_3_results['facility_cleaned'] == 'restaurant']
rest1 = rest.assign(perc=rest['count'] / rest['count'].sum() * 100).assign(cum=lambda x: x['perc'].cumsum())[['facility_cleaned', 'results', 'perc']]
rest_final=rest1[rest1['results'].isin(['Pass','Fail','Pass w/ Conditions'])]
rest_final


# In[ ]:


grocery = (rest_3_results[rest_3_results['facility_cleaned'] == 'grocery store']
      .assign(perc=lambda x: x['count'] / x['count'].sum() * 100)
      [['facility_cleaned', 'results', 'perc']])
grocery_final=grocery[grocery['results'].isin(['Pass','Fail','Pass w/ Conditions'])]
grocery_final


# In[ ]:


school = rest_3_results[rest_3_results['facility_cleaned'] == 'school'].assign(perc=lambda x: x['count'] / x['count'].sum() * 100, cum=lambda x: x['perc'].cumsum())[['facility_cleaned', 'results', 'perc']]
school_final=school[school['results'].isin(['Pass','Fail','Pass w/ Conditions'])]
school_final


# In[ ]:


result = pd.concat([rest_final,grocery_final, school_final], axis=0)

# Reset the index if needed
result = result.reset_index(drop=True)
result


# In[ ]:


fig = px.bar(result, x='facility_cleaned', y='perc', color='results', barmode='group')

# Customize the chart layout (optional)
fig.update_layout(
    title='Clustered Bar Chart of Facilities & its results',
    xaxis_title='Facility type',
    yaxis_title='Percentage',
    legend_title='Result',
    height=500,  # Adjust the height of the chart
    width=700,
    title_x=0.42,
)

fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))  # Set bar border color
fig.update_traces(marker_line_color='black')  # Set bar border color
fig.update_traces(opacity=0.8) 
#fig.update_traces(marker_color='blue')

# Add data labels
fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')

# Show the chart
fig.show()


# ## Analysis of Inspection types

# In[ ]:


inspection=pd.DataFrame(df.inspectiontype.value_counts()).reset_index()
inspection['percentage'] = (inspection['count'] / inspection['count'].sum() * 100).round(1).sort_values(ascending=False)
inspection


# In[ ]:


fig = px.bar(inspection[0:3], x='inspectiontype', y='percentage', text='percentage', title='Inspection type analysis')
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig.update_yaxes(title_text='% of inspections')
fig.update_xaxes(tickangle=0)
fig.update_layout(width=600, height=550,title_x=0.5)
fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))  # Set bar border color
fig.update_traces(marker_line_color='black')  # Set bar border color
fig.update_traces(opacity=0.8) 
fig.update_layout(bargap=0.1, bargroupgap=0.3)
fig.show()


# In[ ]:


inspect1=df[df['inspectiontype'].isin(['Canvass','License','Complaint'])].groupby(['inspectiontype','results'])['inspectionid'].count().reset_index()
inspect1.sort_values(by='inspectionid',ascending=False)
inspect1
inspect1['sum_inspectionid'] = inspect1.groupby('inspectiontype')['inspectionid'].transform('sum')
inspect1['percentage'] = inspect1['inspectionid'] / inspect1['sum_inspectionid']*100
inspect1['percentage']=round(inspect1['percentage'],1)
inspect1 = inspect1.sort_values(by=['inspectiontype', 'percentage'], ascending=[True, False])
inspect_final=inspect1[inspect1['results'].isin(['Pass','Fail','Pass w/ Conditions'])]
inspect_final


# In[ ]:


fig = px.bar(inspect_final, x='inspectiontype', y='percentage', color='results', barmode='group')

# Customize the chart layout (optional)
fig.update_layout(
    title='Clustered Bar Chart of Facilities & its results',
    xaxis_title='Facility type',
    yaxis_title='Percentage',
    legend_title='Result',
    height=500,  # Adjust the height of the chart
    width=700,
    title_x=0.42,
)

fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))  # Set bar border color
fig.update_traces(marker_line_color='black')  # Set bar border color
fig.update_traces(opacity=0.8) 
#fig.update_traces(marker_color='blue')

# Add data labels
fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')

# Show the chart
fig.show()


# ## Analysis of Restaurants

# In[ ]:


# Calculate the total counts per dbaname
total_counts = df['dbaname'].value_counts().head()

# Calculate the failed counts per dbaname
failed_counts = df[df['results'] == 'Fail']['dbaname'].value_counts().head()

# Calculate the failure percentage
failed_perc_df = (failed_counts / total_counts).reset_index().rename(columns={'index': 'dbaname', 'dbaname': 'failure_percentage'})
failed_perc=failed_perc_df.sort_values(by = 'failure_percentage', ascending = False)
failed_perc['count'].fillna(0,inplace=True)
fail=failed_perc[failed_perc['failure_percentage'].isin(['DUNKIN DONUTS','MCDONALD\'S','SUBWAY'])].reset_index()
fail['count']=fail['count']*100
fail.drop('index',axis=1,inplace=True)
fail=fail.rename(columns={'failure_percentage': 'Restaurant', 'count': 'failure%'})
fail


# In[ ]:


fig = px.bar(fail, x='Restaurant', y='failure%', text='failure%', title='Restaurant analysis')
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig.update_yaxes(title_text='% of inspections')
fig.update_xaxes(tickangle=0)
fig.update_layout(width=600, height=550,title_x=0.5)
fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))  # Set bar border color
fig.update_traces(marker_line_color='black')  # Set bar border color
fig.update_traces(opacity=0.8) 
fig.update_layout(bargap=0.1, bargroupgap=0.3)
fig.show()


# ## Analysis of Violations
# 

# In[ ]:


# combining the all the lists obtained for all the rows, to get the top 5 occurences of violations over the years in the whole data
combined_list = list(chain.from_iterable(x if x is not None else [] for x in df[~(df['violation_ids'].isna())].violation_ids))
 
# making a dictionary for each element using the collections library
element_count = Counter(combined_list)
 
print(element_count)


# In[ ]:


top_violations = sorted(element_count.items(), key=lambda item: item[1], reverse=True)[:5]

# Convert the top 5 into a DataFrame
df_top_violations = pd.DataFrame(top_violations, columns=['violation_code', 'count'])
df_top_violations


# In[ ]:


import pandas as pd
import plotly.express as px
 
# Convert element_count (Counter) to a DataFrame
df_violations = pd.DataFrame.from_dict(element_count, orient='index', columns=['Count'])
df_violations.reset_index(inplace=True)
df_violations.rename(columns={'index': 'ViolationNumber'}, inplace=True)
 
# Sort the DataFrame by Count in descending order
df_violations = df_violations.sort_values(by='Count', ascending=False)
df_violations.ViolationNumber = df_violations.ViolationNumber.astype(str)
print(df_violations)
 
# Get the top 5 violations
top_5_violations = df_violations.head(5)
 
# Create a bar plot using Plotly with the "index" column as a category
fig = px.bar(
    top_5_violations,
    x='ViolationNumber',
    y='Count',
    labels={'ViolationNumber': 'Violation Code', 'Count': 'Count'},
    title='Top 5 Violations Over the Years',
)
 
# # Customize the y-axis with tick values every 15,000 units
# y_tick_values = list(range(0, 75001, 15000))  # Generate tick values from 0 to 75,000 with a step of 15,000
# fig.update_yaxes(tickmode='array', tickvals=y_tick_values)
 
fig.update_layout(width=600, height=500)
fig.update_layout(title_x=0.5)
fig.update_traces(marker_line_color='black', marker_line_width=2, opacity=0.8)
 
# Show the plot
fig.show()


# ## Time Series Forecasting

# In[ ]:


filtered_df = df[df['results'].isin(['Pass', 'Fail'])]
filtered_df['results'].value_counts()
# Convert the inspectiondate to datetime format and aggregate by month
filtered_df['inspectiondate'] = to_datetime(filtered_df['inspectiondate'])
monthly_data = filtered_df.groupby([filtered_df['inspectiondate'].dt.to_period('M'), 'results']).size().unstack(fill_value=0)
monthly_data.index = monthly_data.index.to_timestamp()


# In[ ]:


# Extract the 'Pass' column
pass_counts = monthly_data['Pass']

# Perform Augmented Dickey-Fuller test for stationarity
adf_test = adfuller(pass_counts)
print("ADF Statistic:", adf_test[0])
print("p-value:", adf_test[1])


# In[ ]:


# Plot ACF and PACF
plt.figure(figsize=(12, 6))
plot_acf(pass_counts, lags=20)
plt.title('Autocorrelation Function')
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(pass_counts, lags=20)
plt.title('Partial Autocorrelation Function')
plt.show()


# In[ ]:


# Convert dates and prepare the dataset
filtered_df['inspectiondate'] = to_datetime(filtered_df['inspectiondate'])
monthly_data = filtered_df.groupby([filtered_df['inspectiondate'].dt.to_period('M'), 'results']).size().unstack(fill_value=0)
monthly_data.index = monthly_data.index.to_timestamp()
pass_counts = monthly_data['Pass']


# In[ ]:


# Convert dates and prepare the dataset
filtered_df['inspectiondate'] = pd.to_datetime(filtered_df['inspectiondate'])
monthly_data = filtered_df.groupby([filtered_df['inspectiondate'].dt.to_period('M'), 'results']).size().unstack(fill_value=0)
monthly_data.index = monthly_data.index.to_timestamp()
pass_counts = monthly_data['Pass'].reset_index()
pass_counts.columns = ['ds', 'y']  # Prophet requires the column names to be 'ds' and 'y'


# In[ ]:


# Create and fit the model
model = Prophet()
model.fit(pass_counts)

# Create a dataframe for future dates (5 years into the future)
future = model.make_future_dataframe(periods=2, freq='Y')
forecast = model.predict(future)

# Plot the forecast
fig1 = model.plot(forecast)

# Show and/or save the plot
plt.show()


# In[ ]:


# Load your data (same steps as before)
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go

# Create and fit the model
model = Prophet()
model.fit(pass_counts)

# Create a dataframe for future dates (5 years into the future)
future = model.make_future_dataframe(periods=2, freq='Y')
forecast = model.predict(future)

# Create Plotly graph
fig = go.Figure()

# Add actual data
fig.add_trace(go.Scatter(x=pass_counts['ds'], y=pass_counts['y'], mode='lines', name='Actual'))

# Add forecast data
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

# Add uncertainty intervals (if desired)
fig.add_trace(go.Scatter(
    name='Upper Bound',
    x=forecast['ds'],
    y=forecast['yhat_upper'],
    mode='lines',
    marker=dict(color="#444"),
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    name='Lower Bound',
    x=forecast['ds'],
    y=forecast['yhat_lower'],
    marker=dict(color="#444"),
    line=dict(width=0),
    mode='lines',
    fillcolor='rgba(68, 68, 68, 0.3)',
    fill='tonexty',
    showlegend=False
))

# Update layout
fig.update_layout(title='Prophet Forecast', xaxis_title='Date', yaxis_title='Pass Counts', hovermode='x')

# Show and/or save the plot
fig.show()


# In[ ]:


import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go

# Create and fit the model
model = Prophet()
model.fit(pass_counts)

# Create a dataframe for future dates (5 years into the future)
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

# Filter the forecast to the forecasted period only
forecasted_period = forecast[forecast['ds'] > pass_counts['ds'].max()]

# Filter the actual data to the last part that corresponds to the forecasted period
# Assuming pass_counts is sorted by date
last_actual_date = pass_counts['ds'].max()
actual_for_forecasted_period = pass_counts[pass_counts['ds'] > last_actual_date - pd.DateOffset(years=5)]

# Create Plotly graph
fig = go.Figure()

# Add actual data for the last part
fig.add_trace(go.Scatter(x=actual_for_forecasted_period['ds'], y=actual_for_forecasted_period['y'], mode='lines', name='Actual'))

# Add forecast data
fig.add_trace(go.Scatter(x=forecasted_period['ds'], y=forecasted_period['yhat'], mode='lines', name='Forecast'))

# Update layout
fig.update_layout(title='Prophet Forecast for Next 5 Years', xaxis_title='Date', yaxis_title='Pass Counts', hovermode='x')

# Show and/or save the plot
fig.show()


# In[ ]:


# Create and fit the model
model = Prophet()
model.fit(pass_counts)

# Create a dataframe for future dates (5 years into the future)
future = model.make_future_dataframe(periods=24, freq='M')  # Adjusted to 60 months for 5 years
forecast = model.predict(future)

# Filter the forecast to the forecasted period only
forecasted_period = forecast[forecast['ds'] > pass_counts['ds'].max()]

# Filter the actual data to the last part that corresponds to the forecasted period
# Assuming pass_counts is sorted by date
last_actual_date = pass_counts['ds'].max()
actual_for_forecasted_period = pass_counts[pass_counts['ds'] > last_actual_date - pd.DateOffset(years=5)]

# Create Plotly graph
fig = go.Figure()

# Add actual data for the last part
fig.add_trace(go.Scatter(x=actual_for_forecasted_period['ds'], y=actual_for_forecasted_period['y'], mode='lines', name='Actual'))

# Add forecast data
fig.add_trace(go.Scatter(x=forecasted_period['ds'], y=forecasted_period['yhat'], mode='lines', name='Forecast'))

# Add upper bound of forecast
fig.add_trace(go.Scatter(x=forecasted_period['ds'], y=forecasted_period['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dot')))

# Add lower bound of forecast
fig.add_trace(go.Scatter(x=forecasted_period['ds'], y=forecasted_period['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dot')))

# Update layout
fig.update_layout(title='Prophet Forecast for Next 2 Years', xaxis_title='Date', yaxis_title='Pass Counts', hovermode='x',title_x=0.5)

# Show and/or save the plot
fig.show()


# In[ ]:


# Split the data into training and testing sets
train = pass_counts[:-12]  # Excluding the last 12 months for testing
test = pass_counts[-12:]  # Last 12 months

# Fit the model on the training set
model = Prophet()
model.fit(train)

# Make predictions on the test set
future = test.drop('y', axis=1)
forecast = model.predict(future)

# Calculate RMSE
rmse = sqrt(mean_squared_error(test['y'], forecast['yhat']))
print('RMSE:', rmse)


# In[ ]:




