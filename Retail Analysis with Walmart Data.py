#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing required libraries
import pandas as pd
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


# In[2]:


# Import Data set

df_walmart = pd.read_csv('Walmart_Store_sales.csv')


# In[3]:


# default display the first five rows from the dataset
df_walmart.head()


# In[4]:


#basic infomation about dataset
df_walmart.info()
df_walmart.shape


# In[5]:


# find the maximum value in each column
df_walmart.max()


# In[6]:


# which store has maximum sales in this dataset?
df_walmart.loc[df_walmart['Weekly_Sales'] == df_walmart['Weekly_Sales'].max()]
#maximum weekly sales


# In[7]:


# Which store has maximum standard deviation i.e. the sales vary a lot. Also, find out the coefficient of varience
#(cov)
# Grouping by store and finding the standard deviation and mean of each store.

maxstd=pd.DataFrame(df_walmart.groupby('Store').agg({'Weekly_Sales':['std','mean']}))

# resetting the index.
maxstd=maxstd.reset_index()

# Now we know that COV is  std/mean we are doing this for each store.
maxstd['Cov'] = (maxstd[('Weekly_Sales','std')]/maxstd[('Weekly_Sales','mean')])*100

#finding the store with maximum standard deviation.
maxstd.loc[maxstd[('Weekly_Sales','std')]==maxstd[('Weekly_Sales','std')].max()]


# In[8]:


# which store has good quarterly growth rate in Q3'2012
# converting the data type of date column to datetime
df_walmart['Date']= pd.to_datetime(df_walmart['Date'])
df_walmart.head()

# defining the start and end date of Q3 and Q2

Q2_date_from = pd.Timestamp(date(2012,4,1))
Q2_date_to = pd.Timestamp(date(2012,6,30))

Q3_date_from = pd.Timestamp(date(2012,7,1))
Q3_date_to = pd.Timestamp(date(2012,9,30))


#collecting the data of Q3 and Q2 from original dataset.
Q2data=df_walmart[(df_walmart['Date']> Q2_date_from) & (df_walmart['Date']<Q2_date_to)]
Q3data=df_walmart[(df_walmart['Date']> Q3_date_from) & (df_walmart['Date']<Q3_date_to)]

# Finding sum weekly sales of each store in Q2

Q2 = pd.DataFrame(Q2data.groupby('Store')['Weekly_Sales'].sum())
Q2.reset_index(inplace=True)
Q2.rename(columns={'Weekly_Sales': 'Q2_Weekly_Sales'},inplace=True)

# Finding sum weekly sales of each store in Q3

Q3 = pd.DataFrame(Q3data.groupby('Store')['Weekly_Sales'].sum())
Q3.reset_index(inplace=True)
Q3.rename(columns={'Weekly_Sales': 'Q3_Weekly_Sales'},inplace=True)

#merging Q2 and Q3 data on store as a common column
Q3_growth = Q2.merge(Q3,how='inner',on='Store')


# In[9]:


# Calculating growth rate of each store and collecting it into a dataframe

Q3_growth['Growth_Rate'] = (Q3_growth['Q3_Weekly_Sales'] - Q3_growth['Q2_Weekly_Sales'])/Q3_growth['Q2_Weekly_Sales']

Q3_growth['Growth_Rate']= round(Q3_growth['Growth_Rate'],2)
Q3_growth.sort_values('Growth_Rate',ascending=False).head(1)


# In[10]:


Q3_growth.sort_values('Growth_Rate',ascending=False).tail(1)


# In[11]:


#Some holidays have a negative impact on sales. Find out holidays which have higher sales than the mean sales in non-holiday season for all stores together
#finding the mean sales of holiday and non holiday

df_walmart.groupby('Holiday_Flag')['Weekly_Sales'].mean()


# In[12]:


#marking the holiday dates 
Christmas1 = pd.Timestamp(date(2010,12,31) )
Christmas2 = pd.Timestamp(date(2011,12,30) )
Christmas3 = pd.Timestamp(date(2012,12,28) )
Christmas4 = pd.Timestamp(date(2013,12,27) )

Thanksgiving1=pd.Timestamp(date(2010,11,26) )
Thanksgiving2=pd.Timestamp(date(2011,11,25) )
Thanksgiving3=pd.Timestamp(date(2012,11,23) )
Thanksgiving4=pd.Timestamp(date(2013,11,29) )

LabourDay1=pd.Timestamp(date(2010,2,10) )
LabourDay2=pd.Timestamp(date(2011,2,9) )
LabourDay3=pd.Timestamp(date(2012,2,7) )
LabourDay4=pd.Timestamp(date(2013,2,6) )

SuperBowl1=pd.Timestamp(date(2010,9,12) )
SuperBowl2=pd.Timestamp(date(2011,9,11) )
SuperBowl3=pd.Timestamp(date(2012,9,10) )
SuperBowl4=pd.Timestamp(date(2013,9,8) )


# In[13]:


#Calculating the mean sales during the holidays
Christmas_mean_sales=df_walmart[(df_walmart['Date'] == Christmas1) | (df_walmart['Date'] == Christmas2) | (df_walmart['Date'] == Christmas3) | (df_walmart['Date'] == Christmas4)]
Thanksgiving_mean_sales=df_walmart[(df_walmart['Date'] == Thanksgiving1) | (df_walmart['Date'] == Thanksgiving2) | (df_walmart['Date'] == Thanksgiving3) | (df_walmart['Date'] == Thanksgiving4)]
LabourDay_mean_sales=df_walmart[(df_walmart['Date'] == LabourDay1) | (df_walmart['Date'] == LabourDay2) | (df_walmart['Date'] == LabourDay3) | (df_walmart['Date'] == LabourDay4)]
SuperBowl_mean_sales=df_walmart[(df_walmart['Date'] == SuperBowl1) | (df_walmart['Date'] == SuperBowl2) | (df_walmart['Date'] == SuperBowl3) | (df_walmart['Date'] == SuperBowl4)]
#
list_of_mean_sales = {'Christmas_mean_sales' : round(Christmas_mean_sales['Weekly_Sales'].mean(),2),
'Thanksgiving_mean_sales': round(Thanksgiving_mean_sales['Weekly_Sales'].mean(),2),
'LabourDay_mean_sales' : round(LabourDay_mean_sales['Weekly_Sales'].mean(),2),
'SuperBowl_mean_sales':round(SuperBowl_mean_sales['Weekly_Sales'].mean(),2),
'Non holiday weekly sales' : df_walmart[df_walmart['Holiday_Flag'] == 0 ]['Weekly_Sales'].mean()}
list_of_mean_sales


# In[14]:


#Provide a monthly and semester view of sales in units and give insights

#Monthly sales 
monthly = df_walmart.groupby(pd.Grouper(key='Date', freq='1M')).sum()# groupby each by month
monthly=monthly.reset_index()
fig, ax = plt.subplots(figsize=(10,8))
X = monthly['Date']
Y = monthly['Weekly_Sales']
plt.plot(X,Y)
plt.title('Month Wise Sales')
plt.xlabel('Monthly')
plt.ylabel('Weekly_Sales')

#Quaterly Sales 
Quaterly = df_walmart.groupby(pd.Grouper(key='Date', freq='3M')).sum()
Quaterly = Quaterly.reset_index()
fig, ax = plt.subplots(figsize=(10,8))
X = Quaterly['Date']
Y = Quaterly['Weekly_Sales']
plt.plot(X,Y)
plt.title('Quaterly Wise Sales')
plt.xlabel('Quaterly')
plt.ylabel('Weekly_Sales')
#Semester Sales 
Semester = df_walmart.groupby(pd.Grouper(key='Date', freq='6M')).sum()
Semester = Semester.reset_index()
fig, ax = plt.subplots(figsize=(10,8))
X = Semester['Date']
Y = Semester['Weekly_Sales']
plt.plot(X,Y)
plt.title('Semester Wise Sales')
plt.xlabel('Semester')
plt.ylabel('Weekly_Sales')


# In[ ]:




