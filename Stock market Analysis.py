#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# In[5]:


get_ipython().system('pip install chart-studio')




# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import chart_studio.plotly as py
from plotly.offline import plot 


# In[9]:


tesla=pd.read_csv("C:\\Users\\20106\\Desktop\\datasetsandcodefilesstockmarketprediction_1\\tesla.csv")


# In[12]:


tesla


# In[14]:


tesla['Date'] = pd.to_datetime(tesla['Date'])


# In[15]:


print(f'Dataframe contains stock prices between {tesla.Date.min()} and {tesla.Date.max()}')


# In[16]:


print(f'Total days={(tesla.Date.max()-tesla.Date.min())} days')


# In[17]:


tesla.describe()


# In[18]:


tesla[["Open", "High", "Low", "Close", "Adj Close"]].plot(kind='box')


# In[19]:


import plotly.graph_objects as go

layout = go.Layout(
    title='Stock price of Tesla',
    xaxis=dict(
        title='Date',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Price',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

tesla_data = [{'x': tesla['Date'], 'y': tesla['Close']}]
plot = go.Figure(data=tesla_data, layout=layout)


# In[20]:


import plotly.offline as py
py.iplot(plot)


# In[22]:


X=np.array(tesla.index).reshape(-1,1)


# In[23]:


Y=tesla['Close']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=101)


# In[24]:


from sklearn.preprocessing import StandardScaler


# In[25]:


Scaler=StandardScaler().fit(X_train)


# In[26]:


from sklearn.linear_model import LinearRegression


# In[27]:


lm=LinearRegression()


# In[28]:


lm.fit(X_train,Y_train)


# In[29]:


trace0 = go.Scatter(
    x=X_train.T[0],
    y=Y_train,
    mode='markers',
    name='Actual'
)

trace1 = go.Scatter(
    x=X_train.T[0],
    y=lm.predict(X_train).T,
    mode='lines',
    name='Predicted'
)
tesla_data=[trace0,trace1]
layout.xaxis.title.text='Day'
plot2=go.Figure(data=tesla_data,layout=layout)


# In[30]:


plot2


# In[32]:


from sklearn.metrics import r2_score, mean_squared_error as mse

# Assuming you have defined variables Y_train, X_train, Y_test, X_test, lm

scores = f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'R2 Score'.ljust(10)}{r2_score(Y_train, lm.predict(X_train))}\t {r2_score(Y_test, lm.predict(X_test))}
{'MSE'.ljust(10)}{mse(Y_train, lm.predict(X_train))}\t {mse(Y_test, lm.predict(X_test))}
'''

print(scores)


# lm.fit(X_train,Y_train)
