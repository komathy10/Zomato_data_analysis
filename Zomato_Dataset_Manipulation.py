#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r"C:\Users\ADMIN\Desktop\data set\Zomato_dataset.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.columns


# In[7]:


df.isnull().sum()


# In[8]:


df['Delivery_Rating'] = df['Delivery_Rating'].fillna(df['Delivery_Rating'].median())


# In[9]:


df['Best_Seller'] = df['Best_Seller'].fillna("None")


# In[10]:


df.isnull().sum()


# In[11]:


df['Dining_Rating'] = df['Dining_Rating'].fillna(0)


# In[12]:


df.isnull().sum()


# In[13]:


df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")
df.columns


# In[14]:


df.info()


# In[15]:


# Feature Engineering
df['Avg_Rating'] = (df['Dining_Rating'] + df['Delivery_Rating']) / 2


# In[16]:


plt.figure(figsize=(8,5))
sns.histplot(df['Dining_Rating'], kde=True)
plt.title("Distribution of Dining Rating")
plt.show()


# In[19]:


plt.figure(figsize=(10,6))
df['Cuisine'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Cuisines")
plt.xlabel("Cuisine")
plt.ylabel("Count")
plt.show()


# In[21]:


plt.figure(figsize=(8,5))
df['Prices'].value_counts().plot(kind='bar', color='purple')
plt.title("Restaurants by Price Category")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()


# In[24]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='Votes', y='Avg_Rating', data=df)
plt.title("Votes vs Average Rating")
plt.show()


# In[27]:


numeric_df = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# In[ ]:




