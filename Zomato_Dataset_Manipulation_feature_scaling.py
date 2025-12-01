#!/usr/bin/env python
# coding: utf-8

# In[1]:


#libraries uploading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#loading dataset
df = pd.read_csv(r"C:\Users\ADMIN\Desktop\data set\Zomato_dataset.csv")
df.head()


# In[3]:


#data cleaning and manipulation
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


#visualization
plt.figure(figsize=(8,5))
sns.histplot(df['Dining_Rating'], kde=True)
plt.title("Distribution of Dining Rating")
plt.show()


# In[20]:


df['Avg_Rating'] = (df['Dining_Rating'] + df['Delivery_Rating']) / 2
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


# In[22]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='Votes', y='Avg_Rating', data=df)
plt.title("Votes vs Average Rating")
plt.show()


# In[24]:


numeric_df = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# In[25]:


# Feature Engineering
df['Avg_Rating'] = (df['Dining_Rating'] + df['Delivery_Rating']) / 2
df['Total_Votes'] = df['Dining_Votes'] + df['Delivery_Votes']
df.head(5)


# In[29]:


#Binnning
df['Price_Bin'] = pd.cut(
    df['Prices'],
    bins=[0, 200, 400, 700, 2000],
    labels=['Low', 'Medium', 'High', 'Premium']
)
df.head(10)


# In[30]:


df['Rating_Bin'] = pd.cut(
    df['Avg_Rating'],
    bins=[0, 2.5, 3.5, 4.5, 5],
    labels=['Poor', 'Average', 'Good', 'Excellent']
)
df.tail(10)


# In[33]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['City_encoded'] = le.fit_transform(df['City'])
df['Primary_Cuisine_encoded'] = le.fit_transform(df['Cuisine'])
df['Place_Name_encoded'] = le.fit_transform(df['Place_Name'])
df.head()


# In[34]:


#hot-encode
top10 = df['Cuisine'].value_counts().head(10).index
df['Cuisine_Top10'] = df['Cuisine'].apply(lambda x: x if x in top10 else 'Other')

df = pd.get_dummies(df, columns=['Cuisine_Top10'], drop_first=True)
df.head()


# In[40]:


# feature scaling 
#Min-Max Scaling (0 to 1)
from sklearn.preprocessing import MinMaxScaler

m = MinMaxScaler()
m_scaled = m.fit_transform(df.iloc[:, [1,2,3,4,10,11]])

df_scaled_minmax = pd.DataFrame(
    m_scaled,
    columns=df.columns[[1,2,3,4,10,11]]
)

print(df_scaled_minmax.head())


# In[39]:


#standardization(z - score)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df.iloc[:, [1,2,3,4,10,11]])

df_scaled_standard = pd.DataFrame(
    scaled_data,
    columns=df.columns[[1,2,3,4,10,11]]
)
print(df_scaled_standard.head())


# In[41]:


#Polynomial Features
from sklearn.preprocessing import PolynomialFeatures

numeric_data = df.iloc[:, 1:5]     # columns 1 to 4 → 4 numeric features

# Creating polynomial features (degree = 2)
poly = PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(numeric_data)

data_poly = pd.DataFrame(poly_features, 
                         columns=poly.get_feature_names_out(numeric_data.columns))

data_poly.head()


# In[ ]:




