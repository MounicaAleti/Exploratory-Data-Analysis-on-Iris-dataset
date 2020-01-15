
# coding: utf-8

# # Hands-on Practice for Module 1: Exploratory Data Analysis

# ### 0. Importing necessary packages

# In[ ]:


# data loading and computing functionality
import pandas as pd
import numpy as np
import scipy as sp

# datasets in sklearn package
from sklearn.datasets import load_digits

# visualization packages
import seaborn as sns
import matplotlib.pyplot as plt

#PCA, SVD, LDA
from sklearn.decomposition import PCA
from scipy.linalg import svd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



# ### 1. Loading data, determining samples, attributes, and types of attributes

# <span style="color:red">**Question:** </span> Where is the data obtained from?
# 
# <span style="color:green">Answer: </span> 
# Data is obtained from the URL https://raw.githubusercontent.com/plotly/datasets/master/iris.csv that is originally part of the UCI Machine Learning repository. https://archive.ics.uci.edu/ml/datasets/iris
# 

# In[2]:


import pandas as pd
iris_df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/iris.csv')


# In[3]:


type(iris_df)


# <span style="color:red">**Question:** </span> What does the data capture?
# 
# 

# 
# <span style="color:green">Answer: </span> 
# Data captures four properties (Sepal Length, Sepal Width, Petal Length, Petal Width) of three types of Iris plants. 
# 

# <span style="color:red">**Question:** </span> How many data points are there? 

# In[4]:


iris_df.shape


# <span style="color:green">Answer: </span> 
# There are 50 instances/data points for each type. Collectively, 150 data points. 
# 

# <span style="color:red">**Question:** </span> What is the dimensionality?
# 

# In[5]:


iris_df.dtypes


# In[6]:


iris_df.head()


# <span style="color:green">Answer: </span> 
# There are four attributes (Sepal Length, Sepal Width, Petal Length, Petal Width) and one label (Name).
# 

# <span style="color:red">**Question:** </span> What type of attributes are present in the dataset? 

# In[7]:


iris_df.dtypes


# <span style="color:green">Answer: </span> 
# All four attributes are continuous-valued.

# ### 2. Generating summary statistics

# <span style="color:red">**Question:** </span> What are range of values these numeric attributes take? 

# In[8]:


iris_df.describe()


# <span style="color:red">**Question:** </span> What are the mean values for each of the attributes?

# In[9]:


from pandas.api.types import is_numeric_dtype

for col in iris_df.columns:
    if is_numeric_dtype(iris_df[col]):
        print('%s:' % (col))
        print('\t Mean = %.2f' % iris_df[col].mean())
        


# <span style="color:red">**Question:** </span> What is the variance for each of the attributes?

# In[10]:


from pandas.api.types import is_numeric_dtype

for col in iris_df.columns:
    if is_numeric_dtype(iris_df[col]):
        print('%s:' % (col))
        print('\t Variance = %.2f' % iris_df[col].var())        


# <span style="color:red">**Question:** </span> Visually examine how the attribute PetalLength is distributed and comment if the data is Normally distributed? 

# Introducing **Seaborn**, a statistical data visualization library
# 
# Visualizing a histogram for a numerical attribute using distplot function in seaborn

# In[11]:


sns.distplot(iris_df['PetalLength']);


# <span style="color:green">Answer: </span> 
# PetalLength is not Normally distributed. The distribution appears to be bimodal. 

# <span style="color:red">**Question:** </span> Visually examine how the label (Name) attribute is distributed and comment if the data is uniformly distributed? 

# In[12]:


sns.countplot(iris_df['Name']);


# <span style="color:green">Answer: </span> 
# The label is uniformly distributed as all classes have exactly 50 data points. 

# ### 3. Geometric view and Probabilistic view

# For this part, we will restrict to SepalLength and SepalWidth attributes as we can only visualize 2D space.

# <span style="color:red">**Question:** </span> Show the Geometric view of the data on a 2D space along with the mean. 

# In[13]:


iris_df_new = iris_df[['SepalLength','SepalWidth']]


# In[14]:


iris_df_new.head()


# In[1]:


fig, ax = plt.subplots()
sns.scatterplot(x='SepalLength',y='SepalWidth',data=iris_df_new,ax=ax)
mu = np.mean(iris_df_new.values,0)
sns.scatterplot(x=[mu[0], mu[0]],y=[mu[1], mu[1]],color='r',ax=ax)


# <span style="color:red">**Question:** </span> Based on the geometric view of the data, which of the points [6.5, 3.0], [7.5, 3.0] are more closer to the mean?

# <span style="color:green">Answer: </span> [6.5, 3.0] is more closer to the mean shown in red circle.

# <span style="color:red">**Question:** </span> Show the probabilistic view of the data. Assume that the data is drawn from a 2D distribution.

# In[16]:


from scipy.stats import multivariate_normal

mu = np.mean(iris_df_new.values,0)
Sigma = np.cov(iris_df_new.values.transpose())

min_length = np.min(iris_df_new.values[:,0]);
min_width = np.min(iris_df_new.values[:,1]);
max_length = np.max(iris_df_new.values[:,0]);
max_width = np.max(iris_df_new.values[:,1]);
x, y = np.mgrid[min_length:max_length:50j, min_width:max_width:50j]

positions = np.empty(x.shape + (2,))
positions[:, :, 0] = x; 
positions[:, :, 1] = y

F = multivariate_normal(mu, Sigma)
Z = F.pdf(positions)


# In[17]:


fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.imshow(np.rot90(Z), cmap='coolwarm', extent=[min_length,max_length, min_width,max_width], alpha=0.3)
cset = ax.contour(x, y, Z, colors='k', alpha=0.7)
plt.scatter(iris_df_new.values[:,0],iris_df_new.values[:,1],alpha=0.8)
ax.set_xlabel('SepalLength')
ax.set_ylabel('SepalWidth')
plt.title('Probabilistic view')


# <span style="color:red">**Question:** </span> Based on the probabilistic view of the data, which of the points [5.8, 3.0], [6.5, 3.0] are more likely to be generated from the bivariate normal distribution? Provide your reason.

# <span style="color:green">Answer: </span> [5.8, 3.0]. The probability density at this point is higher than that of [6.5, 3.0].

# ### 3. Understanding the (in)dependencies among attributes using Covariance matrix

# <span style="color:red">**Question:** </span> What is the covariance matrix? 

# Selecting the relevant data...

# In[18]:


data = iris_df.values[:,0:4]
data[1:10,:]


# In[19]:


def mycov(data, col_a, col_b):
    mu = np.mean(data, axis=0) #compute mean
    sum = 0;
    for i in range(0, len(data)):
        sum += ((data[i,col_a] - mu[col_a]) * (data[i,col_b] - mu[col_b]))

    return sum/(len(data)-1)


# In[20]:


[mycov(data,0,0), mycov(data,0,1), mycov(data,0,2), mycov(data,0,3)]


# In[21]:


print('Covariance:')
iris_df.cov()


# <span style="color:red">**Question:** </span> Which pairs of attributes co-vary in the same direction?

# <span style="color:green">Answer: </span> 
# 
# SepalLength, PetalLength
# 
# SepalLength, PetalWidth
# 
# PetalLength, PetalWidth

# <span style="color:red">**Question:** </span> Which pairs of attributes are highly correlated? List all such pairs.

# In[22]:


print('Correlation:')
iris_df.corr()


# <span style="color:green">Answer: </span> 
# Highly correlated pairs listed in decreasing order of correlation.
# 
# PetalLength, PetalWidth
# 
# SepalLength, PetalLength
# 
# SepalLength, PetalWidth
# 

# <span style="color:red">**Question:** </span> Which pairs of attributes are uncorrelated/weakly correlated? 

# <span style="color:green">Answer: </span> 
# Highly correlated pairs listed in decreasing order of correlation.
# 
# SepalLength, SepalWidth
# 

# ### 4. Visualizing relationships between attributes  

# <span style="color:red">**Question:** </span>
# Visualize the iris dataset using a pairplot and comment if PetalLength and PetalWidth have positive covariance. 

# *pairplot* function in seaborn library simultaneously generates histograms for individual attributes and pairwise scatter plots.

# In[23]:


sns.pairplot(iris_df)


# <span style="color:green">Answer: </span> 
# 
# From the above plot, PetalLength and PetalWidth have positive covariance.
# 

# <span style="color:red">**Question:** </span>
# Visualize the iris dataset using a pairplot and comment if the three classes can be separated if SepalLength and SepalWidth are the only variables used. 

# Pair plots allows you to do separate histograms and color scatter plots based on a categorical attribute.

# In[24]:


import seaborn as sns
sns.pairplot(iris_df, hue="Name")


# <span style="color:green">Answer: </span> 
# 
# From the above plot, we can observe that for SepalLength and SepalWidth attributes the orange and green datapoints overlap. Hence, SepalLength and SepalWidth attributes cannot separate the three classes.
# 

# ### 5. Dimensionality Reduction: PCA
# 

# <span style="color:red">**Question:** </span>
# Project points in the digits dataset onto a two-dimensional space using PCA. 

# Steps involved in PCA
# 1. Input data: set of points in $R^d$
# 2. Compute covariance matrix $\Sigma$ (a $d \times d$ matrix)
# 3. Compute Eigenvectors of $\Sigma$
# 4. Select $r$ Eigenvectors (based on a parameter or based on variance explained) corresponding to the highest eigenvalues
# 5. Project data on to the new $r$ dimensional space

# Step 1: Load data

# In[25]:


digits = load_digits()
digits.data.shape


# In[26]:


sns.heatmap(digits.data)


# Plotting one data point as an 8x8 image

# In[27]:


sns.heatmap(np.reshape(digits.data[3,:],[8,8]))


# Step 2: Compute covariance matrix $\Sigma$ (a $d \times d$ matrix)

# In[28]:


digits_cov = np.empty([np.size(digits.data,1), np.size(digits.data,1)]);
for i in range (0, np.size(digits.data,1)):
    for j in range (i, np.size(digits.data,1)):
        digits_cov[i,j] = mycov(digits.data,i,j);
        digits_cov[j,i] = digits_cov[i,j];


# In[29]:


sns.heatmap(digits_cov)


# Step 3: Compute Eigenvectors of $\Sigma$

# In[30]:


w,v = np.linalg.eig(digits_cov)


# In[31]:


w


# In[32]:


np.shape(v)


# In[33]:


np.matmul(v,  np.transpose(v))


# In[34]:


sns.heatmap(v)


# Step 4: Select $r$ Eigenvectors (based on a parameter or based on variance explained) corresponding to the highest eigenvalues

# Variance captured by each of the principal directions

# In[35]:


plt.bar(np.arange(64),w)
plt.xlabel('principal components')
plt.ylabel('variance')


# Cumulative variance captured

# In[36]:


np.cumsum(w)/np.sum(w)


# In[37]:


plt.bar(np.arange(64),np.cumsum(w)/np.sum(w))
plt.xlabel('principal components')
plt.ylabel('Cumulative variance covered')


# Step 5: Project data on to the new $r$ dimensional space

# In[38]:


projected_data = np.matmul(digits.data,v[:,0:2])


# In[39]:


sns.heatmap(projected_data)


# In[40]:


plt.scatter(projected_data[:, 0], projected_data[:, 1],
            c=digits.target, edgecolor='none', alpha=0.8,
            cmap=plt.cm.get_cmap('hsv', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();


# A much simpler way to do PCA using library function from sklearn:

# In[41]:


pca = PCA(2)  # project from 64 to 2 dimensions
projected = pca.fit_transform(digits.data)
print(digits.data.shape)
print(projected.shape)    


# In[42]:


plt.scatter(projected[:, 0], projected[:, 1],
            c=digits.target, edgecolor='none', alpha=0.8,
            cmap=plt.cm.get_cmap('hsv', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();


# ### 6. Singular Value Decomposition

# <span style="color:red">**Question:** </span>
# Compute SVD on the following matrix A using svd() function from scipy library. 

# In[43]:


A = np.array([
    [1, 1, 1, 0, 0],
    [3, 3, 3, 0, 0],
    [4, 4, 4, 0, 0],
    [5, 5, 5, 0, 0],
    [0, 0, 0, 4, 4],
    [0, 0, 0, 5, 5],
    [0, 0, 0, 2, 2]])


# In[44]:


sns.heatmap(A)


# Computing SVD using scipy library function svd()

# In[45]:


U, S, V = svd(A, full_matrices = False)


# In[46]:


sns.heatmap(U)


# In[47]:


sns.heatmap(V)


# Reconstructing the matrix from the factors

# In[48]:


sns.heatmap(np.matmul(np.matmul(U,np.diag(S)), V))


# Visualizing the spectral decomposition from U and V. The first element in spectral decomposition is $\delta_1u_1v_1^T$ is

# In[49]:


sns.heatmap(S[0]*np.outer(U[:,0],V[0,:]))


# The second element $\delta_2u_2v_2^T$ is

# In[50]:


sns.heatmap(S[1]*np.outer(U[:,1],V[1,:]))


# <span style="color:red">**Question:** </span>
# Determine number of spectral values that must be used to capture 90% of the data. 

# In[51]:


S


# First, plotting the variance captured by each spectral value. 

# In[52]:


plt.bar(np.arange(5),S)
plt.xlabel('Components')
plt.ylabel('variance covered')


# Second, plotting the fraction of variance captured by first $r$ spectral values. 

# In[53]:


plt.bar(np.arange(5),np.cumsum(S)/np.sum(S))
plt.xlabel('Components')
plt.ylabel('Cumulative variance covered')


# <span style="color:green">Answer: </span> 
# 
# First two spectral values are needed to capture 90% of the variance.

# ### 7. Linear Discriminant Analysis

# We will use iris data to study LDA. 

# In[54]:


X = iris_df.values[:,0:4]
y = iris_df.values[:,4] 


# We will use the first 100 samples. The first 50 are of the class 'Iris-setosa' and the rest are of the class 'Iris-versicolor'.

# In[55]:


X = X[0:100,:]


# In[56]:


X = X.astype(float)


# In[57]:


y = y[0:100]


# <span style="color:red">**Question:** </span>
# Plot the heatpmap of the data. And determine which attributes can be used for projection so the two classes are well separated. 

# In[58]:


plt.figure(figsize=(20,10))
ax = sns.heatmap(X,cmap='PiYG')
ax.set(xlabel='Attributes', ylabel='Samples')


# <span style="color:green">Answer: </span> 
# 
# The last two attributes are useful to separate the two classes. 

# <span style="color:red">**Question:** </span>
# Using only the first two attributes, project the selected points in X (below) from the iris dataset using LDA. Determine if the two classes are separated despite choosing the first two attributes. Compute the absolute difference between the two means in the projected space. 
# Hint: Use LinearDiscriminantAnalysis function from scikit library. 

# **Steps involved in performing LDA**
# 1. Input data: set of points in $R^d$
# 2. Compute mean vectors $\mu_1$ and $\mu_2$
# 3. Compute between class scatter matrix $S_B$
# 4. Compute within class scatter matrix $S_W$
# 5. Compute the matrix $S_W^{-1}S_B$
# 4. Compute the first eigenvector ($v_1$) of the matrix $S_W^{-1}S_B$
# 5. Project data on to this eigenvector $Xv_1$
# 
# LinearDiscriminantAnalysis function accomplishes all of this.

# In[59]:


lda = LinearDiscriminantAnalysis(n_components=2)
X_r1 = lda.fit(X[:,0:2], y).transform(X[:,0:2])


# In[60]:


plt.figure(figsize=(20,10))
ax = sns.heatmap(X_r1,cmap='PiYG')
ax.set(xlabel='Attributes', ylabel='Samples')


# In[61]:


fig = sns.scatterplot(x=np.arange(np.size(X_r1)),y=X_r1[:,0],hue=y)
plt.ylabel('LDA projection axis')
plt.show(fig)


# Computing the absolute difference between the means in the projected space.

# In[62]:


abs(np.mean(X_r1[y=='Iris-setosa',0]) - np.mean(X_r1[y=='Iris-versicolor',0]))

