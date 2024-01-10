#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])

# Split the data into features (X) and target variable (y)
X = data.drop('target', axis=1)
y = data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Choose and train a machine learning model (Logistic Regression in this case)
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Display the evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns

# Pairplot for visualizing relationships between features
sns.pairplot(data, hue='target', palette='viridis')
plt.show()

# Box plot for each feature
plt.figure(figsize=(15, 8))
for i, feature in enumerate(iris['feature_names']):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='target', y=feature, data=data, palette='viridis')
    plt.title(f'{feature} distribution by target')
plt.tight_layout()
plt.show()


# In[3]:


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])

# Split the data into features (X) and target variable (y)
X = data.drop('target', axis=1)
y = data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Choose and train the Support Vector Machine model
svm_model = SVC(kernel='linear', C=1)  # You can choose different kernels and adjust parameters
svm_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_svm = svm_model.predict(X_test)

# Evaluate the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')

# Display the evaluation metrics for SVM
print("Support Vector Machine (SVM) Metrics:")
print(f"Accuracy: {accuracy_svm:.2f}")
print(f"Precision: {precision_svm:.2f}")
print(f"Recall: {recall_svm:.2f}")


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns

# Pairplot for visualizing relationships between features
sns.pairplot(data, hue='target', palette='viridis')
plt.show()

# Box plot for each feature
plt.figure(figsize=(15, 8))
for i, feature in enumerate(iris['feature_names']):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='target', y=feature, data=data, palette='viridis')
    plt.title(f'{feature} distribution by target')
plt.tight_layout()
plt.show()


# In[5]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


# In[2]:


iris = datasets.load_iris()
X = iris.data
y = iris.target


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm = SVC()


# In[4]:


param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_
best_svm = SVC(C=best_params['C'], kernel=best_params['kernel'])
best_svm.fit(X_train_scaled, y_train)
y_pred = best_svm.predict(X_test_scaled)


# In[5]:


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


# In[6]:


print("Best Parameters:", best_params)
print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)


# In[ ]:





# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns

# Pairplot for visualizing relationships between features
sns.pairplot(data, hue='target', palette='viridis')
plt.show()

# Box plot for each feature
plt.figure(figsize=(15, 8))
for i, feature in enumerate(iris['feature_names']):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='target', y=feature, data=data, palette='viridis')
    plt.title(f'{feature} distribution by target')
plt.tight_layout()
plt.show()


# In[ ]:




