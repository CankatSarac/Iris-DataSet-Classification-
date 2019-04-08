#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Kütüphaneleri yüklüyorum
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[16]:


# Ödevde verilen İris Datasetini yükledim
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)


# In[3]:


#Data set hakkında kafamda düşünce olması adına kaç tane row ve column olduğunu öğrenmek istiyorum
print(dataset.shape)


# In[4]:


# head komutunu kullanarak en üstten 20 tane veriyi görmek istiyorum
print(dataset.head(20))


# In[5]:


# Data set hakkında temel bilgileri öğreniyorum describe komutu ile count, mean, min ve max değerlerini görüyorum.
print(dataset.describe())


# In[6]:


# class distribution yapıyorum 150 olan rowu classlara böldüm
print(dataset.groupby('class').size())


# In[7]:


# descriptionsdan gördüğüm 4 coloumn değer için box ve whisker plots grafiği uyguladım.
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# In[19]:


# histogram olusturdum
dataset.hist()
plt.show()


# In[17]:


# scatter plot matrix oluşturdum
scatter_matrix(dataset)
plt.show()


# In[10]:


# Validation dataseti Ayırdım 80% ve 20% olacak sekilde train ve test set olmak uzere
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# In[11]:


#  10-fold cross validation kullandim 10 parcaya ayirdim, train 9 ve test 1
seed = 7
scoring = 'accuracy'


# In[12]:


#Logistic Regression (LR) soruda istenen 6 farkli algoritma kullaniyorum
#Linear Discriminant Analysis (LDA)
#K-Nearest Neighbors (KNN).
#Classification and Regression Trees (CART).
#Gaussian Naive Bayes (NB).
#Support Vector Machines (SVM).
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# In[14]:


# Yukarida belirttigim Algoritmalari karsilastiriyorum
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[20]:



# Tahmin yapiyorum
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[ ]:




