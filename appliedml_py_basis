# ## Machine Learning Basics: Why Python?

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) #ignora i warning di sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()

#What is ML => Is basically fitting a function to examples and using that function to generalize and make predictions about new examples
model = LogisticRegression(max_iter = 10000).fit(iris["data"], iris["target"]) #fit is a method contained in this logistic regression model
#data is the training data. We have a fit model stored as model and with this we can make predictions.
model.predict(iris["data"]) #use this model to make predictions using the features

'''
Some definitions: 
deeplearning = this is fitting functions to examples where those functions are connected layers of nodes; 
with the intent to generalize and make predictions about new examples.
Ai = 1) weak Ai is an intelligence specifically designed to focus on a narrow task
   = 2) strong/General AI is a machine with consciousness, sentience, and a mind; general intelligence capable of any and all cognitive functions and reasoning that a human
   is capable of
'''
#Let's do some Exploratory Data Analysis
#Why? Because you need to understand the shape of the data, learn which features might be useful and than use this info to set the cleaning that will come next
#What? Getting counts or distributions of all variables to understand the shape, look at the data type for each feature, check missing data, understand correl and maybe identify duplicates
#Data cleaning=> focus on shape data in a way a model can best pick up on the signal, remove irrelevant data and adjust features to be acceptable for a model. Also is important to anonymize and encode categorical variables, fill miss. data and prune/scale data to account for skewed data/outliers.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

titanic = pd.read_csv('../../../titanic.csv')
titanic.head()

# Drop all categorical features (eliminale)
cat_feat = ['PassengerId', 'Name', 'Ticket', 'Sex', 'Cabin', 'Embarked']
titanic.drop(cat_feat, axis =1, inplace= True) #inplace fa modificare il dataframe direttamente
titanic.head() #only continuous features


# ### Explore continuous features

titanic.describe()
#mean = 0.38 is import if you need to classify
#n.b: age has missing values

#which feature is a strong indicator to indicate if a passenger survived or not?
#a good way is to groupby the two levels of Survived (0,1) and generate average value of other features at those two levels.
titanic.groupby("Survived").mean() #ci divide il db per i valori di survived e ci torna l'average value per ogni feature

#Now we want to check if the missing values where somehow systematic or random
titanic.groupby(titanic["Age"].isnull()).mean() #is null mette true or false
#we can see that people with no age info were less likely to survive, had higher class nr, fewer parents or children and a lower fare


# ### Plot continuous features

for i in ['Age', 'Fare']: #continuous variable
    died = list(titanic[titanic['Survived'] == 0][i].dropna()) #this will grab non missing values and assigne them to two lists
    survived = list(titanic[titanic['Survived'] == 1][i].dropna())
    xmin = min(min(died), min(survived))
    xmax = max(max(died), max(survived))
    width = (xmax - xmin) / 40
    sns.distplot(died, color='r', kde=False, bins=np.arange(xmin, xmax, width))
    sns.distplot(survived, color='g', kde=False, bins=np.arange(xmin, xmax, width))
    plt.legend(['Did not survive', 'Survived'])
    plt.title('Overlaid histogram for {}'.format(i))
    plt.show()

#I create a loop because I want to do this for all three features
dataz = ["Pclass", "SibSp", "Parch"]
for i in range(len(dataz)):
    plt.figure(i)
    sns.catplot(x=dataz[i], y="Survived", data=titanic, kind="point", aspect=2,) #we plot the survived rate for each level of this feature
#aspect control the size

titanic["family_cnt"] = titanic["SibSp"] + titanic["Parch"]
sns.catplot(x="family_cnt", y="Survived", data=titanic, kind="point", aspect=2,)

#the more people in your family, the less likely you're to survive


# Drop all categorical features
cat_feat = ['PassengerId', 'Name', 'Ticket', 'Sex', 'Cabin', 'Embarked']
titanic.drop(cat_feat, axis=1, inplace=True)
titanic.head()

# Drop irrelevant continuous variable
titanic.drop('PassengerId', axis=1, inplace=True)
titanic.head()


# ### Fill missing for `Age`

titanic.groupby(titanic['Age'].isnull()).mean()

#with the mean value
titanic["Age"].fillna(titanic["Age"].mean(), inplace = True)
titanic.isnull().sum() #to check that is zero
titanic.head()


# ### Combine `SibSp` & `Parch`

#Whenever we can mantain the relation and the info we should combine some variables to clear the picture
titanic["Family_cnt"] = titanic["SibSp"] + titanic["Parch"]
#then you have to clean the previous info in order to not be redundant
titanic.drop(["SibSp", "Parch"], axis=1, inplace = True)
titanic.head()

# Drop all continuous features
cont_feat = ['PassengerId', 'Pclass', 'Name', 'Age', 'SibSp', 'Parch', 'Fare']
titanic.drop(cont_feat, axis=1, inplace=True)
titanic.head()


# ### Explore categorical features
#
# Explore `Sex`, `Cabin`, and `Embarked`.
titanic.info()
#as you can see there are missing values for cabin
titanic.groupby(titanic['Cabin'].isnull()).mean() #dramatic split where over 60% of the people who have non missing values survived, while less than 30% of the one with no missing values survived
titanic['Cabin_ind'] = np.where(titanic['Cabin'].isnull(), 0, 1) #dà a cabin valori 0 e 1
titanic.head(10)


# ### Plot categorical features
for i, col in enumerate(['Cabin_ind', 'Sex', 'Embarked']):
    plt.figure(i)
    sns.catplot(x=col, y='Survived', data=titanic, kind='point', aspect=2, )

#pivot tables are good for exploring the relationship between multiple variables
titanic.pivot_table('Survived', index='Sex', columns='Embarked', aggfunc='count')
#so here you understand that in Southampton the majority who sailed where men, the ones with lower possibility to survive
titanic.pivot_table('Survived', index='Cabin_ind', columns='Embarked', aggfunc='count')
#also here we know that a lot of people of South. have no room


# ### Split into train, validation, and test set

#Per creare il training set devi dividere tra features (by dropping the survived field) e
features = titanic.drop('Survived', axis=1)
labels = titanic['Survived'] #e assegni quelli droppati a labels (is the format required for split method)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.4, random_state= 42)
#perché 0.4? Il motivo è che questo metodo non riesce a dividere le features e le labels in tre, ma solo in due (60% test e 40% test)
#e in un secondo step prendiamo questo 40% e lo dividiamo a metà (metà validation e metà test). come qui sotto.
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size = 0.5, random_state= 42)

#to check that everything is ok
print(len(labels), len(y_train), len(y_val),len(y_test))

#holdout test set=> generalization of the test set => sample of data not used in fitting a model, and used to evaluate the model's ability to generalize to unseen data
#K-Fold Cross-Validation => data is divided into k subsets and the holdout method is repeated k times.
#each time, one of the k subsets is used as the test set and the other k-1 subsets are combined to used to train the model. Ogni volta va a valutare diversamente le predizioni per ogni k.
#Alla fine ottieni un array of scores e puoi o utilizzarlo tutto o prendere la media.
#Il nostro k lo mettiamo a 5, quindi fivefold cross-validation

#Two components of evaluation framework => 1) Evaluation metrics - how are we gauging the accuracy of the model?
# 2) Process => how to split the data => how do we leverage our full dataset to mitigate likelikood of overfitting or underfitting.

#1) In our case is a classification model (survived or not). So here we'll use accuracy = # predicted correctly / total # of examples
#we'll also use Precision => # predicted as surviving that actually survived / total # predicted to survive
#and lastly we'll focus on recall => #predicted as survivng that actually survived /total # that actually survived


#2) Process: - run fivefold cross validation and select best models, then re-fit models on full training set, evaluate those models on the validation set and pick the best one.
#Lastly, we'll evalyated the best model on test set to gauge its ability to generalize to unseen data

#Total Error = Bias + Variance + Irreducible Error (we only control the first twos)
#A good model has a somehow medium complexity, low bias and low variance. In other word the train error is low but also the test error is low.

#The are actually two primary ways to tune model for optimal complexity =>
#a) Hyperparameter tuning - choosing a set of optimal hyperparameters for fitting an algorithm
#b) Regularization - technique used to reduce overfitting by discouraging overly complex models in some way

#a) An hyperparameter is a configuration that is external to the model, whose value cannot be estimated from data, and whose value guides how the algorithm learns parameter values from the data.
#ex: max depth of tree, features to consider, etc.

#b) related to Occam's razor (-whenever possible, choose the simplest answer to a problem")
# A first example is the Ridge regression and lasso regression, that's adding a penalty to the loss function to constrain coefficients, while the second one is Dropout, where
#some nodes are ignored during training which forces the other nodes to take on more or less responsibility for the input/output.



# #### Combine `SibSp` & `Parch`

for i, col in enumerate(['SibSp', 'Parch']):
    plt.figure(i)
    sns.catplot(x=col, y='Survived', data=titanic, kind='point', aspect=2, )

titanic['Family_cnt'] = titanic['SibSp'] + titanic['Parch']


# #### Drop unnnecessary variables

titanic.drop(['PassengerId', 'SibSp', 'Parch'], axis=1, inplace=True)
titanic.head(10)


# ### Write out cleaned data
titanic.to_csv('../../../titanic_cleaned.csv', index=False)

# ### Clean categorical variables
# 1. Create an indicator variable in place of `Cabin`
# 2. Convert `Sex` to a numeric variable
# 3. Drop irrelevant/repetitive variables (`Cabin`, `Embarked`, `Name`, `Ticket`)

# #### Create indicator for `Cabin`

titanic['Cabin_ind'] = np.where()

# #### Convert `Sex` to numeric
gender_num = {'male': 0, 'female': 1}

# #### Drop unnecessary variables
titanic.drop(['Cabin', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

# ### Write out cleaned data
titanic.to_csv('../../../titanic_cleaned.csv', index=False)


# ## Pipeline: Split data into train, validation, and test set
#
# Using the Titanic dataset from [this](https://www.kaggle.com/c/titanic/overview) Kaggle competition.
#
# In this section, we will split the data into train, validation, and test set in preparation for fitting a basic model in the next section.

# ### Read in Data

from sklearn.model_selection import train_test_split

titanic = pd.read_csv('../../../titanic_cleaned.csv')

# ### Split into train, validation, and test set
#

features = titanic.drop('Survived', axis=1)
labels = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

for dataset in [y_train, y_val, y_test]:
    print(round(len(dataset) / len(labels), 2))
#check if you split right

# ### Write out data

X_train.to_csv('../../../train_features.csv', index=False)
X_val.to_csv('../../../val_features.csv', index=False)
X_test.to_csv('../../../test_features.csv', index=False)

y_train.to_csv('../../../train_labels.csv', index=False)
y_val.to_csv('../../../val_labels.csv', index=False)
y_test.to_csv('../../../test_labels.csv', index=False)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

tr_features = pd.read_csv('../../../train_features.csv')
tr_labels = pd.read_csv('../../../train_labels.csv', header=None)


# ### Fit and evaluate a basic model using 5-fold Cross-Validation
rf = RandomForestClassifier()
scores = cross_val_score(rf, tr_features, tr_labels.values.ravel(), cv=5)
scores
#array([0.82407407, 0.85046729, 0.79439252, 0.82075472, 0.82075472])


## Pipeline: Tune hyperparameters
# In this section, we will tune the hyperparameters for the basic model we fit in the last section.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

tr_features = pd.read_csv('../../../train_features.csv')
tr_labels = pd.read_csv('../../../train_labels.csv', header=None)


# ### Hyperparameter tuning
def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))
    
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))

rf = RandomForestClassifier()
parameters = {
    'n_estimators': [5, 50, 100],
    'max_depth': [2, 10, 20, None]
}

cv = GridSearchCV(rf, parameters, cv=5)
cv.fit(tr_features, tr_labels.values.ravel())

print_results(cv)

'''
BEST PARAMS: {'max_depth': 10, 'n_estimators': 5}

0.76 (+/-0.116) for {'max_depth': 2, 'n_estimators': 5}
0.796 (+/-0.119) for {'max_depth': 2, 'n_estimators': 50}
0.803 (+/-0.117) for {'max_depth': 2, 'n_estimators': 100}
0.828 (+/-0.074) for {'max_depth': 10, 'n_estimators': 5}
0.816 (+/-0.028) for {'max_depth': 10, 'n_estimators': 50}
0.826 (+/-0.046) for {'max_depth': 10, 'n_estimators': 100}
0.785 (+/-0.106) for {'max_depth': 20, 'n_estimators': 5}
0.813 (+/-0.027) for {'max_depth': 20, 'n_estimators': 50}
0.809 (+/-0.029) for {'max_depth': 20, 'n_estimators': 100}
0.794 (+/-0.04) for {'max_depth': None, 'n_estimators': 5}
0.809 (+/-0.037) for {'max_depth': None, 'n_estimators': 50}
0.818 (+/-0.035) for {'max_depth': None, 'n_estimators': 100}
'''

# ## Pipeline: Evaluate results on validation set

# In this section, we will use what we learned in last section to fit the best few models on the full training set and then evaluate the model on the validation set.

from sklearn.metrics import accuracy_score, precision_score, recall_score

tr_features = pd.read_csv('../../../train_features.csv')
tr_labels = pd.read_csv('../../../train_labels.csv', header=None)

val_features = pd.read_csv('../../../val_features.csv')
val_labels = pd.read_csv('../../../val_labels.csv', header=None)

te_features = pd.read_csv('../../../test_features.csv')
te_labels = pd.read_csv('../../../test_labels.csv', header=None)


# ### Fit best models on full training set
#
# Results from last section:
# ```
# 0.76 (+/-0.116) for {'max_depth': 2, 'n_estimators': 5}
# 0.796 (+/-0.119) for {'max_depth': 2, 'n_estimators': 50}
# 0.803 (+/-0.117) for {'max_depth': 2, 'n_estimators': 100}
# --> 0.828 (+/-0.074) for {'max_depth': 10, 'n_estimators': 5}
# 0.816 (+/-0.028) for {'max_depth': 10, 'n_estimators': 50}
# --> 0.826 (+/-0.046) for {'max_depth': 10, 'n_estimators': 100}
# 0.785 (+/-0.106) for {'max_depth': 20, 'n_estimators': 5}
# 0.813 (+/-0.027) for {'max_depth': 20, 'n_estimators': 50}
# 0.809 (+/-0.029) for {'max_depth': 20, 'n_estimators': 100}
# 0.794 (+/-0.04) for {'max_depth': None, 'n_estimators': 5}
# 0.809 (+/-0.037) for {'max_depth': None, 'n_estimators': 50}
# --> 0.818 (+/-0.035) for {'max_depth': None, 'n_estimators': 100}
# ```

rf1 = RandomForestClassifier(n_estimators=5, max_depth=10)
rf1.fit(tr_features, tr_labels.values.ravel())

rf2 = RandomForestClassifier(n_estimators=100, max_depth=10)
rf2.fit(tr_features, tr_labels.values.ravel())

rf3 = RandomForestClassifier(n_estimators=100, max_depth=None)
rf3.fit(tr_features, tr_labels.values.ravel())

'''
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    max_depth=None, max_features='auto', max_leaf_nodes=None,
    min_impurity_decrease=0.0, min_impurity_split=None,
    min_samples_leaf=1, min_samples_split=2,
    min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
    oob_score=False, random_state=None, verbose=0,
    warm_start=False)
'''
# ### Evaluate models on validation set

for mdl in [rf1, rf2, rf3]:
    y_pred = mdl.predict(val_features)
    accuracy = round(accuracy_score(val_labels, y_pred), 3)
    precision = round(precision_score(val_labels, y_pred), 3)
    recall = round(recall_score(val_labels, y_pred), 3)
    print('MAX DEPTH: {} / # OF EST: {} -- A: {} / P: {} / R: {}'.format(mdl.max_depth,
                                                                         mdl.n_estimators,
                                                                         accuracy,
                                                                         precision,
                                                                         recall))

'''
    MAX DEPTH: 10 / # OF EST: 5 -- A: 0.799 / P: 0.778 / R: 0.737
    MAX DEPTH: 10 / # OF EST: 100 -- A: 0.832 / P: 0.859 / R: 0.724
    MAX DEPTH: None / # OF EST: 100 -- A: 0.793 / P: 0.791 / R: 0.697
'''


# ### Evaluate the best model on the test set


y_pred = rf2.predict(te_features)
accuracy = round(accuracy_score(te_labels, y_pred), 3)
precision = round(precision_score(te_labels, y_pred), 3)
recall = round(recall_score(te_labels, y_pred), 3)
print('MAX DEPTH: {} / # OF EST: {} -- A: {} / P: {} / R: {}'.format(rf2.max_depth,
                                                                     rf2.n_estimators,
                                                                     accuracy,
                                                                     precision,
                                                                     recall))


#MAX DEPTH: 10 / # OF EST: 100 -- A: 0.792 / P: 0.75 / R: 0.646
