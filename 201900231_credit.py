import pandas as pd 
import numpy as np
# from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn import svm
# import itertools
import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab
# from sklearn.model_selection import train_test_split
import seaborn
from sklearn.metrics import accuracy_score
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot
from tensorflow.keras import layers
import keras
# from tensorflow.keras.utils import to_categorical
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

df = pd.read_csv(r'D:\My lectures\level 3\sel 1\dataset\creditcard.csv') # Reading the file .csv
df = pd.DataFrame(df)

# A Piece Of Code To Display All Columns In Console
pd.set_option('display.max_columns', None)

df.head()

df["Class"].unique()
#one hot encoding
df=pd.concat([df, pd.get_dummies(df['Class'],prefix='Class',drop_first=True)],axis=1)
df.drop(['Class'],axis=1,inplace= True)

# normailization  (Amount)
df['Amount'] = df['Amount'] /df['Amount'].abs().max()

# drob useless coulmns
df.drop(['id'],axis=1,inplace= True)

df.info()

# We seperate ours data in two groups : a train dataset and a test dataset
'''
Y_col = 'Class_True'
X_cols = df.loc[:, df.columns != Y_col].columns
train_x, test_x, train_y, test_y = train_test_split(df[X_cols], df[Y_col],test_size=0.1)
'''
df_one=df[df.Class_True == 1]
df_zero=df[df.Class_True == 0]
df_sample=df_zero.sample(300)
df_train = df_one.append(df_sample) # We gather the frauds with the no frauds. 
df_train = df_train.sample(frac=1) # Then we mix our dataset

train_x = df_train.drop(['Time', 'Class_True'],axis=1) # We drop the features Time (useless), and the Class (label)
train_y = df_train['Class_True'] # We create our label
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
test_x = df.drop(['Time', 'Class_True'],axis=1)
test_y = df['Class_True']
test_x = np.asarray(test_x)
test_y = np.asarray(test_y)


# SVM Model
classifier = svm.SVC(kernel='rbf')
classifier.fit(train_x, train_y)
prediction_SVM = classifier.predict(test_x)
accuracy_score(test_y, prediction_SVM)
## confusion_matrix for svm
confusion_matrix= confusion_matrix(test_y, prediction_SVM)
ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
plt.show()
## roc curve for svm
roc_curve(test_y,prediction_SVM)
ns_fpr, ns_tpr, _ = roc_curve(test_y, prediction_SVM)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Roc curve for svm')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

## learning curve
def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    
    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes.legend(loc="best") 
    
    return plt

fig, axes = plt.subplots(1 , 1,figsize=(10, 10))

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

title = "Learning Curves "

plot_learning_curve(
   classifier , title, train_x,train_y,axes=axes, ylim=(0.7, 1.01), cv=cv, n_jobs=4
)

plt.show()

#ANN model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(train_x.shape[1],)),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(2,activation='sigmoid'),
])
# # Configure a model for mean-squared error regression.
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics = ['accuracy'])


# y_test_all_cat=to_categorical(y_test_all, num_classes=2)
checkpoint = keras.callbacks.ModelCheckpoint(filepath='D:\My lectures\level 3\gerges hana\Level 3\Part 1\Selected 1\Section\checkpoint.h5', verbose=1, save_best_only=True)
history=model.fit(train_x, train_y, batch_size=100, epochs=10,callbacks=[checkpoint],validation_split=0.1)
y_predict=model.predict(test_x).argmax(axis=1)


#loss curve for ann
print("Loss Curve :")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
print(history.history.keys())

## roc curve for ann
roc_curve(test_y,y_predict)
ns_fpr, ns_tpr, _ = roc_curve(test_y, y_predict)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Roc curve for ann')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

## confusion_matrix for svm
import sklearn.metrics as metrics
confusion_matrix= metrics.confusion_matrix(test_y, y_predict)
ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
ax.set_title('Seaborn Confusion Matrix for Ann \n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
plt.show()
