import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


# df = pd.read_csv('magic04.data') # this line prints the data without any labels (attribute names)
cols = ['fLength', ' fWidth', 'fSize', 'fConc', 'fConc1', 'fAssym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
df = pd.read_csv('Magic04.data', names=cols) 
# print(df.head())  # prints first 5 rows with column names

# print(df['class'].unique()) # prints: ['g' 'h'] for gamma and hadrons

# convert g and h into int (0, 1)

df["class"] = (df['class'] == 'g').astype(int) # if the dataframe is equal to g then it is assigned 1. else 0
# print(df.head()) #prints class in 0 and 1s

for label in cols[:-1]:
    plt.hist(df[df['class'] ==1][label], color = 'blue', label='gamma', alpha=0.7, density = True) #plots a histogram blue in colot with opacity0.7 (denisty, )
    plt.hist(df[df['class'] ==0][label], color = 'red', label='hadron', alpha=0.7, density = True)
    plt.title(label)
    plt.ylabel('Probability')
    plt.xlabel(label)
    plt.legend
    # plt.show()

# the line df[df['class'] ==1][label] gets all the values for a particular label from the gamma class (since it is equal to one)\


# train, validation and test data sets

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
# 80% for training and validation and the rest for testing

def scale_dataset(dataframe, oversample = False):
    '''Scales all the X values (labels) to make them easier to plot on a 2-d graph and compare 
    
    '''
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample: 
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y) 

    data = np.hstack((X, np.reshape(y, (-1, 1)))) #hstack is horizontal stack --> takes 2 arrays and horizaontally stacks them together
    # X is a 2-D object, however y is just 1-D so, we reshape y using np.reshape

    return data, X, y

# print(len(train[train['class'] == 1])) #gamma = 7400
# print(len(train[train['class'] == 0])) #hadron = 4017
# since there is way less datapoints for hadrons, we oversample the hadrons so that the values match better

train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)

# print(sum(y_train == 0) )
# print(sum(y_train == 1) ) # prints the same value 7383

#KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors=37)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

# print(classification_report(y_test, y_pred))


#Naive Bayes method 

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)
# print(classification_report(y_test, y_pred)) 
#the accuracy using this method is .72 which is quite less when compared to .84
# so wont be using this method 

# Log Regression

# we can play with the type of graph (excel vibes)
from sklearn.linear_model import LogisticRegression

lg_model = LogisticRegression()
lg_model = lg_model.fit(X_train, y_train)
y_pred = lg_model.predict(X_test)
# print(classification_report(y_test, y_pred)) # accuracy is .79


# SVM
from sklearn.svm import SVC

svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred)) # accuracy is .79

#Accuracy is: 0.86 which is also more than KNN

# Neural Networks

import tensorflow as tf

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot (history. history['loss'], label='loss' )
    ax1.plot(history.history['val_loss'], label = 'val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary crossentropy' )
    ax1.grid (True)

    ax2.plot (history. history['accuracy'], label='accuracy' )
    ax2.plot(history.history['val_accuracy'], label = 'val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('accuracy')
    ax2.grid (True)


    plt.show()



#epoch = training cycles

def train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation='sigmoid'),#projecting our prediction to be 0 or 1

    ])


    nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])

    history = nn_model.fit(
    X_train, y_train, epochs = epochs, batch_size = batch_size, validation_split=0.2, 
    )

    return nn_model, history


least_val_loss = float('inf')
least_loss_model = None
epochs = 100
for num_nodes in [64]:
    for dropout_prob in [0.2]:
        for lr in [0.001]:
            for batch_size in [64]:
                print(f"{num_nodes} nodes, dropout_prob{dropout_prob}, lr {lr}, batch size {batch_size}")
                model, history = train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs)
                # plot_history(history)
                val_loss = model.evaluate(X_valid, y_valid)[0]
                if val_loss < least_val_loss:
                    least_val_loss = val_loss
                    least_loss_model = model


y_pred = least_loss_model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int).reshape(-1,)
print(y_pred)
print(classification_report(y_test, y_pred))

