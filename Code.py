import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset
fashionTrain = pd.read_csv('fashion-mnist_train.csv')
fashionTest = pd.read_csv('fashion-mnist_test.csv')

train = np.array(fashionTrain, dtype = 'float32')
test = np.array(fashionTest, dtype = 'float32')

# Visualize an Image
plt.imshow(train[6900, 1:].reshape(28, 28))

# To show all images
w = 15
l = 15
fig, axes = plt.subplots(l, w, figsize = (17, 17))
axes = axes.ravel() #Flaten
for i in np.arange(0, w*l):
    index = np.random.randint(0, 60000)
    axes[i].imshow(train[index, 1:].reshape((28, 28)))
    axes[i].set_title(train[index, 0:], fontsize = 8)
    axes[i].axis('off')
plt.subplots_adjust(hspace = 0.4)

# Training
xTrain = train[:, 1:]/255
yTrain = train[:, 0]
xTest = test[:, 1:]/255
yTest = test[:, 0]

# Validation Dataset
from sklearn.model_selection import train_test_split
xTrain, xValidate, yTrain, yValidate = train_test_split(xTrain, yTrain,
                                        test_size = 0.2, random_state 
                                         = 12345)

xTrain = xTrain.reshape(xTrain.shape[0], *(28, 28, 1))
xTest = xTest.reshape(xTest.shape[0], *(28, 28, 1))
xValidate = xValidate.reshape(xValidate.shape[0], *(28, 28, 1)) 

# Creating a CNN
import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

cnn = Sequential()
cnn.add(Conv2D(32, 3, 3, input_shape = (28, 28, 1), activation = 'relu'))
cnn.add(MaxPooling2D(poo_size = (2, 2)))
cnn.add(Flatten())
cnn.add(Dense(output_dim = 32, activate = 'relu'))
cnn.add(Dense(output_dim = 10, activate = 'sigmoid'))
cnn.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(lr = 
                0.01), metrics = ['accuracy'])
epochs = 50
cnn.fit(xTrain,
        yTrain,
        batch_size = 512,
        verbose = 1,
        validation_data = (xValidate, yValidate))

eval = cnn.evaluate(xTest, yTest)
print('Test Accuracy : {:.3f}'.format(eval[1]))
