import numpy as np
from keras import optimizers
from keras import metrics
from keras import losses
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scipy.io.arff import loadarff
from sklearn import preprocessing
import matplotlib.pyplot as plt
METRICS = [
metrics.FalsePositives(name='fp'),
metrics.FalseNegatives(name='fn'),
metrics.TruePositives(name='tp'),
metrics.TrueNegatives(name='tn'),
metrics.BinaryAccuracy(name='accuracy'),
]
mean_squared_error = losses.squared_hinge

KDDTrain, train_metadata = loadarff("KDDTrain+.arff")
KDDTest, test_metadata = loadarff("KDDTrain+_20Percent.arff")
training_nparray = np.asarray(KDDTrain.tolist()) 
testing_nparray = np.asarray(KDDTest.tolist())

enc = preprocessing.OrdinalEncoder()

encoded_dataset = enc.fit_transform(training_nparray) 
X_train = encoded_dataset[:, :-1] 
y_train = np.ravel(encoded_dataset[:, -1:]) 
encoded_dataset = enc.fit_transform(testing_nparray)
X_test = encoded_dataset[:, :-1]
y_test = np.ravel(encoded_dataset[:, -1:])

model = Sequential()
model.add(Dense(1024, input_dim=41, activation='relu'))
model.add(Dense(768, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
optimizer=optimizers.SGD(learning_rate = 0.001,momentum = 0.8), 
metrics=METRICS)

history = model.fit(X_train, y_train,
epochs=40,
batch_size=128)
score = model.evaluate(X_test, y_test, batch_size=128)

plt.plot(history.history['accuracy'])

plt.title('model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

total_datapoints = X_test.shape[0]
percent_correct = score[5] * 100
correct_datapoints = int(round(total_datapoints * percent_correct) / 100)
mislabeled_datapoints = total_datapoints - correct_datapoints

print(score[1])
print(score[2])
print(score[3])
print(score[4])
print("results:\n")
print("Total datapoints: %d\nCorrect datapoints: %d\nMislabeled datapoints: %d\nPercent correct: %.2f%%"
% (total_datapoints, correct_datapoints, mislabeled_datapoints, percent_correct))
