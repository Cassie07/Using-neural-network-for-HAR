import numpy as np
from numpy import float32,int32
np.random.seed(42)
import tensorflow as tf
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

tf.set_random_seed(42)
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
from keras import backend as K
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout,Flatten
from data import load_data
from utils import confusion_matrix
from keras.callbacks import TensorBoard, ModelCheckpoint
import os
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

LABELS = ['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING']
lables=np.array(LABELS)
print(lables.shape)
print(lables)
CHECK_ROOT = 'checkpoint/'
if not os.path.exists(CHECK_ROOT):
    os.makedirs(CHECK_ROOT)
epochs = 20 # 30
batch_size = 16
n_hidden = 32

def one_hot(y_):
    """
    Function to encode output labels from number indexes.
    E.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    """
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

def _count_classes(y):
    return len(set([tuple(category) for category in y]))

X_train, X_test, Y_train, Y_test = load_data()
y_test=Y_test.argmax(1)

timesteps = len(X_train[0])
input_dim = len(X_train[0][0])
n_classes = _count_classes(Y_train)
print("n_classes",n_classes)

# LSTM
#model = Sequential()
#model.add(LSTM(n_hidden, input_shape=(timesteps, input_dim)))
#model.add(LSTM(n_hidden, input_shape=(timesteps, input_dim)))
#model.add(Dropout(0.5))
#model.add(Dense(n_classes, activation='sigmoid'))
#model.compile(loss='categorical_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])

# bidLSTM
model = Sequential()
model.add(Bidirectional(LSTM(n_hidden, return_sequences=True), input_shape=(timesteps, input_dim)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(n_classes, activation='sigmoid')) #n_classes
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# callback: draw curve on TensorBoard
tensorboard = TensorBoard(log_dir='log', histogram_freq=0, write_graph=True, write_images=True)
# callback: save the weight with the highest validation accuracy
filepath=os.path.join(CHECK_ROOT, 'weights-improvement-{val_acc:.4f}-{epoch:04d}.hdf5')
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
model.fit(X_train,
          Y_train,
          batch_size=batch_size,
          validation_data=(X_test, Y_test),
          epochs=epochs,callbacks=[tensorboard, checkpoint])

# Evaluate
print(confusion_matrix(Y_test, model.predict(X_test)))
predict=model.predict(X_test)
print("predict#################",predict)
pred_index_total=[]
for pred in predict:
    pred_index = []
    pred_list=pred.tolist()
    index_max=pred_list.index(max(pred_list))
    pred_index.append(index_max)
    pred_index_total.append(np.array(pred_index))
print(pred_index_total)
one_hot_predictions=one_hot(np.array(pred_index_total))
print("one_hot_predictions%%%%%%%%%",one_hot_predictions)
prediction=one_hot_predictions.argmax(1)
confusion_matrix = metrics.confusion_matrix(y_test, prediction)
print("%%%%%%%%%%%%%%%",confusion_matrix)

# Plot Results:
width = 12
height = 12
normalised_confusion_matrix = np.array(confusion_matrix, dtype=float32)/np.sum(confusion_matrix)*100
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix,
    interpolation='nearest',
    cmap=plt.cm.rainbow
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks,lables,rotation=90)
plt.yticks(tick_marks,lables)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()