import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import os
import random

SEED = 42
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # TF 2.1
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

train_df = pd.read_csv("train.tsv", sep='\t', header=None, usecols=[0, 1], names=["Text", "Emotion"])
test_df = pd.read_csv("test.tsv", sep='\t', header=None, usecols=[0, 1], names=["Text", "Emotion"])
val_df = pd.read_csv("dev.tsv", sep='\t', header=None, usecols=[0, 1], names=["Text", "Emotion"])

with open('emotions.txt') as f:
  emotion_list = f.read().split("\n")

def get_emotion(emotion_label):
    emotion_label = emotion_label.split(",")[0]
    return emotion_label

train_df["Emotion"] = train_df["Emotion"].apply(get_emotion)
test_df["Emotion"] = test_df["Emotion"].apply(get_emotion)
val_df["Emotion"] = val_df["Emotion"].apply(get_emotion)

# Remove Duplicates with the same texts
train_df = train_df.drop_duplicates("Text")
test_df = test_df.drop_duplicates("Text")
val_df = val_df.drop_duplicates("Text")

# convert to in
train_df['Emotion'] = train_df['Emotion'].astype('int')
test_df['Emotion'] = test_df['Emotion'].astype('int')
val_df['Emotion'] = val_df['Emotion'].astype('int')

# Convert Emotion into One-Hot Enocded Data
oh = preprocessing.OneHotEncoder()
y_train = oh.fit_transform(train_df["Emotion"].values.reshape(-1,1))
y_test = oh.transform(test_df["Emotion"].values.reshape(-1,1))
y_val = oh.transform(val_df["Emotion"].values.reshape(-1,1))

# Convert sparse matrices to tensors so that it can be used for model training
y_train = tf.convert_to_tensor(csr_matrix(y_train).todense(), tf.float32)
y_test = tf.convert_to_tensor(csr_matrix(y_test).todense(), tf.float32)
y_val = tf.convert_to_tensor(csr_matrix(y_val).todense(), tf.float32)

# Convert Data into Tensorflow Datasets
train_ds =  tf.data.Dataset.from_tensor_slices((train_df["Text"], y_train)).shuffle(buffer_size=1000).batch(64)
val_ds =  tf.data.Dataset.from_tensor_slices((val_df["Text"], y_val)).shuffle(buffer_size=1000).batch(64)
test_ds = tf.data.Dataset.from_tensor_slices((test_df["Text"], y_test))

tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

def build_bert_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
  preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3", name='preprocessing')
  encoder_inputs = preprocessor(text_input)
  encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1", trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.2)(net)
  net = tf.keras.layers.Dense(512, activation="relu")(net)
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(28, activation="softmax", name='classifier')(net)
  return tf.keras.Model(text_input, net)

model = build_bert_model()
loss = tf.keras.losses.CategoricalCrossentropy()
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model.compile(optimizer="adam", loss=loss, metrics="accuracy")
history = model.fit(x=train_ds, validation_data=val_ds, epochs=15, callbacks=[callback])

model.save_weights('bert.h5')

test_predictions = model.predict(test_df["Text"])
test_predictions = np.argmax(test_predictions, axis=1)
test_labels = np.argmax(y_test, axis=1)
test_acc = accuracy_score(test_labels, test_predictions)
print(f"Test Accuracy is {test_acc*100}%")
