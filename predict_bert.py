import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

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
model.load_weights('bert.h5')

with open('emotions.txt') as f:
  emotion_list = f.read().split("\n")

def predict_custom(sentence):
  yhat = model.predict([sentence])
  predictions = np.flip(np.argsort(yhat, axis=1)[:, -3:]) # top 3
  return [emotion_list[i] for i in predictions[0].tolist()]

print(predict_custom("My favourite food is anything I didn't have to cook myself.")) # 'love', 'admiration', 'neutral'
print(predict_custom("Thanks! I love watching him every week")) # 'gratitude', 'love', 'admiration'
print(predict_custom("There is too much garbage in the street and it stinks. When will the council clean this mess up?")) # 'curiosity', 'disgust', 'annoyance'
print(predict_custom("I have done so well on my exam after studying so hard. I scored really high marks and so proud of myself.")) # 'admiration', 'joy', 'pride'

