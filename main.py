import tensorflow as tf

loaded = tf.saved_model.load('./kaggle_model')
print(list(loaded.signatures.keys()))

signature_key = 'serving_default'
signature = loaded.signatures[signature_key]

print(signature)