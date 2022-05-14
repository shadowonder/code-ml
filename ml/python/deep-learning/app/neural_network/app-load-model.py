from tensorflow import keras

# 读取模型
model = keras.models.load_model('fashion_model.h5')
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
predictions = model.predict(test_images)
print(predictions.shape)  # (10000, 10)

model = keras.models.model_from_json('{}')
model.summary()

model.load_weights('feshion.weights')
