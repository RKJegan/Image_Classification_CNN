import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image


# Load Dataset (CIFAR-10)

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Normalize pixel values
X_train, X_test = X_train / 255.0, X_test / 255.0

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']


# Build CNN Model

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# Compile Model

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Train Model

history = model.fit(X_train, y_train,
                    epochs=10,
                    validation_data=(X_test, y_test))


# Evaluate Model

test_loss, test_acc = model.evaluate(X_test, y_test)
#print("Test Accuracy:", test_acc)

# Predict User Image

def predict_user_image(img_path):
    img = image.load_img(img_path, target_size=(32,32))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    print("Predicted Class:", predicted_class)
    print("Confidence:", round(np.max(prediction)*100,2), "%")

# Take User Input

user_image_path = input("Enter image path: ")
predict_user_image(user_image_path)

