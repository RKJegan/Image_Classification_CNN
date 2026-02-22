
🖼️ TensorFlow Image Classification (CIFAR-10)

This project implements an Image Classification model using TensorFlow & Keras.
The model is trained on the CIFAR-10 dataset and predicts the class of a user-provided image.

📌 Project Overview

Dataset: CIFAR-10

Model: Convolutional Neural Network (CNN)

Classes:

airplane

automobile

bird

cat

deer

dog

frog

horse

ship

truck

The model trains on 60,000 32×32 color images and predicts the category of a custom image input by the user.

🧠 Model Architecture

The CNN model consists of:

Conv2D (32 filters, 3×3)

MaxPooling2D

Conv2D (64 filters, 3×3)

MaxPooling2D

Conv2D (64 filters, 3×3)

Flatten Layer

Dense (64 neurons)

Output Dense Layer (10 classes, Softmax)

Loss Function: sparse_categorical_crossentropy
Optimizer: Adam
Metric: Accuracy

⚙️ Requirements

Install required libraries: pip install tensorflow numpy matplotlib

🚀 How to Run

Clone the repository:

git clone https://github.com/your-username/your-repo-name.git

Navigate to project folder:

cd your-repo-name

Run the script:

python image_classification.py

After training completes, enter the image path:

Enter image path: C:/Users/YourName/Desktop/test.jpg

The model will output:

Predicted Class

Confidence Percentage

📁 Project Structure
├── image_classification.py
├── README.md
