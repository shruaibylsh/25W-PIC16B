
# %% [markdown]
# Have you ever wondered how computers tell apart images showing different categories in the "verify if you are human" questions? In this blog post, we'll explore image classification using Keras and TensorFlow datasets. We'll build a system that can distinguish between pictures of cats and dogs – similar to how these verification systems might identify cars, crosswalks, or traffic lights.

# %% [markdown]
# # Data Preparation
# ## Loading Packages and Obtaining Data
# 
# Let's import the necessary libraries for our project:
# 

# %%
import os
import keras
from keras import utils
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# Let's first load the dataset. We’ll be using the `cats_vs_dogs` dataset from Kaggle, which contains labeled images of cats and dogs. We’ll split the dataset into training, validation, and test sets:

# %%
train_ds, validation_ds, test_ds = tfds.load(
    "cats_vs_dogs",
    # 40% for training, 10% for validation, and 10% for test (the rest unused)
    split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
    as_supervised=True,  # Include labels
)

print(f"Number of training samples: {train_ds.cardinality()}")
print(f"Number of validation samples: {validation_ds.cardinality()}")
print(f"Number of test samples: {test_ds.cardinality()}")

# %% [markdown]
# The dataset contains images of different sizes, which is problematic for neural networks that expect inputs of consistent dimensions. Let's resize all images to a fixed size of 150x150 pixels:

# %%
resize_fn = keras.layers.Resizing(150, 150)

train_ds = train_ds.map(lambda x, y: (resize_fn(x), y))
validation_ds = validation_ds.map(lambda x, y: (resize_fn(x), y))
test_ds = test_ds.map(lambda x, y: (resize_fn(x), y))

# %% [markdown]
# To ensure efficient training, we'll optimize our data pipeline:

# %%
from tensorflow import data as tf_data
batch_size = 64

train_ds = train_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
validation_ds = validation_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
test_ds = test_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()

# %% [markdown]
# ## Understanding the Data Set
# Before training a model, it’s important to understand the dataset. Let’s visualize some images to get a sense of what we’re working with. We’ll create a function to display three random cat images and three random dog images:

# %%
def visualize_cats_and_dogs(dataset):
    cat_images = []
    dog_images = []

    # retrive 3 images for cats and dogs each
    for images, labels in dataset.take(1): # take 1 batch
        for image, label in zip(images, labels):
            if label == 0 and len(cat_images) < 3:
                cat_images.append(image.numpy())
            elif label == 1 and len(dog_images) < 3:
                dog_images.append(image.numpy())
            if len(cat_images) == 3 and len(dog_images) == 3:
                break

    plt.figure(figsize=(10, 5))
    for i in range(3):
        plt.subplot(2, 3, i + 1)
        plt.imshow(cat_images[i] / 255.0)
        plt.title("Cat")
        plt.axis("off")

        plt.subplot(2, 3, i + 4)
        plt.imshow(dog_images[i] / 255.0)
        plt.title("Dog")
        plt.axis("off")

    plt.show()

visualize_cats_and_dogs(train_ds)

# %% [markdown]
# Next, it's also important for us to know the distribution of labels in the dataset. This helps us establish a baseline for our model, which is the model tat always guesses the most frequent label. We'll treat this as the benchmark for improvement.
# 
# Let’s compute the number of cat and dog images in the training set:

# %%
labels_iterator = train_ds.unbatch().map(lambda image, label: label).as_numpy_iterator()

cat_count = 0
dog_count = 0

for label in labels_iterator:
    if label == 0:
        cat_count += 1
    else:
        dog_count += 1

baseline_accuracy = max(cat_count, dog_count) / (cat_count + dog_count) * 100

print(f"Number of cat images: {cat_count}")
print(f"Number of dog images: {dog_count}")
print(f"Baseline accuracy: {baseline_accuracy:.2f}%")

# %% [markdown]
# We see that the accuracy for our baseline model is 50.17%, indicating an even distribution of cats and dogs in our dataset.
# 
# For the next steps, our goal is to build a model that performs significantly better than this baseline.

# %% [markdown]
# # Basic CNN Model with Keras
# Let's start by building our first convolutional neural network (CNN) model using Keras. We’ll include convolutional layers, pooling layers, and dropout layers to create a robust architecture. Finally, we’ll train the model and evaluate its performance.

# %%
from keras import layers, models

model1 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Binary classification (cat or dog)
])

# %% [markdown]
# After experimenting with different parameters for the model (using 2 conv2D and maxpooling layers, larger dropour rates), here is the number of parameters in each layer we are using for model 1, which achieves decent accuracy;

# %%
model1.summary()

# %% [markdown]
# ## Training the Model
# Let's now train our model for 20 epochs.

# %%
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

hist1 = model1.fit(train_ds, epochs = 20, validation_data = validation_ds)


# %% [markdown]
# ## Visualizing Model Accuracy
# Now, let's visualize our model training results to better understand its accuracy. We’ll plot the training and validation accuracy to evaluate the model’s performance.
# 

# %%
# Plot training and validation accuracy
def visualize_model_accuracy(history):
  plt.plot(history.history['accuracy'], label='Training Accuracy')
  plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.title('Training and Validation Accuracy')
  plt.legend()
  plt.show()

visualize_model_accuracy(hist1)

# %% [markdown]
# **During training, the validation accuracy of the model stablizes between 55% to 60% during training.** It is slightly better than the baseline model by 5%.
# 
# However, overfitting is observed because the training accuracy keeps increasing while the validation accuracy stabilizes at a value well below it. This indicates that we have overfitted our model on the training data set such that it does not generalizes too well to the validation set.

# %% [markdown]
# # Model with Data Augmentation
# In this section, we’ll improve our model by adding data augmentation layers. Data augmentation is a technique that artificially expands the training dataset by applying random transformations (e.g., flipping, rotating, zooming) to the images. This helps the model generalize better and reduces overfitting.

# %% [markdown]
# ## Adding Data Augmentation Layers
# We’ll use 2 argumentation layers:
# - RandomFlip: Randomly flips images horizontally or vertically.
# - RandomRotation: Randomly rotates images by a specified angle.
# 
# Let’s visualize the effect of these transformations on a sample image:

# %%
# Load a sample image
for images, labels in train_ds.take(1):
    sample_image = images[0].numpy()

# Define augmentation layers
flip_layer = layers.RandomFlip("horizontal_and_vertical")
rotate_layer = layers.RandomRotation(0.15)  # Rotate by up to 15%

# Apply augmentations
flipped_images = [flip_layer(sample_image) for _ in range(3)]
rotated_images = [rotate_layer(sample_image) for _ in range(3)]

# Plot the original, flipped, and rotated images
plt.figure(figsize=(10, 7))

# Row 1: Original and flipped images
plt.subplot(2, 3, 1)
plt.imshow(sample_image / 255.0)
plt.title("Original")
plt.axis("off")

for i, img in enumerate(flipped_images):
    plt.subplot(2, 3, i + 2)
    plt.imshow(img / 255.0)
    plt.title(f"Flipped {i + 1}")
    plt.axis("off")

# Row 2: Rotated images
for i, img in enumerate(rotated_images):
    plt.subplot(2, 3, i + 4)
    plt.imshow(img / 255.0)
    plt.title(f"Rotated {i + 1}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Training the Model
# Now, let’s create model 2 which includes the data augmentation layers. The architecture will be similar to model1, but with augmentation layers added at the beginning:

# %%
model2 = models.Sequential([
    layers.Input(shape=(150, 150, 3)),

    # Data augmentation layers
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),

    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Fully connected layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model2.summary()

# %% [markdown]
# Now, let's train the model again for 20 epochs.

# %%
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

hist2 = model2.fit(train_ds, epochs = 20, validation_data = validation_ds)

# %% [markdown]
# ## Visualizing Model Accuracy

# %%
visualize_model_accuracy(hist2)

# %% [markdown]
# **During training, the validation accuracy of the model keeps increasing and stablizes between 75% and 80%**, which is about 20% better than model 1. There is not so much overfitting as the trends of the training and validation accuracy align very well.

# %% [markdown]
# # Data Preprocessing Model
# Now, we’ll enhance our model by adding data preprocessing to normalize the pixel values of the images. Normalizing pixel values (e.g., scaling them to a range of 0 to 1 or -1 to 1) can help the model train faster and converge more effectively. We’ll incorporate this preprocessing step into our model pipeline.

# %%
def preprocess():
  i = keras.Input(shape=(150, 150, 3))
  # The pixel values have the range of (0, 255), but many models will work better if rescaled to (-1, 1.)
  # outputs: `(inputs * scale) + offset`
  scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
  x = scale_layer(i)
  preprocessor = keras.Model(inputs = i, outputs = x)
  return preprocessor

preprocess_layer = preprocess()

# %% [markdown]
# ## Training the Model

# %%
model3 = models.Sequential([
    layers.Input(shape=(150, 150, 3)),

    # Preprocessing layer
    preprocess_layer,

    # Data augmentation layers
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),

    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),


    # Fully connected layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model3.summary()

# %%
model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

hist3 = model3.fit(train_ds, epochs = 20, validation_data = validation_ds)

# %% [markdown]
# ## Visualizing Model Accuracy

# %%
visualize_model_accuracy(hist3)

# %% [markdown]
# **During training, the validation accuracy of the model stablizes between 85% and 88%**, which is about 30% better than model 1. There is very little evidence of overfitting - the training accuracy is a little (about 0.02%) higher than the validation accuracy.

# %% [markdown]
# ## Transfer Learning Model
# Now, we’ll leverage transfer learning to build a highly accurate model for classifying cats and dogs. Transfer learning allows us to use a pre-trained model (trained on a large dataset like ImageNet) as a starting point for our task. This approach is especially useful when working with limited data, as it enables us to benefit from the features learned by the pre-trained model.
# 
# We’ll use `MobileNetV3Large`, a pre-trained model, as the base for our new model.

# %%
IMG_SHAPE = (150, 150, 3)
base_model = keras.applications.MobileNetV3Large(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

i = keras.Input(shape=IMG_SHAPE)
x = base_model(i, training = False)
base_model_layer = keras.Model(inputs = i, outputs = x)

# %% [markdown]
# ## Training the Model
# Now, let’s build model4 using data augmentation layers from previous models, `MobileNetV3Large` as the base model, and additional layers for classification.

# %%
model4 = models.Sequential([
    layers.Input(shape=(150, 150, 3)),

    # Data augmentation layers
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),

    # Base model (MobileNetV3Large)
    base_model_layer,

    # Additional layers
    layers.GlobalMaxPooling2D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model4.summary()

# %% [markdown]
# In the summary of model 4, we notice that there are 2,996,352 non-trainable parameters, which are hidden in the base_model_layer. Therefore, we are only training 961 parameters here.

# %%
model4.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

hist4 = model4.fit(train_ds, epochs = 20, validation_data = validation_ds)

# %% [markdown]
# ## Visualizing Model Accuracy
# 

# %%
visualize_model_accuracy(hist4)

# %% [markdown]
# **During training, the validation accuracy of the model stablizes above 95%.** This is 30% better than model 1! There is little overfitting as the accuracy of the validation set exceeds that of the training set.

# %% [markdown]
# # Summary and Comparison
# Now that we have built four different models, let's compare their results and evaluate their performance on the unseen test dataset. This will give us a clear understanding of how well our best model generalizes to new data.
# 
# Let's compare their validation accuracy on the same plot:

# %%
# Plot validation accuracy for all models
plt.figure(figsize=(10, 6))

plt.plot(hist1.history['val_accuracy'], label='Model 1')
plt.plot(hist2.history['val_accuracy'], label='Model 2')
plt.plot(hist3.history['val_accuracy'], label='Model 3')
plt.plot(hist4.history['val_accuracy'], label='Model 4')

plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy of All Models')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# We see that the validation accuracy of model 4, the transfer learning model, achieves the highest validation accuracy and is well above other models.
# 
# Let's also test the performance of the four models on the test set:

# %%
# Evaluate model1 on the test set
test_loss1, test_accuracy1 = model1.evaluate(test_ds)
print(f"Model 1 - Test Accuracy: {test_accuracy1 * 100:.2f}%")

# Evaluate model2 on the test set
test_loss2, test_accuracy2 = model2.evaluate(test_ds)
print(f"Model 2 - Test Accuracy: {test_accuracy2 * 100:.2f}%")

# Evaluate model3 on the test set
test_loss3, test_accuracy3 = model3.evaluate(test_ds)
print(f"Model 3 - Test Accuracy: {test_accuracy3 * 100:.2f}%")

# Evaluate model4 on the test set
test_loss4, test_accuracy4 = model4.evaluate(test_ds)
print(f"Model 4 - Test Accuracy: {test_accuracy4 * 100:.2f}%")

# %% [markdown]
# We see that the test accuracy for model 4 is 95.83%, well above that for other models that we built from scratch. This shows that transfer learning might be a really effective approach in model training!


