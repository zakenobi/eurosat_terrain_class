# Eurosat terrain classification
This repository is made for the Predictive Modeling course project at EPF. The goal of the project is to classify the Eurosat dataset using different machine learning algorithms. The dataset is available at https://www.tensorflow.org/datasets/catalog/eurosat

## Exploring the data
We can easily import the images with `tensorflow_datasets` and split them into train, validation and test sets. This gives us the following amounts:

```Python
Number of training samples:  16200
Number of validation samples:  5400
Number of test samples:  5400
```

The first 9 images of the dataset are shown below:

![Eurosat dataset](./images/examples.png)

We can see that the size of the images seem very similar, as well as the resolution. The images are also in RGB format. The labels are the following:

```Python
['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
```

For peace of mind, we can check size and that the images are indeed in RGB format:

```Python
for image, label in train_ds.take(5):
    print(image.numpy().shape)
```
```Python
(64, 64, 3)
(64, 64, 3)
(64, 64, 3)
(64, 64, 3)
(64, 64, 3)
```

This is very good, as we can use the images as they are without any heavy preprocessing. Having the same size and resolution is necessary for the convolutional neural networks we will use later because they expect a fixed size input.

## Preprocessing
The first thing we do is make sur the size of the images is the same. Then at the same time generate batches of images and labels. We also shuffle the data and normalize the images.

```Python
train_ds = train_ds.map(resize_image).batch(BATCH_SIZE)\
    .prefetch(buffer_size=tf.data.AUTOTUNE)

val_ds = val_ds.map(resize_image).batch(BATCH_SIZE)\
    .prefetch(buffer_size=tf.data.AUTOTUNE)

test_ds = test_ds.map(resize_image).batch(BATCH_SIZE)\
    .prefetch(buffer_size=tf.data.AUTOTUNE)
```

Additionally, in each model we have a rescaling layer that normalizes the images. This is done because we want the RGB values between 0 and 255.

```Python
layers.Rescaling(1./255, input_shape=(img_height, img_width, 3))
```

## Models

We have tried 3 different models. The first one is a simple convolutional neural network. The second one is a more complex and optimized convolutional neural network. The last one is a transfer learning model using the resnet50 model.

### Simple CNN

The first model is a simple convolutional neural network. It has 3 convolutional layers with 32, 64 and 128 filters respectively. Each convolutional layer is followed by a max pooling layer. The last convolutional layer is followed by a flatten layer and a relu layer with 128 neurons. The output layer is a dense layer with 10 neurons.

```Python
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])
```

This model gives a test accuracy of 0.83 with 10 epochs. The training and validation accuracy and loss are shown below:

![Simple CNN](./images/basic_accuracy.png)

We can see that the model is close to overfitting. This is something we will try to avoid in the next model. The confusion matrix is shown below:

![Simple CNN confusion matrix](./images/basic_confusion_matrix.png)

Over all the model does a good job at classifying the images. The most difficult classes to classify are the PermanentCrop, River and Highway classes. This is probably because these classes are very similar to each other. Maybe a better preprocessing could help with this by using data augmentation.

### Optimized CNN

