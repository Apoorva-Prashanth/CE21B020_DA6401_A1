import wandb
import tensorflow as tf
from tensorflow import keras
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

wandb.init(project="fashion-mnist_DA6401_A1", name="Class_image_plot")


#loading the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# List of FashionMNIST labels, source: https://github.com/zalandoresearch/fashion-mnist
class_names = ["T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

plt.figure(figsize=(10, 5))

#find unique class and log
unique_labels = set()
for i in range(len(train_labels)):
    label = train_labels[i]
    if label not in unique_labels:
        plt.subplot(2, 5, len(unique_labels) + 1)
        plt.imshow(train_images[i], cmap="gray")
        plt.title(class_names[label])
        plt.axis("off")
        unique_labels.add(label)
    if len(unique_labels) == 10:
        break

plt.tight_layout()

#log figure in WandB
wandb.log({"fashion_mnist_classes": wandb.Image(plt)})
wandb.finish()


