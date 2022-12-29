import numpy as np
import keras
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from preprocessor import Preprocessor


class Model:
    def preprocess_data(self, train_annot_path, test_annot_path, resize_dims):
        """If there is need to preprocess data from zero."""
        prep = Preprocessor()
        self.train_imgs, self.train_labels = prep.preprocess_data(
            annot_path=train_annot_path,
            new_annot_path="",
            resize_dims=resize_dims,
            new_images_path="",
            save_new_data=False,
        )
        self.test_imgs, self.test_labels = prep.preprocess_data(
            annot_path=test_annot_path,
            new_annot_path="",
            resize_dims=resize_dims,
            new_images_path="",
            save_new_data=False,
        )

    def load_data(
        self, train_imgs_path, test_imgs_path, train_labels_path, test_labels_path
    ):
        with open(train_imgs_path, "rb") as f:
            self.train_imgs = np.load(f)
        with open(test_imgs_path, "rb") as f:
            self.test_imgs = np.load(f)
        with open(train_labels_path, "rb") as f:
            self.train_labels = np.load(f)
        with open(test_labels_path, "rb") as f:
            self.test_labels = np.load(f)

    def process_data_for_training(self):

        # 0 is labeled as 10, so there will be shift
        self.train_labels[self.train_labels == 10] = 0
        self.test_labels[self.test_labels == 10] = 0

        # Convert train and test images from uint8 into 'float64' type
        self.train_imgs = self.train_imgs.astype("float64")
        self.test_imgs = self.test_imgs.astype("float64")

        # normalize image data
        self.train_imgs /= 255.0
        self.test_imgs /= 255.0

        # One-hot encoding of train and test labels
        self.lb = LabelBinarizer()
        self.train_labels = self.lb.fit_transform(self.train_labels)
        self.test_labels = self.lb.fit_transform(self.test_labels)

        # Split train data into train and validation sets

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.train_imgs, self.train_labels, test_size=0.1, random_state=22
        )

    def plot_class_distribution(self):
        # plot class distribution
        plt.figure(figsize=(15, 8))  # change your figure size as per your desire here
        plt.hist(
            self.train_labels,
            bins=10,
            histtype="bar",
            color="blue",
            stacked=True,
            fill=True,
            edgecolor="black",
            linewidth=1.2,
        )
        plt.xticks(range(10))
        plt.title("Number of items per class")
        plt.xlabel("Classes")
        plt.ylabel("Count")
        plt.show()

    def create_model(self):
        self.datagen = ImageDataGenerator(
            rotation_range=15, zoom_range=[0.9, 1.1], height_shift_range=0.10
        )

        self.model = keras.Sequential(
            [
                keras.layers.Conv2D(
                    32,
                    (3, 3),
                    padding="same",
                    activation="relu",
                    input_shape=(32, 32, 3),
                ),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(32, (5, 5), padding="same", activation="relu"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.3),
                keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(64, (5, 5), padding="same", activation="relu"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.3),
                keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(128, (5, 5), padding="same", activation="relu"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.3),
                keras.layers.Flatten(),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(10, activation="softmax"),
            ]
        )

        early_stopping = keras.callbacks.EarlyStopping(patience=5)
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            "runnings/best_cnn.h5", save_best_only=True
        )
        self.callbacks = [early_stopping, model_checkpoint]
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001, amsgrad=True),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def fit(self, epochs=10, batch_size=16):
        self.model.fit(
            self.datagen.flow(self.X_train, self.y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=self.callbacks,
        )
    
    def predict(self, images):
                
        y_pred = self.model.predict(images)

        y_pred = self.lb.inverse_transform(y_pred, self.lb.classes_)
        return y_pred
    

    def plot_confusion_matrix(self, y_test, y_pred):
        
        matrix = confusion_matrix(y_test, y_pred, labels=self.lb.classes_)

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(matrix, annot=True, cmap='Blues', fmt='d', ax=ax)
        plt.title('Confusion Matrix for test dataset')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.show()

m = Model()
# loading and processing data
# m.preprocess_data("train_annotations.csv", "test_annotations.csv", (32, 32))
m.load_data("train_imgs.npy", "test_imgs.npy", "train_labels.npy", "test_labels.npy")
m.process_data_for_training()

# creating and fitting model
m.create_model()
m.model.load_weights("runnings/best_cnn.h5")
# m.fit(epochs=1, batch_size=32)

# testing and confusion matrix
y_pred = m.predict(m.test_imgs)
y_test = m.lb.inverse_transform(m.test_labels, m.lb.classes_)
m.plot_confusion_matrix(y_test, y_pred)
