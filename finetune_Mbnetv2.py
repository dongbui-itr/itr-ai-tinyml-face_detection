import numpy as np
import tensorflow as tf
import cv2
import glob

from matplotlib import pyplot as plt

input_shape = (96, 96, 3)
path_train = "/home/ttb/Downloads/data_yolovX/WIDER_train/WIDER_train/images"
path_test = "/home/ttb/Downloads/data_yolovX/WIDER_val/WIDER_val/images"
model_file = '/home/ttb/Downloads/face-detection-yolov3-keras-main/MbNetV2Custom.keras'
model_file_dense = '/home/ttb/Downloads/face-detection-yolov3-keras-main/MbNetV2Custom_dense.keras'
img_test = "/home/ttb/Downloads/data_yolovX/WIDER_train/WIDER_train/images/56--Voter/56_Voter_peoplevoting_56_512.jpg"


class mobilenet_custom:
    def __init__(self, _file_path_save, _last_layer_name):
        self.file_path = _file_path_save
        self.last_layer_name = _last_layer_name
        self.model_custom = None

    def mobilenet_custom_model(self, _mobilenet_model):
        self.model_custom = tf.keras.models.Sequential()
        for layer in _mobilenet_model.layers:
            layer.trainable = False
            if layer.name != "block_2_add":
                print(f"{layer.name}, {layer.input.shape}, {layer.output.shape}")
                self.model_custom.add(layer)
            if layer.name == self.last_layer_name:
                self.model_custom.add(tf.keras.layers.Conv2D(16, 3,
                                                             activation='relu',
                                                             padding="same",
                                                             input_shape=(12, 12, 16)))
                self.model_custom.add(tf.keras.layers.Conv2D(32, 3,
                                                             activation='relu',
                                                             padding="same"))
                # self.model_custom.add(tf.keras.layers.Dense(16))
                # self.model_custom.add(tf.keras.layers.Dense(32))
                self.model_custom.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.softmax))
                break
        self.model_custom.summary()
        return self.model_custom

    def save_model(self):
        self.model_custom.save(self.file_path)

    def load_model(self):
        model_load = tf.keras.models.load_model(self.file_path)
        model_load.summary()
        return model_load


def load_data(data_path_train, data_path_val):
    x_train = np.array([])
    x_val = np.array([])
    y_train = np.array([])
    y_val = np.array([])
    count_train = 0
    count_val = 0
    num_path = 40
    num_file_in_path = 100

    for path in glob.glob(data_path_train + "/*")[:num_path]:
        for photo_filename in glob.glob(path + "/*")[:num_file_in_path]:
            label_file = (("/".join(photo_filename.split("/")[:5])
                          + "/result/"
                          + "/".join(photo_filename.split("/")[6:])).split(".")[0]
                          + "/label_img_split.txt")
            s = np.array((open(label_file).read()
                          .replace('[', ' ')
                          .replace(']', ' ')
                          ).split(','))
            y_train = np.concatenate((y_train, s), axis=0)
            img = cv2.imread(photo_filename)
            img_rs = cv2.resize(img, (96, 96))
            img_rs = img_rs.flatten()
            x_train = np.concatenate((x_train, img_rs), axis=0)
            count_train += 1

    for path in glob.glob(data_path_val + "/*")[:int(num_path/2)]:
        for photo_filename in glob.glob(path + "/*")[:int(num_file_in_path/2)]:
            label_file = (("/".join(photo_filename.split("/")[:5])
                          + "/result/"
                          + "/".join(photo_filename.split("/")[6:])).split(".")[0]
                          + "/label_img_split.txt")
            s = np.array((open(label_file).read()
                          .replace('[', ' ')
                          .replace(']', ' ')
                          ).split(','))
            y_val = np.concatenate((y_val, s), axis=0)

            img = cv2.imread(photo_filename)
            img_rs = cv2.resize(img, (96, 96))
            img_rs = img_rs.flatten()
            x_val = np.concatenate((x_val, img_rs), axis=0)
            count_val += 1

    x_train = np.reshape(x_train, (count_train, 96, 96, 3)).astype(np.uint8)
    x_val = np.reshape(x_val, (count_val, 96, 96, 3)).astype(np.uint8)
    y_train = np.reshape(y_train, (count_train, 12, 12)).astype(np.uint8)
    y_val = np.reshape(y_val, (count_val, 12, 12)).astype(np.uint8)

    # temp = x_val[:, :, :, 0][-1]
    # plt.imshow(temp)
    # plt.show()

    return x_train, x_val, y_train, y_val


def pre_train():
    x_train, x_val, y_train, y_val = load_data(path_train, path_test)
    last_layer_name = "block_3_project_BN"
    model_mb_net = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=input_shape,
        include_top=True,
        weights='imagenet',
        classifier_activation='softmax',
    )
    model_custom = mobilenet_custom(model_file, last_layer_name)
    model_custom.mobilenet_custom_model(model_mb_net)
    model_custom.save_model()
    model_test = model_custom.load_model()
    model_test.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    history = model_test.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_val, y_val))
    model_test.save(model_file_dense)


def main():
    pre_train()
    model_pre_train = mobilenet_custom(model_file_dense, "")
    model_load = model_pre_train.load_model()
    model_load.summary()

    img = cv2.imread(img_test)
    img_rs = cv2.resize(img, (96, 96))
    img_rs = np.expand_dims(img_rs, axis=0)
    y_hat = model_load.predict(img_rs)
    print(y_hat)


if __name__ == '__main__':
    main()
