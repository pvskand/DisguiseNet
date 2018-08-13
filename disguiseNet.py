import keras
from keras.utils.np_utils import to_categorical
from keras import regularizers
from keras import optimizers
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.layers import Flatten, Dense, Input, Merge, Subtract, Multiply, Lambda
from keras.layers.normalization import BatchNormalization
from keras.engine import Model
from scipy.misc import imread, imresize, imshow
from keras import backend as K
from keras.engine.topology import Layer
from keras.objectives import categorical_crossentropy, mean_squared_error
import random
import numpy as np
import tensorflow as tf

base_dir = ''


def get_data_from_file(file):
    with open(file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    data_list = []
    for i, val in enumerate(content):
        ii = val.split(' ')
        temp = [ii[0], ii[1], ii[2], ii[3], ii[4]]
        data_list.append(temp)
    data_list = np.asarray(data_list)
    return data_list


def load_data(training_np):
    training = np.load(training_np)
    # random.shuffle(training)
    size = training.shape[0]
    train_data = np.zeros((size, 224, 224, 6), dtype=np.float32)
    train_labels = np.zeros((size, ))
    count = 0
    for i in training:
        if count >= size:
            break
        img1 = imread(base_dir + i[0])
        img1 = imresize(img1, (224, 224))
        img1 = np.float32(img1)

        img1[:, :, 0] -= 93.5940
        img1[:, :, 1] -= 104.7624
        img1[:, :, 2] -= 129.1863

        train_data[count, :, :, 0:3] = img1
        # image 2
        img2 = imread(base_dir + i[1])
        img2 = imresize(img2, (224, 224))
        img2 = np.float32(img2)

        img2[:, :, 0] -= 93.5940
        img2[:, :, 1] -= 104.7624
        img2[:, :, 2] -= 129.1863

        train_data[count, :, :, 3:6] = img2
        train_labels[count] = int(i[2])

        count += 1
    train_data /= 255.0
    return train_data, train_labels


def euc_dist(x):
    'Merge function: euclidean_distance(u,v)'
    return K.sqrt(K.sum(K.square(x[0] - x[1]), axis=-1, keepdims=True))


def euc_dist_shape(input_shape):
    'Merge output shape'
    shape = list(input_shape)
    outshape = (shape[0][0], 1)
    return tuple(outshape)


def contrastive_loss(y, d):
    margin = 0.2
    return K.mean(y * 0.5 * K.square(d) +
                  (1 - y) * 0.5 * K.square(K.maximum(margin - d, 0)))


def model():

    # VGG model initialization with pretrained weights

    vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))
    last_layer = vgg_model.get_layer('pool5').output
    for i in vgg_model.layers[0:16]:
        i.trainable = False
    print vgg_model.summary()
    # fc8 = Dense(2, activation='sigmoid', name='fc8')(last_layer)
    custom_vgg_model = Model(vgg_model.input, last_layer)
    original_img = Input(shape=(224, 224, 3), name='original')
    imp_disguise = Input(shape=(224, 224, 3), name='imp_disguise')

    original_net_out = custom_vgg_model(original_img)
    original_net_out = Flatten()(original_net_out)

    imp_net_out = custom_vgg_model(imp_disguise)
    imp_net_out = Flatten()(imp_net_out)

    concat = keras.layers.Concatenate(axis=-1)([original_net_out, imp_net_out])

    fc1 = Dense(2048, activation="relu")(concat)
    fc2 = Dense(1024, activation="relu")(fc1)
    fc3 = Dense(
        1, activation="sigmoid", name='imposter_disguise_classification')(fc2)

    model = Model([original_img, imp_disguise], [fc3])
    print model.summary()
    return model


def train(model):
    x_train, y_train = load_data(training_np)
    x_val, y_val = load_data(validation_np)

    train_labels_verification = to_categorical(y_train, num_classes=2)

    val_labels_verification = to_categorical(y_val, num_classes=2)

    sgd = optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)

    model.compile(
        loss=[mean_squared_error], optimizer=sgd, metrics=['accuracy'])
    print model.summary()

    model.fit(
        [x_train[:, :, :, 0:3], x_train[:, :, :, 3:6]], [y_train],
        batch_size=75,
        epochs=50,
        verbose=1,
        shuffle=True,
        validation_data=([x_val[:, :, :, 0:3], x_val[:, :, :, 3:6]], [y_val]))
    pred = model.predict([x_val[:, :, :, 0:3], x_val[:, :, :, 3:6]])


if __name__ == "__main__":

    # For the training stage
    accu = 0
    accu_list = []

    training_np = 'training.npy'  # 'training.npy contains the pairs of image paths with labels for training'
    # testing_np = 'data1/testing_1.txt'
    validation_np = 'val.npy'  # val.npy contains the pairs of image paths with labels for validation

    model = model()
    train(model)
    model.save_weights("best_model.h5")
