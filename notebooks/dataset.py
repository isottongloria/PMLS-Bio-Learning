import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow import keras


def shuffle_data(xt, yt, xv, yv):

    # xt, yt: training samples and labels
    # xv, yv: validation samples and labels

    pt = np.random.permutation(len(xt))
    pv = np.random.permutation(len(xv))
    return(xt[pt], yt[pt], xv[pv], yv[pv])
    
def load_and_preprocess(data_name, dataset_loader, img_shape, Nc, val_split=0.1, is_color=False):

    if data_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        print(x_train.shape)
        # reshape to (# samples, 784)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]* x_train.shape[3])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2]* x_test.shape[3])

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        x_train -= x_train.mean(axis=0, keepdims=True)
        x_test -= x_test.mean(axis=0, keepdims=True)
        x_train /= np.linalg.norm(x_train, ord=2, axis=1, keepdims=True)
        x_test /= np.linalg.norm(x_test, ord=2, axis=1, keepdims=True)

        # shuffle
        x_train, y_train, x_test, y_test = shuffle_data(x_train, y_train, x_test, y_test)

        # indices of validation splilt
        val_idx = np.random.choice(x_train.shape[0], int(val_split * x_train.shape[0]), replace=False)

        #split validation
        x_val = x_train[val_idx]
        y_val = y_train[val_idx]
        x_train = np.delete(x_train, val_idx, axis=0)
        y_train = np.delete(y_train, val_idx, axis=0)

        # some logging
        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)
        print("x_val shape:", x_val.shape)
        print("y_val shape:", y_val.shape)
        print(x_train.shape[0], "train samples")
        print(x_test.shape[0], "test samples")

        # convert class vectors to binary class matrices
        #y_train = keras.utils.to_categorical(y_train, Nc)
        #y_val = keras.utils.to_categorical(y_val, Nc)
        #y_test = keras.utils.to_categorical(y_test, Nc)

        (x_train_plot, y_train_plot), (x_test_plot, y_test_plot) = keras.datasets.cifar10.load_data()
        for i in range(9):
            plt.subplot(3,3,i+1)
            num = random.randint(0, len(x_train_plot))
            plt.imshow(x_train_plot[num].reshape(32,32,3), cmap='BuGn', interpolation='none')
            plt.title("Class {}".format(y_train_plot[num]))
		
        plt.tight_layout()
        return x_train, y_train, x_val, y_val, x_test, y_test

    if data_name == 'mnist': 
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        print(x_train.shape)
        # reshape to (# samples, 784)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        x_train -= x_train.mean(axis=0, keepdims=True)
        x_test -= x_test.mean(axis=0, keepdims=True)
        x_train /= np.linalg.norm(x_train, ord=2, axis=1, keepdims=True)
        x_test /= np.linalg.norm(x_test, ord=2, axis=1, keepdims=True)

        # shuffle
        x_train, y_train, x_test, y_test = shuffle_data(x_train, y_train, x_test, y_test)

        # indices of validation splilt
        val_idx = np.random.choice(x_train.shape[0], int(val_split * x_train.shape[0]), replace=False)

        #split validation
        x_val = x_train[val_idx]
        y_val = y_train[val_idx]
        x_train = np.delete(x_train, val_idx, axis=0)
        y_train = np.delete(y_train, val_idx, axis=0)

        # some logging
        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)
        print("x_val shape:", x_val.shape)
        print("y_val shape:", y_val.shape)
        print(x_train.shape[0], "train samples")
        print(x_test.shape[0], "test samples")

        # convert class vectors to binary class matrices
        #y_train = keras.utils.to_categorical(y_train, Nc)
        #y_val = keras.utils.to_categorical(y_val, Nc)
        #y_test = keras.utils.to_categorical(y_test, Nc)

        (x_train_plot, y_train_plot), (x_test_plot, y_test_plot) = keras.datasets.mnist.load_data()
        for i in range(9):
            plt.subplot(3,3,i+1)
            num = random.randint(0, len(x_train_plot))
            plt.imshow(x_train_plot[num].reshape(28,28), cmap='BuGn', interpolation='none')
            plt.title("Class {}".format(y_train_plot[num]))
		
        plt.tight_layout()
        return x_train, y_train, x_val, y_val, x_test, y_test


    else:
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        print(x_train.shape)
        # reshape to (# samples, 784)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        x_train -= x_train.mean(axis=0, keepdims=True)
        x_test -= x_test.mean(axis=0, keepdims=True)
        x_train /= np.linalg.norm(x_train, ord=2, axis=1, keepdims=True)
        x_test /= np.linalg.norm(x_test, ord=2, axis=1, keepdims=True)

        # shuffle
        x_train, y_train, x_test, y_test = shuffle_data(x_train, y_train, x_test, y_test)

        # indices of validation splilt
        val_idx = np.random.choice(x_train.shape[0], int(val_split * x_train.shape[0]), replace=False)

        #split validation
        x_val = x_train[val_idx]
        y_val = y_train[val_idx]
        x_train = np.delete(x_train, val_idx, axis=0)
        y_train = np.delete(y_train, val_idx, axis=0)

        # some logging
        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)
        print("x_val shape:", x_val.shape)
        print("y_val shape:", y_val.shape)
        print(x_train.shape[0], "train samples")
        print(x_test.shape[0], "test samples")

        # convert class vectors to binary class matrices
        #y_train = keras.utils.to_categorical(y_train, Nc)
        #y_val = keras.utils.to_categorical(y_val, Nc)
        #y_test = keras.utils.to_categorical(y_test, Nc)

        (x_train_plot, y_train_plot), (x_test_plot, y_test_plot) = keras.datasets.fashion_mnist.load_data()
        for i in range(9):
            plt.subplot(3,3,i+1)
            num = random.randint(0, len(x_train_plot))
            plt.imshow(x_train_plot[num].reshape(28,28), cmap='BuGn', interpolation='none')
            plt.title("Class {}".format(y_train_plot[num]))
		
        plt.tight_layout()

        return x_train, y_train, x_val, y_val, x_test, y_test


