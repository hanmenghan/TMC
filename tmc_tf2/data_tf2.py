import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


class MultiViewData(object):
    """
    load multi-view data
    """

    def __init__(self, root, train=True):
        """
        :param root: data name and path
        :param train: load training set or test set
        """
        super(MultiViewData, self).__init__()
        self.root = root
        self.train = train
        data_path = self.root + '.mat'

        dataset = sio.loadmat(data_path)
        view_number = int((len(dataset) - 5) / 2)
        self.X = dict()
        if train:
            for v_num in range(view_number):
                self.X[v_num] = normalize(dataset['x' + str(v_num + 1) + '_train'])
            y = dataset['gt_train']
        else:
            for v_num in range(view_number):
                self.X[v_num] = normalize(dataset['x' + str(v_num + 1) + '_test'])
            y = dataset['gt_test']

        if np.min(y) == 1:
            y = y - 1
        tmp = np.zeros(y.shape[0])
        y = np.reshape(y, np.shape(tmp))
        self.y = y

    def get_dataset(self):
        return tf.data.Dataset.from_tensor_slices((self.X, self.y))


def normalize(x, min=0):
    if min == 0:
        scaler = MinMaxScaler([0, 1])
    else:  # min=-1
        scaler = MinMaxScaler((-1, 1))
    norm_x = scaler.fit_transform(x)
    return norm_x
