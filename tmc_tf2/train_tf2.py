import os

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from model_tf2 import TMC
from data_tf2 import MultiViewData
import warnings

warnings.filterwarnings("ignore")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--lambda-epochs', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--data_name', type=str, default='handwritten_6views')
    parser.add_argument('--cuda_device', type=str, default='-1',
                        help='CUDA device index to be used in training. '
                             'This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. '
                             'Specify it as -1 to disable GPUs.')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    args.data_path = os.path.join('../datasets/', args.data_name)
    args.dims = [[240], [76], [216], [47], [64], [6]]
    args.views = len(args.dims)

    train_loader = MultiViewData(args.data_path, train=True).get_dataset().batch(args.batch_size).shuffle(50000)
    test_loader = MultiViewData(args.data_path, train=False).get_dataset().batch(args.batch_size)
    N_mini_batches = len(train_loader)
    print('The number of training images = %d' % N_mini_batches)

    model = TMC(10, args.views, args.dims, args.lambda_epochs)
    optimizer = Adam(learning_rate=args.lr, decay=1e-5)


    def train(epoch):
        loss_meter = AverageMeter()
        for data, target in train_loader:
            with tf.GradientTape() as tape:
                evidences, evidence_a, loss = model(data, target, epoch)
            # compute gradients and take step
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            loss_meter.update(loss.numpy())


    def test(epoch):
        model.trainable = False
        loss_meter = AverageMeter()
        correct_num, data_num = 0, 0
        for data, target in test_loader:
            data_num += target.shape[0]
            evidences, evidence_a, loss = model(data, target, epoch)
            predicted = tf.cast(tf.argmax(evidence_a, 1), dtype=tf.uint8)
            correct_num += tf.reduce_sum(tf.cast(predicted == target, dtype=tf.uint8))
            loss_meter.update(loss.numpy())

        return loss_meter.avg, correct_num / data_num


    for epoch in range(1, args.epochs + 1):
        train(epoch)

    test_loss, acc = test(epoch)
    print('====> acc: {:.4f}'.format(acc))
