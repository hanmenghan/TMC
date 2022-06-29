import os
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import TMC
from data import Multi_view_data
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR',
                        help='learning rate')
    args = parser.parse_args()
    args.data_name = 'handwritten_6views'
    args.data_path = 'datasets/' + args.data_name
    args.dims = [[240], [76], [216], [47], [64], [6]]
    args.views = len(args.dims)

    train_loader = torch.utils.data.DataLoader(
        Multi_view_data(args.data_path, train=True), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        Multi_view_data(args.data_path, train=False), batch_size=args.batch_size, shuffle=False)
    N_mini_batches = len(train_loader)
    print('The number of training images = %d' % N_mini_batches)

    model = TMC(10, args.views, args.dims, args.lambda_epochs)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    model.cuda()

    def train(epoch):
        model.train()
        loss_meter = AverageMeter()
        for batch_idx, (data, target) in enumerate(train_loader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].cuda())
            target = Variable(target.long().cuda())
            # refresh the optimizer
            optimizer.zero_grad()
            evidences, evidence_a, loss = model(data, target, epoch)
            # compute gradients and take step
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())


    def test(epoch):
        model.eval()
        loss_meter = AverageMeter()
        correct_num, data_num = 0, 0
        for batch_idx, (data, target) in enumerate(test_loader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].cuda())
            data_num += target.size(0)
            with torch.no_grad():
                target = Variable(target.long().cuda())
                evidences, evidence_a, loss = model(data, target, epoch)
                _, predicted = torch.max(evidence_a.data, 1)
                correct_num += (predicted == target).sum().item()
                loss_meter.update(loss.item())

        print('====> acc: {:.4f}'.format(correct_num/data_num))
        return loss_meter.avg, correct_num/data_num

    for epoch in range(1, args.epochs + 1):
        train(epoch)

    test_loss, acc = test(epoch)
    print('====> acc: {:.4f}'.format(acc))
