import argparse
import sys
import torch.optim as optim
from random import sample
from torch.optim.lr_scheduler import MultiStepLR
from data.build_graph import *
from models.model import Net
import warnings
from utils.logger import logger
warnings.filterwarnings("ignore", category=Warning)

parser = argparse.ArgumentParser(description='SLI-GNN')
parser.add_argument('data_src', metavar='PATH', help='data source: data/dataset/data_src')
parser.add_argument('filename', metavar='F', help='csv filename(dataset/targets/filename.csv)')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression',
                    help='complete a regression or ''classification task (default: regression)')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run (default: 10)')
parser.add_argument('--pooling', choices=['mean', 'max', 'add'],
                    default='mean', help='global pooling layer (default: mean)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0008188863779378466, type=float,
                    metavar='LR', help='initial learning rate (default: ''0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: ''[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-space', '-p', default=10, type=int,
                    metavar='N', help='print space (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=0.8, type=float, metavar='N',
                         help='number of training data to be loaded (default 0.6)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--valid-ratio', default=0.1, type=float, metavar='N',
                         help='percentage of validation data to be loaded (default '
                              '0.2)')
valid_group.add_argument('--valid-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                        help='percentage of test data to be loaded (default 0.2)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')
parser.add_argument('--atom-fea-len', default=208, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=154, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--nbr-fea-len', default=204, type=int, metavar='N',
                    help='number of bond features')
parser.add_argument('--n-conv', default=14, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--l1', default=0, type=int, metavar='N',
                    help='number of hidden layers before pooling')
parser.add_argument('--l2', default=2, type=int, metavar='N',
                    help='number of hidden layers after pooling')
parser.add_argument('--n-classes', default=2, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('--patience', default=7, type=int, metavar='N',
                    help='How long to wait after last time validation loss improved.(default=7)')
attention_group = parser.add_mutually_exclusive_group()
attention_group.add_argument('--attention', '-GAT', action='store_true',
                             help='Attention or not.(default: False)')
attention_group.add_argument('--dynamic-attention', '-DA', action='store_true',
                             help='Dynamic attention or not.(default: False)')
parser.add_argument('--n-heads', default=1, type=int, metavar='N',
                    help='Number of multi-head-attentions.(default=1, useful on attention mechanism)')
parser.add_argument('--dropout-p', '-d', default=0, type=float, metavar='N',
                    help='dropout - p.(default=0)')
parser.add_argument('--early-stopping', '-es', action='store_true',
                    help='if early stopping or not (default: False)')
parser.add_argument('--transfer', action='store_true', help='default: False')

parser.add_argument('--max-num-nbr', default=12, type=int, metavar='N',
                    help='max number of neighbors')
parser.add_argument('--radius', '-r', default=5, type=int, metavar='N', help='Radius of sphere')
parser.add_argument('--step', default=0.1, type=int, metavar='N', help='distance step')

parser.add_argument('--properties', choices=['N', 'G', 'P', 'NV', 'E', 'R', 'V', 'EA', 'I'],
                    nargs='*', action='append', default=[['N']],
                    help='properties list initializing atom features (default: Only atom number)')

args = parser.parse_args(sys.argv[1:])
best_loss = 1e10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    global args, best_loss
    path = "data/dataset/" + args.data_src
    targets_filename = "data/dataset/targets/" + args.filename + ".csv"

    logger.info('dataset path = {}'.format(path))
    logger.info('neighbor search radius = {}'.format(args.radius))
    logger.info('max neighbor number = {}'.format(args.max_num_nbr))

    properties_list = args.properties[0]
    dataset = GraphData(path=path, targets_filename=targets_filename, max_num_nbr=args.max_num_nbr, radius=args.radius,
                        properties_list=properties_list, step=args.step)

    logger.info('dataset prepared, total {} materials'.format(len(dataset)))
    logger.info('start split dataset by {}:{}:{}'.format(args.train_ratio * 10,
                                                         args.valid_ratio * 10, args.test_ratio * 10))
    train_loader, valid_loader, test_loader = \
        train_val_test_split(dataset,
                             batch_size=args.batch_size,
                             train_ratio=args.train_ratio,
                             valid_ratio=args.valid_ratio,
                             test_ratio=args.test_ratio,
                             num_workers=args.workers,
                             train_size=args.train_size,
                             valid_size=args.valid_size,
                             test_size=args.test_size)

    orig_bond_fea_len = dataset.bond_feature_encoder.num_category

    model = Net(orig_bond_fea_len=orig_bond_fea_len,
                atom_fea_len=args.atom_fea_len,
                nbr_fea_len=args.nbr_fea_len,
                n_conv=args.n_conv,
                h_fea_len=args.h_fea_len,
                l1=args.l1, l2=args.l2,
                classification=True if args.task == 'classification' else False,
                n_classes=args.n_classes,
                attention=args.attention,
                dynamic_attention=args.dynamic_attention,
                n_heads=args.n_heads,
                max_num_nbr=args.max_num_nbr,
                pooling=args.pooling,
                p=args.dropout_p,
                properties_list=properties_list,
                atom_ref=None)
    model.to(device)

    logger.info('Normalizer initializing')
    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        if len(dataset) < 500:
            logger.warning('Dataset has less than 500 data points. '
                           'Lower accuracy is expected. ')
            sample_target = [dataset[i].y for i in range(len(dataset))]
        else:
            sample_target = [dataset[i].y for i in sample(range(len(dataset)), 500)]
        normalizer = Normalizer(torch.tensor(sample_target), model.atomref_layer)
    logger.info('Model initializing')

    if args.task == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), args.lr,
                           weight_decay=args.weight_decay)

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

    train_losses = []
    valid_losses = []

    if args.resume:
        checkpoint_path = 'weights/' + args.resume
        if os.path.isfile(checkpoint_path):
            logger.info("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(checkpoint_path, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(checkpoint_path))

    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    logger.info('start training, use {}'.format(device))
    transfer = args.transfer
    for epoch in range(args.start_epoch, args.epochs):
        logger.info("----------Train Set----------")
        train_loss = train(train_loader, model, criterion, optimizer, epoch, normalizer)
        train_losses.append(train_loss)

        logger.info("----------Valid Set----------")
        valid_loss = validate(valid_loader, model, criterion, epoch, normalizer)
        valid_losses.append(valid_loss)

        scheduler.step()

        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args),
        }, is_best, transfer=transfer)
        transfer = False

        if args.early_stopping:
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

    logger.info("Test with the best model")
    best_checkpoint = torch.load('weights/model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    test_loss = test(test_loader, model, criterion, normalizer, path="test")

    logger.info("saving results and loss")
    test(train_loader, model, criterion, normalizer, path="train")
    test(valid_loader, model, criterion, normalizer, path="valid")

    with open('results/loss.csv', 'w') as f:
        writer = csv.writer(f)
        for epoch, (train_loss, valid_loss) in enumerate(zip(train_losses, valid_losses)):
            writer.writerow((epoch, train_loss, valid_loss))
    df = pd.read_csv('results/loss.csv',
                     header=None,
                     names=['EPOCH', 'Train_Loss', 'Valid_Loss'])
    df.to_csv('results/loss.csv', index=False)
    logger.info('----------------training finished---------------------')


def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    running_loss = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
    for batch_idx, data in enumerate(train_loader, 0):
        if args.task == 'regression':
            targets = data.y.unsqueeze(1)
            targets_normed = normalizer.norm(targets)
        else:
            targets = data.y.long()
            targets_normed = targets
        data, targets_normed = data.to(device), targets_normed.to(device)
        outputs = model(data)
        loss = criterion(outputs, targets_normed)

        running_loss.update(loss.item(), targets.size(0))

        if args.task == 'regression':
            mae = mae_metric(normalizer.denorm(outputs.data.cpu()), targets)
            mae_errors.update(mae, targets.size(0))
            if batch_idx % args.print_space == 0:
                logger.info('epoch: %2d, batch_idx: %2d, loss: %.3f, MAE: %.3f' % (
                    epoch + 1, batch_idx + 1, running_loss.avg, mae_errors.avg))
        else:
            accuracy = class_metric(outputs, targets)
            accuracies.update(accuracy, targets.size(0))
            if batch_idx % args.print_space == 0:
                logger.info('epoch: %2d, batch_idx: %2d, loss: %.3f, accuracy: %.3f' % (
                    epoch + 1, batch_idx + 1, running_loss.avg, accuracies.avg))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return running_loss.avg


def validate(valid_loader, model, criterion, epoch, normalizer):
    running_loss = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
    for batch_idx, data in enumerate(valid_loader, 0):
        with torch.no_grad():
            if args.task == 'regression':
                targets = data.y.unsqueeze(1)
                targets_normed = normalizer.norm(targets)
            else:
                targets = data.y.long()
                targets_normed = targets
            data, targets_normed = data.to(device), targets_normed.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets_normed)
            running_loss.update(loss.item(), targets.size(0))

            if args.task == 'regression':
                mae = mae_metric(normalizer.denorm(outputs.data.cpu()), targets)
                mae_errors.update(mae, targets.size(0))
                if batch_idx % args.print_space == 0:
                    logger.info('epoch: %2d, batch_idx: %2d, loss: %.3f, MAE: %.3f' % (
                        epoch + 1, batch_idx + 1, running_loss.avg, mae_errors.avg))
            else:
                accuracy = class_metric(outputs, targets)
                accuracies.update(accuracy, targets.size(0))
                if batch_idx % args.print_space == 0:
                    logger.info('epoch: %2d, batch_idx: %2d, loss: %.3f, accuracy: %.3f' % (
                        epoch + 1, batch_idx + 1, running_loss.avg, accuracies.avg))

    return running_loss.avg


def test(test_loader, model, criterion, normalizer, path="test"):
    test_material_ids = []
    test_targets = []
    test_preds = []
    if args.task == 'classification':
        probabilities = []

    running_loss = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
    for batch_idx, data in enumerate(test_loader, 0):
        with torch.no_grad():
            if args.task == 'regression':
                targets = data.y.unsqueeze(1)
                targets_normed = normalizer.norm(targets)
            else:
                targets = data.y.long()
                targets_normed = targets
            data, targets_normed = data.to(device), targets_normed.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets_normed)
            running_loss.update(loss.item(), targets.size(0))

            material_id = data.material_id
            test_target = targets

            test_material_ids += material_id
            if args.task == 'regression':
                test_pred = normalizer.denorm(outputs.data.cpu())
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
            else:
                probability = nn.functional.softmax(outputs, dim=1)
                probability = probability.tolist()

                prediction = outputs.cpu().detach().numpy()
                test_pred = np.argmax(prediction, axis=1)
                test_preds += test_pred.tolist()

                test_targets += test_target.view(-1).tolist()
                probabilities += probability

            if args.task == 'regression':
                mae = mae_metric(normalizer.denorm(outputs.data.cpu()), targets)
                mae_errors.update(mae, targets.size(0))
                if path == 'test' and batch_idx % args.print_space == 0:
                    logger.info('batch_idx: %2d, loss: %.3f, MAE: %.3f' % (
                        batch_idx + 1, running_loss.avg, mae_errors.avg))
            else:
                accuracy = class_metric(outputs, targets)
                accuracies.update(accuracy, targets.size(0))
                if path == 'test' and batch_idx % args.print_space == 0:
                    logger.info('batch_idx: %2d, loss: %.3f, accuracy: %.3f' % (
                        batch_idx + 1, running_loss.avg, accuracies.avg))

    if args.task == 'regression':
        with open('results/regression/' + path + '_results.csv', 'w') as f:
            writer = csv.writer(f)
            for material_id, pred, target in zip(test_material_ids, test_preds, test_targets):
                writer.writerow((material_id, pred, target))

        df = pd.read_csv('results/regression/' + path + '_results.csv',
                         header=None, names=['Material_ID', 'Prediction', 'Target'])
        df.to_csv('results/regression/' + path + '_results.csv', index=False)
    else:
        with open('results/classification/' + path + '_results.csv', 'w') as f:
            writer = csv.writer(f)
            for material_id, pred, target, probability in zip(test_material_ids, test_preds, test_targets,
                                                              probabilities):
                writer.writerow((material_id, pred, target, probability))

        df = pd.read_csv('results/classification/' + path + '_results.csv',
                         header=None,
                         names=['Material_ID', 'Prediction', 'Target', 'Probabilities'])
        df.to_csv('results/classification/' + path + '_results.csv', index=False)

    return running_loss.avg


if __name__ == '__main__':
    logger.info('-------------------starting task---------------------')
    main()
