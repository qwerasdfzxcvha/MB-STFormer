from argparse import ArgumentParser
from utils import log2csv, ensure_path
import torchmetrics

import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
import torch
from utils import *



def init_model(args):
    model = None
    if args.model=='IFNet':
        model=IFNet(in_planes=30, out_planes = 64,
                 radix = 5, patch_size = 32, time_points = 384,
                 num_classes = 2)
    return model
class DLModel(pl.LightningModule):
    def __init__(self, config):
        super(DLModel, self).__init__()
        self.save_hyperparameters()
        self.net = init_model(config)
        self.test_step_pred = []
        self.test_step_ground_truth = []
        self.acc = torchmetrics.Accuracy(task='multiclass', num_classes=config.num_class)
        self.F1 = torchmetrics.F1Score(task="multiclass", num_classes=config.num_class, average='macro')
        self.config = config

    def forward(self, x):
        return self.net(x)

    def get_metrics(self, pred, y):
        acc = self.acc(pred, y)
        f1 = self.F1(pred, y)
        return acc, f1

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc, f1 = self.get_metrics(y_hat, y)
        self.log_dict(
            {"train_loss": loss, "train_acc": acc, "train_f1": f1},
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc, f1 = self.get_metrics(y_hat, y)
        self.log_dict(
            {"val_loss": loss, "val_acc": acc, "val_f1": f1},
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.test_step_pred.append(y_hat)
        self.test_step_ground_truth.append(y)
        acc, f1 = self.get_metrics(y_hat, y)
        self.log_dict(
            {"test_loss": loss, "test_acc": acc, "test_f1": f1},
            on_epoch=True, prog_bar=True, logger=True
        )
        return {"test_loss": loss, "test_acc": acc, "test_f1": f1}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.max_epochs, 0
        )
        return [optimizer], [scheduler]
def LOSO(test_idx: list, subjects: list, experiment_ID, logs_name, args):
    pl.seed_everything(seed=args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    # load test data
    load_path = osp.join(args.load_path, 'data_{}_{}_{}'.format(args.data_format, args.dataset, args.label_type))
    data_test, label_test = load_data(load_path=load_path, load_idx=test_idx, concat=True)
    # load training data
    train_idx = [item for item in subjects if item not in test_idx]
    data_train, label_train = load_data(load_path=load_path, load_idx=train_idx, concat=True)
    train_idx, val_idx = get_validation_set(train_idx=np.arange(data_train.shape[0]), val_rate=args.val_rate, shuffle=True)
    data_val, label_val = data_train[val_idx], label_train[val_idx]
    data_train, label_train = data_train[train_idx], label_train[train_idx]
    # normalize the data
    data_train, data_val, data_test = normalize(train=data_train, val=data_val, test=data_test)

    print('Train:{} Val:{} Test:{}'.format(data_train.shape, data_val.shape, data_test.shape))
    # reorder the data for some models, e.g. TSception, LGGNet
    #idx, _ = get_channel_info(load_path=load_path, graph_type=args.graph_type)
    # prepare dataloaders
    train_loader = prepare_data_for_training(data=data_train, label=label_train, batch_size=args.batch_size,
                                             shuffle=True)
    val_loader = prepare_data_for_training(data=data_val, label=label_val, batch_size=args.batch_size,
                                           shuffle=False)
    test_loader = prepare_data_for_training(data=data_test, label=label_test, batch_size=1000,
                                            shuffle=False)
    # train and test the model
    model = DLModel(config=args)
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode='max')
    ensure_path(args.save_path)
    logger = TensorBoardLogger(save_dir=args.save_path, version=experiment_ID, name=logs_name)
    # most basic trainer, uses good defaults (1 gpu)
    if args.mixed_precision:
        trainer = pl.Trainer(
            accelerator="gpu", devices=[args.gpu], max_epochs=args.max_epoch, logger=logger,
            callbacks=[checkpoint_callback], precision='16-mixed'
        )
    else:
        trainer = pl.Trainer(
            accelerator="gpu", devices=[args.gpu], max_epochs=args.max_epoch, logger=logger,
            callbacks=[checkpoint_callback]
        )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best_val_metrics = trainer.checkpoint_callback.best_model_score.item()
    results = trainer.test(ckpt_path="best", dataloaders=test_loader)
    results[0]['best_val'] = best_val_metrics
    return results


parser = ArgumentParser()
parser.add_argument('--full-run', type=int, default=1, help='If it is set as 1, you will run LOSO on the same machine.')
parser.add_argument('--test-sub', type=int, default=0, help='If full-run is set as 0, you can use this to leave this '
                                                            'subject only. Then you can divided LOSO on different'
                                                            ' machines')
######## Data ########
parser.add_argument('--dataset', type=str, default='FATIG')
parser.add_argument('--subjects', type=int, default=11)
parser.add_argument('--num-class', type=int, default=2, choices=[2, 3, 4])
parser.add_argument('--label-type', type=str, default='FTG')
parser.add_argument('--num-chan', type=int, default=30) # 24 for TSception
parser.add_argument('--num-time', type=int, default=384)
parser.add_argument('--segment', type=int, default=4, help='segment length in seconds')
parser.add_argument('--trial-duration', type=int, default=60, help='trial duration in seconds')
parser.add_argument('--overlap', type=float, default=0)
parser.add_argument('--sampling-rate', type=int, default=128)
parser.add_argument('--data-format', type=str, default='eeg')
######## Training Process ########
parser.add_argument('--random-seed', type=int, default=2023)
parser.add_argument('--max-epoch', type=int, default=200)
parser.add_argument('--additional-epoch', type=int, default=20)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--val-rate', type=float, default=0.2)

parser.add_argument('--save-path', default='./save/') # change this
parser.add_argument('--load-path', default='C:/Users/SLL/dataset/') # change this
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--mixed-precision', type=int, default=0)
######## Model Parameters ########
parser.add_argument('--model', type=str, default='IFNet')



args = parser.parse_args()
all_sub_list = [0, 4, 21, 30, 34, 40, 41, 42, 43, 44, 52]



if args.full_run:
    sub_to_run = all_sub_list
else:
    sub_to_run = [args.test_sub]

logs_name = 'logs_{}_{}'.format(args.dataset, args.model)
for sub in sub_to_run:
    results = LOSO(
        test_idx=[sub], subjects=all_sub_list,
        experiment_ID='sub{}'.format(sub), args=args, logs_name=logs_name
    )
    log_path = os.path.join(args.save_path, logs_name, 'sub{}'.format(sub))
    ensure_path(log_path)
    log2csv(os.path.join(log_path, 'result.csv'), results[0])