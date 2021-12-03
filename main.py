import json
import sys
import os
import os.path
import math
import argparse
import numpy as np
import config
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils.load_signals import PrepData
from utils.prep_data import train_val_loo_split, train_val_test_split
from sklearn.metrics import roc_curve, auc,accuracy_score, confusion_matrix, precision_score, recall_score,roc_auc_score
from transformer import Transformer
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass


def main(args):
    print('Main')
    dataset = args.dataset
    build_type = args.mode
    with open('SETTINGS_%s.json' %dataset) as f:
        settings = json.load(f)

    makedirs(str(settings['cachedir']))
    makedirs(str(settings['resultdir']))

    if settings['dataset']=='Kaggle2014Pred':
        targets = [
            'Dog_1',
            'Dog_2',
            'Dog_3',
            'Dog_4',
            'Dog_5',
            'Patient_1',
            'Patient_2'
        ]
    elif settings['dataset']=='FB':
        targets = [
            '1',
            '3',
            #'4',
            #'5',
            '6',
            '13',
            '14',
            '15',
            '16',
            '17',
            '18',
            '19',
            '20',
            '21'
        ]
    else:  #CHB-MIT
        targets = [
            # '1',
            # '2',
            # '3',
            # '5',
            # '9',
            '10',
            # '13',
            # '14',
            # '18',
            # '19',
            # '20',
            # '21',
            # '23'
        ]

    for target in targets:  # x:data [Channel, Time, Freq] y:label
        ictal_X, ictal_y = \
            PrepData(target, type='ictal', settings=settings).apply()
        interictal_X, interictal_y = \
            PrepData(target, type='interictal', settings=settings).apply()

        if build_type=='cv':
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")

            if os.path.exists("./weights") is False:
                os.makedirs("./weights")

            # tb_writer = SummaryWriter()
            loo_folds = train_val_loo_split(ictal_X, ictal_y, interictal_X, interictal_y, 0.25)
            ind = 1
            AUC = []
            Sens = []
            FP = []
            InterictalTime = []
            for X_train, y_train, X_val, y_val, X_test, y_test in loo_folds:
                print(X_train.shape, y_train.shape,
                      X_val.shape, y_val.shape,
                      X_test.shape, y_test.shape)
                # 实例化数据集
                batch_size = args.batch_size
                nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
                print('Using {} dataloader workers every process'.format(nw))
                train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                                          batch_size=batch_size,
                                          shuffle=True,
                                          pin_memory=True,
                                          num_workers=nw)

                val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                                        batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=True,
                                        num_workers=nw)

                model = Transformer(num_classes=2, num_channels=X_train.shape[1], frequency_size=X_train.shape[2]).to(device)
                dir = './runs/patient_%s/index_%d' %(target, ind)
                tb_writer = SummaryWriter(dir)
                # if args.weights != "":
                #     assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
                #     weights_dict = torch.load(args.weights, map_location=device)
                #     # 删除不需要的权重
                #     del_keys = ['head.weight', 'head.bias'] if model.has_logits \
                #         else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
                #     for k in del_keys:
                #         del weights_dict[k]
                #     print(model.load_state_dict(weights_dict, strict=False))
                #
                # if args.freeze_layers:
                #     for name, para in model.named_parameters():
                #         # 除head, pre_logits外，其他权重全部冻结
                #         if "head" not in name and "pre_logits" not in name:
                #             para.requires_grad_(False)
                #         else:
                #             print("training {}".format(name))

                pg = [p for p in model.parameters() if p.requires_grad]
                optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
                # Scheduler https://arxiv.org/pdf/1812.01187.pdf
                lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
                scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

                for epoch in range(1, args.epochs+1):
                    # train
                    train_loss, train_acc = train_one_epoch(model=model,
                                                            optimizer=optimizer,
                                                            data_loader=train_loader,
                                                            device=device,
                                                            epoch=epoch)

                    scheduler.step()

                    # validate
                    val_loss, val_acc = evaluate(model=model,
                                                 data_loader=val_loader,
                                                 device=device,
                                                 epoch=epoch)

                    tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
                    tb_writer.add_scalar(tags[0], train_loss, epoch)
                    tb_writer.add_scalar(tags[1], train_acc, epoch)
                    tb_writer.add_scalar(tags[2], val_loss, epoch)
                    tb_writer.add_scalar(tags[3], val_acc, epoch)
                    tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
                    if not(epoch % 10):
                        torch.save(model.state_dict(), "./weights/model-pat{}-cv{}-{}.pth".format(target, ind, epoch))

                test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                                         batch_size=batch_size,
                                         pin_memory=True,
                                         num_workers=nw)

                test_loss, test_acc, pred_cls, pred = test(model=model,
                                             data_loader=test_loader,
                                             device=device,
                                             index=ind)

                # Postprocess.
                pred_cls = pred_cls.cpu().numpy()
                # print(pred_cls.shape, 'pred_cls.shape')
                alarm = k_of_n(pred_cls, k=8*59, n=10*59)
                # y_test = y_test[1::59]
                tp = 0
                fp = 0
                print(len(alarm), 'alarm')
                for i in alarm:
                    if y_test[i] == 1:
                        tp = tp + 1
                    if y_test[i] == 0:
                        fp = fp + 1

                pred = pred.cpu().numpy()
                # print(y_test.shape, pred.shape, 'y_test.shape, pred.shape', y_test[:20])
                # print(pred[1])
                auc = roc_auc_score(y_test, pred[:, 1])
                sens = recall_score(y_test, pred_cls)
                # sens = (tp / (tp + fp))
                print('Patient: {} Index: {} tp: {} fp: {} auc: {:.6f} sens: {:.6f}'.format(target, ind, tp, fp, auc, sens))
                ind += 1
                Sens.append(sens)
                FP.append(fp)
                AUC.append(auc)
                InterictalTime.append((len(y_test) - sum(y_test))//59)

            # Compute the metrics.
            Sens_std = np.std(Sens) / ind
            Sens_mean = np.mean(Sens)
            Sens = [round(sens, 6) for sens in Sens]
            TotalInterictalTime = sum(InterictalTime) / 120     # window=30s /120 = per hour
            FPR = sum(FP) / TotalInterictalTime
            FPR_std = np.std(FP) / (ind * TotalInterictalTime)
            AUC = np.mean(AUC)
            print('Patient: {} Sens: {} Sens_std: {:.6f} Sens_mean: {:.6f} Fpr: {:.6f} FPR_std: {:.6f} AUC: {:.6f}'.format(target, Sens, Sens_std, Sens_mean, FPR, FPR_std, AUC))
            # print("Patient:", target)
            # print('Sens:', Sens, Sens_std)
            # print('Fpr:', FPR, FPR_std)
            # print('AUC:', AUC)


        # elif build_type=='test':
        #     X_train, y_train, X_val, y_val, X_test, y_test = \
        #         train_val_test_split(ictal_X, ictal_y, interictal_X, interictal_y, 0.25, 0.35)
        #     model = ConvNN(target,batch_size=32,nb_classes=2,epochs=100,mode=build_type)
        #     model.setup(X_train.shape)
        #     #model.fit(X_train, y_train)
        #     fn_weights = "weights_%s_%s.h5" %(target, build_type)
        #     if os.path.exists(fn_weights):
        #         model.load_trained_weights(fn_weights)
        #     else:
        #         model.fit(X_train, y_train, X_val, y_val)
        #     model.evaluate(X_test, y_test)


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    # train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size, shuffle = True)
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    train_loss = torch.zeros(1).to(device)  # 训练累计损失
    train_acc_num = torch.zeros(1).to(device)  # 训练累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        X_train, y_train = data
        # print(X_train.shape, y_train.shape)
        # X_train = np.transpose(X_train, (1, 0, 2))  # [Channel, Time, Freq] -> [Time, Channel, Freq]
        sample_num += X_train.shape[0]
        # print('sample_num :', sample_num)

        pred = model(X_train.to(device=device, dtype=torch.float32))
        pred_classes = torch.max(pred, dim=1)[1]
        train_acc_num += torch.eq(pred_classes, y_train.to(device=device)).sum()

        loss = loss_function(pred, y_train.to(device=device))
        loss.backward()
        train_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                train_loss.item() / (step + 1),
                                                                                train_acc_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return train_loss.item() / (step + 1), train_acc_num.item() / sample_num

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        X_val, y_val = data
        # print(X_val.shape, y_val.shape)
        # [Time, Channel, Freq]
        # X_val = np.transpose(X_val, (1, 0, 2))
        sample_num += X_val.shape[0]

        pred = model(X_val.to(device=device, dtype=torch.float32))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, y_val.to(device=device)).sum()

        loss = loss_function(pred, y_val.to(device=device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                           accu_loss.item() / (step + 1),
                                                                           accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def test(model, data_loader, device, index):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    Pred = []
    Cls = []
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        X_test, y_test = data
        # print(X_test.shape, y_test.shape)
        # [Time, Channel, Freq]
        # X_val = np.transpose(X_val, (1, 0, 2))
        sample_num += X_test.shape[0]

        pred = model(X_test.to(device=device, dtype=torch.float32))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, y_test.to(device=device)).sum()

        loss = loss_function(pred, y_test.to(device=device))
        accu_loss += loss

        data_loader.desc = "[Test index {}] loss: {:.3f}, acc: {:.3f}".format(index,
                                                                           accu_loss.item() / (step + 1),
                                                                           accu_num.item() / sample_num)
        Pred.append(torch.nn.functional.softmax(pred, dim=1))
        Cls.append(pred_classes)


    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, torch.cat(Cls), torch.cat(Pred)


def k_of_n(pred_cls, k, n):
    alarm = []
    i = 0
    nn = []
    while i < len(pred_cls):
        nn.append(pred_cls[i])
        if np.sum(nn) >= k:
            alarm.append(i)
            nn = []
            i = i + 60*59  # Refractory Period: 30min

        if len(nn) >= n:
            nn.pop(0)
        i = i + 1
    return alarm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="cv or test. cv is for leave-one-out cross-validation")
    parser.add_argument("--dataset", help="FB, CHBMIT or Kaggle2014Pred")
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=59)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--device', default='cuda:2', help='device id (i.e. 0 or 0,1 or cpu)')
    args = parser.parse_args()
    assert args.mode in ['cv','test']
    main(args)

# def test(args):
#     tp_num = torch.zeros(1).to(device)   # 累计预测正确的正样本数
#     fp_num = torch.zeros(1).to(device)  # 累计预测错误的负样本数
#     fn_num = torch.zeros(1).to(device)  # 累计预测错误的正样本数
#     tp_num += torch.eq(torch.add(pred_classes, y_train.to(device)), 2).sum()  # pred+label = 2 的为TP
#     fp_num += torch.eq(torch.eq(pred_classes, 1), torch.eq(y_train.to(device), 0)).sum()  # label=0且pred=1的为FP
#     fn_num += torch.eq(torch.eq(pred_classes, 0), torch.eq(y_train.to(device), 1)).sum()  # label=1且pred=0的为FN