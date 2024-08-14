import argparse
from tqdm import tqdm
import numpy as np
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import torch.nn as nn

from model import VaDE, cluster_acc
from dataloader import *


if __name__ == '__main__':

    parse = argparse.ArgumentParser(description='VaDE')
    parse.add_argument('--batch_size', type=int, default=800)
    parse.add_argument('--datadir', type=str, default='dataset/mnist/')
    parse.add_argument('--nClusters', type=int, default=10)

    parse.add_argument('--hid_dim', type=int, default=10)
    parse.add_argument('--cuda', type=bool, default=True)
    parse.add_argument('--device', type=str, default='cuda:1')

    args = parse.parse_args()

    DL, _ = get_mnist(args.datadir, args.batch_size)
    vade = VaDE(args)
    if args.cuda:
        vade = vade.to(args.device)
    vade.pre_train(DL, pre_epoch=50)

    opti = Adam(vade.parameters(), lr=2e-3)
    lr_s = StepLR(opti, step_size=10, gamma=0.95)

    writer = SummaryWriter('pytorch/logs')

    epoch_bar = tqdm(range(300))

    for epoch in epoch_bar:
        L = 0
        for x, _ in DL:
            if args.cuda:
                x = x.to(args.device)
            loss = vade.ELBO_loss(x)
            opti.zero_grad()
            loss.backward()
            opti.step()
            L += loss.detach().cpu().numpy()
        
        lr_s.step()

        pre = []
        tru = []

        with torch.no_grad():
            for x, y in DL:
                if args.cuda:
                    x = x.to(args.device)
                
                tru.append(y.numpy())
                pre.append(vade.predict(x))

        tru = np.concatenate(tru, 0)
        pre = np.concatenate(pre, 0)

        writer.add_scalar('loss',L/len(DL), epoch)
        writer.add_scalar('acc',cluster_acc(pre,tru)[0]*100, epoch)
        writer.add_scalar('lr',lr_s.get_last_lr()[0], epoch)

        epoch_bar.write('Loss={:.4f},ACC={:.4f}%,LR={:.4f}'.format(L/len(DL), cluster_acc(pre,tru)[0]*100, lr_s.get_lr()[0]))



