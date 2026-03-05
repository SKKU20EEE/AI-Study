#trian.py

import os
import time
import copy
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import png
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from colormap.colors import Color, hex2rgb
from sklearn.metrics import average_precision_score as ap_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

from dataset import FacadeDataset

# Fix Random seed for reproducibility
def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


N_CLASS = 5

# =========================
#  U-Net base line
# =========================
class DoubleConv(nn.Module):
    """
    (Conv2d -> BatchNorm -> ReLU -> Dropout) x 2
    spatial size 유지 (padding=1)
    """
    def __init__(self, in_ch, out_ch, p=0.2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout2d(p=p)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout2d(p=p)

    def forward(self, x):
        x = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        x = self.drop2(self.relu2(self.bn2(self.conv2(x))))
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n_class = N_CLASS

        # -------- Encoder --------
        self.enc0 = DoubleConv(3, 64)        # (B, 64, 256, 256)
        self.pool0 = nn.MaxPool2d(2)         # (B, 64, 128, 128)

        self.enc1 = DoubleConv(64, 128)      # (B, 128, 128, 128)
        self.pool1 = nn.MaxPool2d(2)         # (B, 128, 64, 64)

        self.enc2 = DoubleConv(128, 256)     # (B, 256, 64, 64)
        self.pool2 = nn.MaxPool2d(2)         # (B, 256, 32, 32)

        self.enc3 = DoubleConv(256, 512)     # (B, 512, 32, 32)
        self.pool3 = nn.MaxPool2d(2)         # (B, 512, 16, 16)

        # -------- Bottleneck --------
        self.bottleneck = DoubleConv(512, 1024)  # (B, 1024, 16, 16)

        # -------- Decoder --------
        self.up3  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512 + 512, 512)

        self.up2  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256 + 256, 256)

        self.up1  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128 + 128, 128)

        self.up0  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec0 = DoubleConv(64 + 64, 64)

        # -------- Output --------
        self.out_conv = nn.Conv2d(64, self.n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.enc0(x)            # (B, 64, 256, 256)
        p0 = self.pool0(x0)          # (B, 64, 128, 128)

        x1 = self.enc1(p0)           # (B, 128, 128, 128)
        p1 = self.pool1(x1)          # (B, 128, 64, 64)

        x2 = self.enc2(p1)           # (B, 256, 64, 64)
        p2 = self.pool2(x2)          # (B, 256, 32, 32)

        x3 = self.enc3(p2)           # (B, 512, 32, 32)
        p3 = self.pool3(x3)          # (B, 512, 16, 16)

        # Bottleneck
        b = self.bottleneck(p3)      # (B, 1024, 16, 16)

        # Decoder
        d3 = self.up3(b)             # (B, 512, 32, 32)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)            # (B, 256, 64, 64)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)            # (B, 128, 128, 128)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)

        d0 = self.up0(d1)            # (B, 64, 256, 256)
        d0 = torch.cat([d0, x0], dim=1)
        d0 = self.dec0(d0)

        logits = self.out_conv(d0)   # (B, N_CLASS, 256, 256)
        return logits


class FocalLossMultiClass(nn.Module):
    def __init__(self, weight=None, ignore_index=255, gamma=1.5):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits: (N,C,H,W)
        targets: (N,H,W)
        """
        N, C, H, W = logits.shape

        logits = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C)
        targets = targets.view(-1) 

        valid = (targets != self.ignore_index)
        logits = logits[valid]
        targets = targets[valid]

        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        ce = F.nll_loss(
            log_probs,
            targets,
            reduction='none',
            weight=self.weight,
        )

        pt = probs[torch.arange(len(targets), device=logits.device), targets]
        focal_weight = (1 - pt) ** self.gamma

        loss = focal_weight * ce
        return loss.mean()


def save_label(label, path):
    colormap = [
        '#000000',
        '#0080FF',
        '#80FF80',
        '#FF8000',
        '#FF0000',
    ]
    assert(np.max(label) < len(colormap))
    colors = [hex2rgb(color, normalise=False) for color in colormap]
    w = png.Writer(label.shape[1], label.shape[0], palette=colors, bitdepth=4)
    with open(path, 'wb') as f:
        w.write(f, label)


def train(trainloader, net, criterion, optimizer, device, epoch):
    start = time.time()
    running_loss = 0.0
    net.train()
    for images, labels in tqdm(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    end = time.time()
    epoch_loss = running_loss / len(trainloader)
    print('[epoch %d] loss: %.3f elapsed time %.3f' %
          (epoch, epoch_loss, end-start))
    return epoch_loss


def test(testloader, net, criterion, device):
    losses = 0.
    cnt = 0
    net.eval()
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            loss = criterion(output, labels)
            losses += loss.item()
            cnt += 1
    avg_loss = losses / cnt
    print(avg_loss)
    return avg_loss


def cal_AP(testloader, net, criterion, device):
    with torch.no_grad():
        net = net.eval()
        preds = [[] for _ in range(N_CLASS)]
        heatmaps = [[] for _ in range(N_CLASS)]
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)  # (B,C,H,W) one-hot

            logits = net(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # (B,C,H,W)
            labels_np = labels.cpu().numpy()

            for c in range(N_CLASS):
                preds[c].append(probs[:, c].reshape(-1))
                heatmaps[c].append(labels_np[:, c].reshape(-1))

        for c in range(N_CLASS):
            preds[c] = np.concatenate(preds[c])
            heatmaps[c] = np.concatenate(heatmaps[c])
            if heatmaps[c].max() == 0:
                ap = float('nan')
            else:
                ap = ap_score(heatmaps[c], preds[c])
            print(f"Class {c} AP = {ap}")

    return None


def get_result(testloader, net, device, folder='output_train'):
    os.makedirs(folder, exist_ok=True)
    with torch.no_grad():
        net = net.eval()
        cnt = 0
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)

            logits = net(images)[0]                 # (C,H,W)
            probs = torch.softmax(logits, dim=0)    # (C,H,W)
            output = probs.cpu().numpy()
            c, h, w = output.shape
            assert(c == N_CLASS)

            y = np.argmax(output, axis=0).astype('uint8')  # (H,W)

            gt = labels.cpu().data.numpy().squeeze(0).astype('uint8')
            save_label(y, './{}/y{}.png'.format(folder, cnt))
            save_label(gt, './{}/gt{}.png'.format(folder, cnt))
            plt.imsave(
                './{}/x{}.png'.format(folder, cnt),
                ((images[0].cpu().data.numpy()+1)*128).astype(np.uint8).transpose(1, 2, 0)
            )

            cnt += 1


def main():
    set_seed(1234)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Use all train / test_dev images
    train_data  = FacadeDataset(flag='train', data_range=(0, 905), onehot=False)   # 905 train images
    test_data   = FacadeDataset(flag='test_dev', data_range=(0, 114), onehot=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)              # no shuffle for evaluation

    ap_data   = FacadeDataset(flag='test_dev', data_range=(0, 114), onehot=True)
    ap_loader = DataLoader(ap_data, batch_size=1, shuffle=False)                  # no shuffle for AP calculation

    name = 'UNet'

    # Hyperparameters
    batch_size_train = 8
    batch_size_val   = 4
    lr               = 1e-3
    weight_decay     = 1e-4
    max_epochs       = 100
    patience         = 10
    val_ratio        = 0.2

    print(f"Using device: {device}")

    # Split train dataset into train/validation
    num_total = len(train_data)
    num_val   = int(num_total * val_ratio)
    indices   = np.arange(num_total)

    val_idx   = indices[:num_val]
    train_idx = indices[num_val:]

    train_subset = Subset(train_data, train_idx)
    val_subset   = Subset(train_data, val_idx)

    print(f"Train size: {len(train_subset)}, Val size: {len(val_subset)}")

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size_train,
        shuffle=True,          # shuffle only for training
        num_workers=2,
        pin_memory=True        # If using GPU, speeds up host->GPU transfer by using pinned (page-locked) memory.
                               # On CPU-only training it gives little/no benefit
    )

    evaluation_loader = DataLoader(
        val_subset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=2,
        pin_memory=True        # Same reason as above (GPU data transfer efficiency).
    )

    # Model
    net = Net().to(device)

    # Loss: Focal loss to address class imbalance / hard pixels
    # Emphasize each Class for better classification
    ce_weights = torch.tensor([1.0, 1.0, 5.0, 1.5, 2.0], device=device)
    criterion  = FocalLossMultiClass(
        weight=ce_weights,
        ignore_index=255,
        gamma=1.5
    )

    # Optimizer: Adam + weight decay (L2 regularization)
    #   weight_decay weight of L2
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)  # ★

    # Scheduler: reduce lr when validation loss plateaus
    # When val loss doesn't improved, reduce lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    # Early stopping
    best_val_loss = float('inf')
    best_state    = None
    patience_cnt  = 0

    train_losses, val_losses, epochs_list = [], [], []

    print('\nStart training')

    for epoch in range(1, max_epochs + 1):
        print('-----------------Epoch = %d-----------------' % epoch)

        train_loss = train(train_loader, net, criterion, optimizer, device, epoch)
        val_loss = test(evaluation_loader, net, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epochs_list.append(epoch)

        # Save best model & early stopping
        # 1e-4  is minimum delta of validation loss
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(net.state_dict())
            patience_cnt = 0
            print(f'Validation improved: {best_val_loss:.4f}')
        else:
            patience_cnt += 1
            print(f'No improvement: early stopping {patience_cnt}/{patience}')

        scheduler.step(val_loss)

        if patience_cnt >= patience:
            print('Early stopping triggered')
            break

    os.makedirs("figures", exist_ok=True)

    plt.figure()
    plt.plot(epochs_list, train_losses, label="Train Loss")
    plt.plot(epochs_list, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: figures/loss_curve.png")

    # Load best model
    if best_state is not None:
        net.load_state_dict(best_state)
        print(f'val loss of final model: {best_val_loss:.4f}')

    print('\nFinished Training, Testing on test set')
    test(test_loader, net, criterion, device)

    print('\nGenerating Unlabeled Result')
    os.makedirs('output_test', exist_ok=True)
    get_result(test_loader, net, device, folder='output_test')

    os.makedirs('models', exist_ok=True)
    torch.save(net.state_dict(), './models/model_{}.pth'.format(name))

    cal_AP(ap_loader, net, criterion, device)


if __name__ == "__main__":
    main()