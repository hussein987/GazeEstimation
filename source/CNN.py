from torch import nn
from torch.nn import functional as F
from torchvision import models
import torch


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels1, labels2 = batch
        outs = self(images)
        loss1 = F.mse_loss(outs['label1'], labels1)
        loss2 = F.mse_loss(outs['label2'], labels2)
        loss = loss1 + loss2
        return loss

    def validation_step(self, batch):
        images, labels1, labels2 = batch
        outs = self(images)
        loss1 = F.mse_loss(outs['label1'], labels1)
        loss2 = F.mse_loss(outs['label2'], labels2)
        loss = loss1 + loss2
        return {'val_loss': loss.detach()}

    @staticmethod
    def validation_epoch_end(outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        return {'val_loss': epoch_loss.item()}

    @staticmethod
    def epoch_end(epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss']))


class CNN1(ImageClassificationBase):
    def __init__(self):
        super(CNN1, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        self.fc1 = nn.Linear(1280, 1)
        self.fc2 = nn.Linear(1280, 1)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        label1 = self.fc1(x)
        label1 = torch.reshape(label1, (1, -1))
        label2 = self.fc2(x)
        label2 = torch.reshape(label2, (1, -1))
        return {'label1': label1[0], 'label2': label2[0]}
