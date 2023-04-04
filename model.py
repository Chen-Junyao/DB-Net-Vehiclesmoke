import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

pretrained_net=models.resnet50(pretrained=False)
num_classes=2

class DB_Net(nn.Module):
    def __init__(self, num_classes):
        super(DB_Net, self).__init__()

        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-7]) #1/2
        self.stage2 =  nn.Sequential(*list(pretrained_net.children())[3:5:]) #1/4
        self.stage3 = list(pretrained_net.children())[5] # 1/8
        self.stage4 = list(pretrained_net.children())[-4]  #1/16

        self.scores1 = nn.Conv2d(64, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(512, num_classes, 1)
        self.scores4 = nn.Conv2d(1024, num_classes, 1)

    def forward(self, x):
        x = self.stage1(x)
        s1 = x  # 1/2

        x = self.stage2(x)
        s2 = x  # 1/4

        x = self.stage3(x)
        s3 = x  # 1/8

        x = self.stage4(x)
        s4 = x  # 1/16

        s1 = self.scores1(s1)
        s1 = F.interpolate(s1, size=s2.shape[2:4])

        s2 = self.scores2(s2)
        s2 = s2 + s1
        s2 = F.interpolate(s2, size=s3.shape[2:4])

        s3 = self.scores3(s3)
        s3 = s3 + s2
        s3 = F.interpolate(s3, size=s4.shape[2:4])

        s4 = self.scores4(s4)
        s4 = s4 + s3

        return s4
