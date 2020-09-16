from torch import nn
import torch as t
from torch.nn import functional as F
from torch.autograd import Variable


class ResidualBlock(nn.Module):
    '''
    实现子module:Residual Block
    '''

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(  # F(x)
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        self.right = shortcut  # 如果shortcut不为None, shortcut = x; F(x) = H(x) - x；当F(X) = 0，则为恒等映射
        # 当输入和输出具有相同的维度时:y = F(x,{Wi}) + X  快捷恒等映射
        # 维度增加时:y = F(x,{Wi}) + WsX , Ws仅在匹配维度时使用。

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)   # X / WsX
        out += residual
        return F.relu(out)


class ResNet(nn.Module):  # 224x224x3
    '''
    实现主module:ResNet34
    ResNet34包含多个layer,每个layer又包含多个residual block
    用子module实现residual block,用_make_layer_函数实现layer
    '''

    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()

        # 前几层图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),  # (224+2*p-)/2(向下取整)+1，size减半->112
            nn.BatchNorm2d(64),  # 112x112x64
            # 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，
            # 这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定.
            # BatchNorm2d()内部的参数如下：
            # 1.num_features：一般输入参数为batch_size*num_features*height*width，
            # 即为其中特征的数量
            # 2.eps：分母中添加的一个值，目的是为了计算的稳定性，默认为：1e-5
            # 3.momentum：一个用于运行过程中均值和方差的一个估计参数
            # （我的理解是一个稳定系数，类似于SGD中的momentum的系数）
            # 4.affine：当设为true时，会给定可以学习的系数矩阵gamma和beta
            nn.ReLU(inplace=True),
            # 与nn.ReLU()计算结果一样，利用in-place计算可以节省内（显）存，同时还可以省去反复申请和释放内存的时间。但是会对原变量覆盖，只要不带来错误就用。
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1
            # kernel_size(int or tuple) - max pooling的窗口大小，可以为tuple，在nlp中tuple用更多，（n,1）
            # stride(int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
            # padding(int or tuple, optional) - 输入的每一条边补充0的层数
            # dilation(int or tuple, optional) – 一个控制窗口中元素步幅的参数
            # return_indices - 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助
            # ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作
        )

        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self._make_layer(64, 128, 3)  # 56x56x64,layer1层输入输出一样，make_layer里，应该不用对shortcut进行处理，但是为了统一操作。
        self.layer2 = self._make_layer(128, 256, 4, stride=2)  # 第一个stride=1,剩下3个stride=2;28x28x128
        self.layer3 = self._make_layer(256, 512, 6, stride=2)  # 14x14x25
        self.layer4 = self._make_layer(512, 512, 3, stride=2)  # 7x7x512

        # 分类用的全连接
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        '''
        构建layer,包含多个residual block
        '''
        # 当维度增加时，对shortcut进行option B的处理
        shortcut = nn.Sequential(  # 首个ResidualBlock需要进行option B处理，维度增加时:y = F(x,{Wi}) + WsX
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),  # 卷积核大小为1，1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
            nn.BatchNorm2d(outchannel)
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        # 如果输入和输出的通道数不一致，或其步长不为1，那么就需要有一个专门的单元将二者转成一致，使其可以相加。

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))  # 后面的几个ResidualBlock,shortcut直接相加

        return nn.Sequential(*layers)

    def forward(self, x):  # 224x224x3
        x = self.pre(x)  # 56x56x64

        x = self.layer1(x)  # 56x56x64 调用残差块
        x = self.layer2(x)  # 28x28x128 调用残差块
        x = self.layer3(x)  # 14x14x256 调用残差块
        x = self.layer4(x)  # 7x7x512 调用残差块

        x = F.avg_pool2d(x, 7)  # 1x1x512
        x = x.view(x.size(0), -1)  # 将输出拉伸为一行：1x512

        return self.fc(x)


model = ResNet()
print(model)
# input = t.autograd.Variable(t.randn(1, 3, 224, 224))
# o = model(input)
# print(o)
# pytorch配套的工具包torchvision已经实现了深度学习中大多数经典模型
# 其中包括ResNet34
# from torchvision import models
# model = model.resnet34()
