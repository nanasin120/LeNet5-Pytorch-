import torch
import torch.nn as nn
import torch.nn.functional as F

class SubSamplingLayer(nn.Module):
    def __init__(self, in_channels):
        super(SubSamplingLayer, self).__init__()

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.weight = nn.Parameter(torch.ones(in_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(in_channels, 1, 1))

    def forward(self, x):
        x = self.pool(x)
        x = x * 4
        x = x * self.weight + self.bias
        return torch.tanh(x)

class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()

        self.conv = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size=5)
    
    def forward(self, x):
        x = self.conv(x)
        return torch.tanh(x)

class C3(nn.Module):
    def __init__(self):
        super(C3, self).__init__()

        self.table = [
            [0, 1, 2], 
            [1, 2, 3], 
            [2, 3, 4], 
            [3, 4, 5], 
            [0, 4, 5], 
            [0, 1, 5],
            [0, 1, 2, 3], 
            [1, 2, 3, 4], 
            [2, 3, 4, 5], 
            [0, 3, 4, 5], 
            [0, 1, 4, 5], 
            [0, 1, 2, 5],
            [0, 1, 3, 4], 
            [1, 2, 4, 5], 
            [0, 2, 3, 5],
            [0, 1, 2, 3, 4, 5]
        ]

        self.convs = nn.ModuleList()

        for i in range(16):
            self.convs.append(nn.Conv2d(in_channels = len(self.table[i]), out_channels=1, kernel_size=5))
    
    def forward(self, x):
        outputs = []
        for i, idx in enumerate(self.table):
            inputs = x[:, idx, :, :]
            outputs.append(self.convs[i](inputs))
        return torch.tanh(torch.cat(outputs, dim=1))

class C5(nn.Module):
    def __init__(self):
        super(C5, self).__init__()

        self.conv = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
    
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return torch.tanh(x)

class F6(nn.Module):
    def __init__(self):
        super(F6, self).__init__()

        self.fc = nn.Linear(in_features=120, out_features=84)
        self.A = 1.7159
        self.S = 2.0/3.0

    def forward(self, x):
        a = self.fc(x)
        return self.A * torch.tanh(self.S * a)

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.C1 = C1()

        self.S2 = SubSamplingLayer(in_channels=6)

        self.C3 = C3()

        self.S4 = SubSamplingLayer(in_channels=16)

        self.C5 = C5()

        self.F6 = F6()

        self.output = nn.Linear(84, 10)

    def forward(self, x):
        x = self.C1(x)
        
        x = self.S2(x)
        
        x = self.C3(x)
        
        x = self.S4(x)
        
        x = self.C5(x)
        
        x = self.F6(x)

        output = self.output(x)

        return output

if __name__ == "__main__":
    model = LeNet5()
    test_input = torch.randn(1, 1, 32, 32) # LeNet-5는 32x32 입력 기준
    out = model(test_input)
    print(f"출력 형태: {out.shape}") # torch.Size([1, 10])
