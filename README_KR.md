# [논문 구현] LeNet-5: 1998년 원형을 최대한 비슷하게 PyTorch로 재현

[English Version](./README.md)

이 프로젝트는 Yann LeCun의 1998년 논문 *"Gradient-Based Learning Applied to Document Recognition"*에 명시된 **LeNet-5** 아키텍처를 PyTorch로 충실히 재현한 프로젝트입니다.

## 프로젝트 구성

*"Gradient-Based Learning Applied to Document Recognition"*에 명시된 **LeNet-5** 아키텍처를 PyTorch로 최대한 그대로 구현한 프로젝트입니다.

## 프로젝트 준비

파이썬 환경은 아나콘다를 이용해 torch, torchvision, tqdm, tensorboard 패키지를 설치해 사용했습니다.
```
conda install torch torchvision tqdm tensorboard
```

## 모델 소개

LeNet-5모델은 논문안에 있는 내용과 최대한 비슷하게 구현했습니다.

논문에서는 input layer와 6개의 hidden layer, 마지막 output layer로 LeNet-5를 구성하고 있습니다.

논문에서 입력되는 데이터는 1x32x32의 손글씨 데이터입니다.

<img width="348" height="768" alt="image" src="https://github.com/user-attachments/assets/8ca63f53-b81a-484c-9088-629f7535a09f" />

최종 모델의 형태는 위와 같습니다.

---

### C1 Layer

C1 layer는 입력 직후에 이용되는 레이어입니다.
```
class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()

        self.conv = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size=5)
    
    def forward(self, x):
        x = self.conv(x)
        return torch.tanh(x)
```
맨 처음 들어오는 데이터는 1x32x32의 데이터입니다. 

1x32x32데이터를 6x28x28로 만들어줍니다.

C1 Output: (Batch, 6, 28, 28)
---

### S2 Layer

S2 Layer는 C1 Layer 다음으로 이용되는 레이어입니다. 이 레이어는 sub-sampling layer입니다.
```
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
```
C1 Layer의 결과는 6x28x28크기의 데이터입니다. 여기서 잠시 채널은 빼두고 28x28만 생각하도록 하겠습니다.

28x28에서 2x2크기의 피처맵을 이용해 값들을 모두 더합니다. 이는 겹치는 부분이 없어야 합니다. 이렇게 되면 28x28에서 14x14가 됩니다.

여기에 학습이 가능한 가중치와 학습이 가능한 편향을 곱하고 더해줍니다.

마지막으로 tanh를 적용해주면 끝입니다.

이 코드에서는 AvgPool2d의 결과에 4를 곱함으로 **"2x2의 값을 모두 더한다"** 를 적용시켰습니다.

그리고 nn.Parameter로 학습 가능한 가중치와 편향을 만들어냈습니다.
```
self.S2 = SubSamplingLayer(in_channels=6)
```
실제 Model 클래스에서는 위 처럼 구현해줬습니다.

S2 Output: (Batch, 6, 14, 14)

---

### C3 Layer

C3 Layer는 S2 Layer 다음으로 이용되는 레이어 입니다. 이 부분이 가장 골치 아픕니다.

<img width="534" height="275" alt="image" src="https://github.com/user-attachments/assets/a546bdac-ea84-45b3-af81-3111691842e5" />
<p align="center"><em>Table 1: Mapping between S2 and C3. (Source: LeCun et al., 1998))</em></p>

위의 표시되어있는 X부분만 연결이 됩니다.

현재 S2 Layer의 output은 6x14x14, 즉 채널이 6인 상태입니다. table에 따르면 0번 채널은 0, 4, 5, 6, 9, 10, 11, 12, 14, 15와만 연결됩니다.

```
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
```
이를 구현하기 위해 nn.ModuleList()를 이용했습니다.

nn.ModuleList()안에는 여러 파이토치 모듈을 넣을수가 있습니다.

이를 통해 C3 Layer의 output채널인 16개의 Conv2d를 넣어주고 각자에 맞는 채널만 연결해줬습니다.

C3 Output: (Batch, 16, 10, 10)

---

### S4 Layer

S4 Layer는 C3 Layer 다음에 이용되는 레이어 입니다. S 레이어는 모두 sub-sampling이므로 
```
self.S4 = SubSamplingLayer(in_channels=16)
```
위의 SubSamplingLayer 클래스를 이용해 간단하게 구현해줬습니다.

S4 Output: (Batch, 16, 5, 5)

---

### C5 Layer

C5 Layer는 S4 Layer 다음으로 이용되는 레이어 입니다.
```
class C5(nn.Module):
    def __init__(self):
        super(C5, self).__init__()

        self.conv = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
    
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return torch.tanh(x)
```
이 레이어에서는 일렬로 쭉 펴주는 작업을 해줍니다.

이전 S4 Layer의 output의 크기는 5x5입니다. 여기에 커널 사이즈5를 적용하면 크기는 1이 됩니다.

그래서 C5 Layer의 output은 120x1x1이 됩니다.

여기에 flatten(x, 1)를 적용해 120x1x1을 120으로 만들어줍니다.

C5 Output: (Batch, 120)

---

### F6 Layer

F6 Layer는 C5 Layer 다음으로 이용되는 레이어 입니다.
```
class F6(nn.Module):
    def __init__(self):
        super(F6, self).__init__()

        self.fc = nn.Linear(in_features=120, out_features=84)
        self.A = 1.7159
        self.S = 2.0/3.0

    def forward(self, x):
        a = self.fc(x)
        return self.A * torch.tanh(self.S * a)
```
120의 데이터를 84로 줄여주며 A와 S값을 이용해 tanh를 적용해줍니다.

A=1.7159와 S=2.0/3.0은 활성화 함수가 가장 비선형적인 구간에서 연산이 이루어지게 하여 학습 속도를 최적화하기 위한 상수입니다.

F6 Output: (Batch, 84)

---

### output Layer

output Layer는
```
self.output = nn.Linear(84, 10)
```
간단하게 Linear로 구현했습니다.

Output: (Batch, 10)

---

## 학습 결과

<img width="513" height="368" alt="image" src="https://github.com/user-attachments/assets/0b323b88-ae28-4c90-a83c-a19ce55c1127" />

<img width="1028" height="371" alt="image" src="https://github.com/user-attachments/assets/e86e6433-b7bd-446e-81cf-fd1065e3d1a7" />

Epochs를 10으로 돌린 결과, 손실은 0.0172, 정확도는 98.6%가 나왔습니다.

