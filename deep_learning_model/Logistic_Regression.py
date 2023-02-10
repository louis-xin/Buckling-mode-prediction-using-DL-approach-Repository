import numpy as np
import sklearn
from sklearn.preprocessing import *
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

data_dir = r'D:\Desktop\Abaqus-python\CollectedData\Data'

data = np.load('{}\Data_1-520.npy'.format(data_dir))

Radius = data[:,0]
Length = data[:,1]
Thickness = data[:,2]
BPangle = data[:,3]

R_t = np.zeros(520)
RtoL = np.zeros(520)

for i in range(520):
    R_t[i] = Radius[i] / Thickness[i]
    RtoL[i] = Radius[i] / Length[i]
    
bucklingmode = np.genfromtxt(r'D:\Desktop\Abaqus-python\DL\buckle.txt') # 1 denotes global buckling and 0 denots local buckling

#%% Standardization

std_radius = (Radius - np.mean(Radius))/np.std(Radius)
std_length = (Length - np.mean(Length))/np.std(Length)
std_thickness = (Thickness - np.mean(Thickness))/np.std(Thickness)
std_BP = (BPangle - np.mean(BPangle))/np.std(BPangle)
std_R_t = (R_t - np.mean(R_t))/np.std(R_t)
std_RtoL = (RtoL - np.mean(RtoL))/np.std(RtoL)

std_data = np.zeros((520,6))
std_data[:,0] = std_radius
std_data[:,1] = std_length
std_data[:,2] = std_thickness
std_data[:,3] = std_BP
std_data[:,4] = std_R_t
std_data[:,5] = std_RtoL

#%% Logistic regression model

inputs = std_data
targets = bucklingmode

inputs = torch.from_numpy(inputs).float()
targets = torch.from_numpy(targets).float().unsqueeze(1)

# train_ds = TensorDataset(inputs, targets)

model = nn.Sequential(
   nn.Linear(6, 1), 
   nn.Sigmoid()
)

optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    hypothesis = model(inputs)

    cost = F.binary_cross_entropy(hypothesis, targets)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 50 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5]) # 예측값이 0.5를 넘으면 True로 간주
        correct_prediction = prediction.float() == targets # 실제값과 일치하는 경우만 True로 간주
        accuracy = correct_prediction.sum().item() / len(correct_prediction) # 정확도를 계산
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format( # 각 에포크마다 정확도를 출력
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))

print(list(model.parameters()))
















