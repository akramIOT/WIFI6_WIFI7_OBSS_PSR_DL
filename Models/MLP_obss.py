# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 13:20:39 2022
DL based analysis  to derive data insights  for predicting the  Throughput  data for  different  sets of  count of  AP's, AP location, RSSI,floor plan map- distance
SINR, Airtime of contention"  is the focus of this ML project with  Simulated  WIFI datasets. IOT WIFI6 or WIFI7 AP's would require very deterministic performance and this  DL based project is done to study the  Throughput results observed.
@author: Akram Sheriff
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from random import randint
import torch
from torch.nn import Module, MSELoss, Linear, ReLU, PReLU, Sequential, BatchNorm1d

##Initializer

data_path = '/Users/akram/AKRAM_CODE_FOLDER/IOT_DL/MLP_OBSS_WIFI6/Data/'             ## Path to where Input data is stored
results_path = '/Users/akram/AKRAM_CODE_FOLDER/IOT_DL/MLP_OBSS_WIFI6/OUTPUT/'         ## Path where Output results will be stored
input_train = data_path+'Train/input_node_files/'                                    ##  Train Data
output_train_sim = data_path+'Train/output_simulator/'                               ##
input_test = data_path+'Test/input_node_files_test/'
output_test_sim = data_path+'Test/output_simulator_test/'

# Device configuration (choose GPU if it is available )
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## Loader Read Simulator
##

def read_output_simulator(fp, N_AP):
    line = fp.readline()  # Initial line (name of the scenaio)
    throughput = fp.readline()  # Throughput
    airTime = fp.readline()  # AirTime

    RSSI = fp.readline()  # RSSI
    interference = np.zeros((N_AP, N_AP))  # Interferences
    for i in range(N_AP):
        inter = fp.readline()
        if (i == 0):
            interference[0] = np.array(inter[1:len(inter) - 2].split(',')).astype(np.float)
        else:
            interference[i] = np.array(inter[:len(inter) - 2].split(',')).astype(np.float)
    SINR = fp.readline()
    SINR = np.array(SINR[1:len(SINR) - 2].split(',')).astype(np.float)
    list_of_SINR = np.split(SINR, np.where(SINR == np.inf)[0][1:])
    RSSI = np.array(RSSI[1:len(RSSI) - 2].split(',')).astype(np.float)
    list_of_RSSI = np.split(RSSI, np.where(RSSI == np.inf)[0][1:])
    airTime = np.array(airTime[1:len(airTime) - 3].split(';'))
    list_of_airTime = list()
    for i in range(N_AP):
        list_of_airTime.append(np.array(airTime[i].split(',')).astype(np.float))
    throughput = np.array(throughput[1:len(throughput) - 2].split(',')).astype(np.float)
    list_of_throughput = np.split(throughput, np.where(RSSI == np.inf)[0][1:])

    return (interference, list_of_RSSI, list_of_SINR, list_of_airTime, list_of_throughput)

##Read data

def load_info(input_nodes_sceXX_path, simulator_file_path, size, N_APs):
    features = np.zeros((size, N_APs, 15))
    target = np.zeros((size, N_APs, 1))

    ##############################################################
    ####################### Features #############################
    ##############################################################
    ## 0: SINR mean                                             ##
    ## 1: SINR std                                              ##
    ## 2: RSSI mean                                             ##
    ## 3: RSSI std                                              ##
    ## 4: Dist mean                                             ##
    ## 5: Dist std                                              ##
    ## 6: num of STAs                                           ##
    ## 7-14: Airtime x Channel                                  ##

    fp = open(simulator_file_path, 'r')
    for f in range(size):
        ##### READ OUTPUT FILE #####
        interferences, RSSI, SINR, list_of_airTime, throughput = read_output_simulator(fp,
                                                                                       N_APs)  # Interference [N_APs,N_APs],  list_RSSI(i)[0-20], list_Throughput(i)[0-20]
        ##### READ INPUT FILE ######
        csvData = pd.read_csv(input_nodes_sceXX_path + format(f, '03') + '.csv', sep=';',
                              usecols=['node_type', 'min_channel_allowed', 'max_channel_allowed'])
        channels_data = csvData.values
        APs_channels_data = channels_data[np.where(channels_data[:, 0] == 0)][:, 1:3]

        csvData = pd.read_csv(input_nodes_sceXX_path + format(f, '03') + '.csv', sep=';',
                              usecols=['node_type', 'x(m)', 'y(m)'])
        positions = csvData.values
        list_of_positions = np.split(positions, np.where(positions[:, 0] == 0)[0])[1:]

        #### Compute available channels ####
        APs_airTime = np.zeros((N_APs, 8))
        APs_openChan = np.zeros((N_APs, 8))
        for ap in range(N_APs):
            APs_airTime[ap, APs_channels_data[ap, 0]:APs_channels_data[ap, 1] + 1] = list_of_airTime[ap] / 100
            APs_openChan[ap, APs_channels_data[ap, 0]:APs_channels_data[ap, 1] + 1] = 1

        for i, thr in enumerate(throughput):
            n = len(thr[1:])
            distances = list_of_positions[i]
            dist = np.zeros((len(distances) - 1))
            for sta in range(1, len(distances)):
                dist[sta - 1] = distance.euclidean(distances[0][1:], distances[sta][1:])

            SINR[i][np.isnan(SINR[i])] = -9
            features[f, i, 0] = np.mean(SINR[i][1:])
            features[f, i, 1] = np.std(SINR[i][1:])
            features[f, i, 2] = np.mean(RSSI[i][1:])
            features[f, i, 3] = np.std(RSSI[i][1:])
            features[f, i, 4] = np.mean(dist)
            features[f, i, 5] = np.std(dist)
            features[f, i, 6] = n
            features[f, i, 7:] = APs_airTime[i]
            target[f, i, 0] = thr[0]

    return (features, target)

## Dataset Loader
####################################
# 1- Dataset class
####################################
class Data(torch.utils.data.Dataset):
    # Initialization method for the dataset
    def __init__(self, input_path, simulator_path, type='train', scen=1):

        APs_info = list()
        APs_throughput = list()

        if type == 'train':
            print("Loading Training Dataset ...", scen)

            if scen == 1:
                print("Loading Scen 1 (A) ...")
                APs_infoA, APs_throughputA = load_info(input_train + 'sce1a/input_nodes_sce1a_deployment_',
                                                       output_train_sim + 'script_output_sce1a.txt', 100, 12)
                print("Loading Scen 1 (B) ...")
                APs_infoB, APs_throughputB = load_info(input_train + 'sce1b/input_nodes_sce1b_deployment_',
                                                       output_train_sim + 'script_output_sce1b.txt', 100, 12)
                print("Loading Scen 1 (C) ...")
                APs_infoC, APs_throughputC = load_info(input_train + 'sce1c/input_nodes_sce1c_deployment_',
                                                       output_train_sim + 'script_output_sce1c.txt', 75, 12)

                for i in range(100):
                    APs_info.append(APs_infoA[i])
                    APs_throughput.append(APs_throughputA[i])
                for i in range(100):
                    APs_info.append(APs_infoB[i])
                    APs_throughput.append(APs_throughputB[i])
                for i in range(75):
                    APs_info.append(APs_infoC[i])
                    APs_throughput.append(APs_throughputC[i])
            elif scen == 2:
                print("Loading Scen 2 (A) ...")
                APs_infoA, APs_throughputA = load_info(input_train + 'sce2a/input_nodes_sce2a_deployment_',
                                                       output_train_sim + 'script_output_sce2a.txt', 100, 8)
                print("Loading Scen 2 (B) ...")
                APs_infoB, APs_throughputB = load_info(input_train + 'sce2b/input_nodes_sce2b_deployment_',
                                                       output_train_sim + 'script_output_sce2b.txt', 100, 8)
                print("Loading Scen 2 (C) ...")
                APs_infoC, APs_throughputC = load_info(input_train + 'sce2c/input_nodes_sce2c_deployment_',
                                                       output_train_sim + 'script_output_sce2c.txt', 75, 8)
                for i in range(100):
                    APs_info.append(APs_infoA[i])
                    APs_throughput.append(APs_throughputA[i])
                for i in range(100):
                    APs_info.append(APs_infoB[i])
                    APs_throughput.append(APs_throughputB[i])
                for i in range(75):
                    APs_info.append(APs_infoC[i])
                    APs_throughput.append(APs_throughputC[i])

        elif type == 'test':
            print("Loading Test Dataset Scen {} ...".format(scen))
            if scen == 1:
                APs_4_info, APs_4_throughput = load_info(input_test + 'test_1/input_nodes_test_sce1_deployment_',
                                                         output_test_sim + 'script_output_test_sce1.txt', 50, 4)
                for i in range(50):
                    APs_info.append(APs_4_info[i])
                    APs_throughput.append(APs_4_throughput[i])
            elif scen == 2:
                APs_6_info, APs_6_throughput = load_info(input_test + 'test_2/input_nodes_test_sce2_deployment_',
                                                         output_test_sim + 'script_output_test_sce2.txt', 50, 6)
                for i in range(50):
                    APs_info.append(APs_6_info[i])
                    APs_throughput.append(APs_6_throughput[i])
            elif scen == 3:
                APs_8_info, APs_8_throughput = load_info(input_test + 'test_3/input_nodes_test_sce3_deployment_',
                                                         output_test_sim + 'script_output_test_sce3.txt', 50, 8)
                for i in range(50):
                    APs_info.append(APs_8_info[i])
                    APs_throughput.append(APs_8_throughput[i])
            elif scen == 4:
                APs_10_info, APs_10_throughput = load_info(input_test + 'test_4/input_nodes_test_sce4_deployment_',
                                                           output_test_sim + 'script_output_test_sce4.txt', 50, 10)
                for i in range(50):
                    APs_info.append(APs_10_info[i])
                    APs_throughput.append(APs_10_throughput[i])
        print("DONE !")
        self.data_info = APs_info
        self.target = APs_throughput

    def __getitem__(self, index):
        data_info = self.data_info[index]
        target = self.target[index]

        info = torch.tensor(np.array(data_info, dtype=np.float32))
        target = torch.tensor(np.array(target, dtype=np.float32))

        # return the features and the target as a tuple
        return (info, target)

        # Return the number of scenarios

    def __len__(self):
        return len(self.target)

## Load data

dataset_train_1 = Data(input_train, output_train_sim, type = 'train', scen=1)
dataset_train_2 = Data(input_train, output_train_sim, type = 'train', scen=2)
dataset_test_4 = Data(input_test, output_test_sim, type = 'test', scen=1)
dataset_test_6 = Data(input_test, output_test_sim, type = 'test', scen=2)
dataset_test_8 = Data(input_test, output_test_sim, type = 'test', scen=3)
dataset_test_10 = Data(input_test, output_test_sim, type = 'test', scen=4)

# Split the dataset (90%, 10%)
train1, validation1 = torch.utils.data.random_split(dataset_train_1, [250,25])
train2, validation2 = torch.utils.data.random_split(dataset_train_2, [250,25])

# Concatenate the Dataset
train_loader = torch.utils.data.DataLoader(dataset= torch.utils.data.ConcatDataset((train1, train2)), batch_size=50)
validation_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset((validation1, validation2)), batch_size=25)
# train_loader = torch.utils.data.DataLoader(dataset= torch.utils.data.ConcatDataset((dataset_train_1, dataset_train_2)), batch_size=55)

test_loader_4 = torch.utils.data.DataLoader(dataset=dataset_test_4, batch_size=50)
test_loader_6 = torch.utils.data.DataLoader(dataset=dataset_test_6, batch_size=50)
test_loader_8 = torch.utils.data.DataLoader(dataset=dataset_test_8, batch_size=50)
test_loader_10 = torch.utils.data.DataLoader(dataset=dataset_test_10, batch_size=50)

############################################################
# 3- Implement an MLP- Multi Layer Perceptron using PyTorch
#1st Layer has  features from SINR, RSSI, DISTANCE  dastsets  with 2 Parallel Blocks of  2 featurses each and 1 Output  PReLu Activation
#Function. 2nd linear layer is formed with 4 neourons  and 3rd layer modles the available airtime and channel distributions.
# 3rd layer outputs the 3 Dimensional vector which is activated by  a PReLu function.
############################################################
class MLP_PyTorch(Module):
    def __init__(self):
        super(MLP_PyTorch, self).__init__()

        self.Linear_sinr = Linear(2, 1)
        self.Linear_rssi = Linear(2, 1)
        self.Linear_dist = Linear(2, 1)
        self.Linear_signal = Sequential(
            Linear(4, 3),
            PReLU(),
        )
        self.Linear_airtime = Sequential(
            Linear(8, 3),
            PReLU(),
        )
        self.Linear_throughput = Sequential(
            Linear(6, 7),
            PReLU(),
            Linear(7, 1),
            ReLU(),
        )
        self.norm3 = BatchNorm1d(4)
        self.prelu = PReLU()

    # Define the feed-forward pass of the Neural Network module using the sub-modules declared in the initializer
    def forward(self, X):
        batch_size = X.size(0)
        N_APs = X.size(1)
        n = X[:, :, 6, np.newaxis]

        sinr = self.Linear_sinr(X[:, :, :2])
        rssi = self.Linear_rssi(X[:, :, 2:4])
        dist = self.Linear_dist(X[:, :, 4:6])

        signal = torch.cat((rssi, dist, sinr, n), axis=2)
        signal = self.norm3(torch.flatten(signal, end_dim=1)).reshape(batch_size, N_APs, 4)
        signal = self.prelu(signal)
        signal = self.Linear_signal(signal)

        airtime = self.Linear_airtime(X[:, :, 7:])

        throughput = self.Linear_throughput(torch.cat((signal, airtime), axis=2))

        return throughput


# Function to TRAIN THE MLP with PyTorch
def train_PyTorch(NN, train_loader, validation_loader, criterion, optimizer=None, num_epochs=5000,
                  model_name='MLP.ckpt', device='cpu'):
    train_list_loss = []
    validation_list_loss = []
    for epoch in range(num_epochs):
        loss_avg = 0
        nBatches = 0
        validation_loss_avg = 0
        validation_nBatches = 0

        NN.train()
        for i, (data, targets) in enumerate(train_loader):
            X = data.to(device)
            y = targets.to(device)

            # reset optimizer at each epoch
            optimizer.zero_grad()

            # make prediction
            yHat = NN.forward(X)

            # compute the loss function
            loss = criterion(yHat, y)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            loss_avg += loss.cpu().item()
            nBatches += 1

        NN.eval()
        for i, (data, targets) in enumerate(validation_loader):
            X = data.to(device)
            y = targets.to(device)

            # make prediction
            yHat = NN.forward(X)

            # compute the loss function
            loss = criterion(yHat, y)

            validation_loss_avg += loss.cpu().item()
            validation_nBatches += 1

        # Print loss and save the value at each iteration
        if epoch % 100 == 0:
            print('Loss {} = {} , {}'.format(epoch, loss_avg / nBatches, validation_loss_avg / validation_nBatches))
        train_list_loss.append(loss_avg / nBatches)
        validation_list_loss.append(validation_loss_avg / validation_nBatches)

    print('Loss {} = {} , {}'.format(epoch, loss_avg / nBatches, validation_loss_avg / validation_nBatches))
    torch.save(NN.state_dict(), results_path + model_name)
    return train_list_loss, validation_list_loss

#Execution
# Initialize a Pytorch MLP

NN = MLP_PyTorch().to(device)

optimizer = torch.optim.Adam(NN.parameters(), lr= 2E-2)
criterion = MSELoss()

#Train MLP using Pytorch:
train_list_loss, validation_list_loss = train_PyTorch(NN, train_loader, validation_loader, criterion, num_epochs = 700, optimizer = optimizer, device=device)

# Plot the evolution of the loss function during training
plt.plot(train_list_loss[10:],c='y', label="Train")
plt.plot(validation_list_loss[10:],c='b', label="Validation")
plt.xlabel('Iterations')
plt.ylabel('Loss Val')
plt.legend(loc="upper right")
plt.show()

## If the loss gets suck, repeat the execution, this may happend when weights are initialized randomly

NN = MLP_PyTorch()
NN.eval()
NN.load_state_dict(torch.load(results_path + 'MLP.ckpt'))

y_train = torch.empty(0)
y_hat_train = torch.empty(0)
for x, y in train_loader:
  y_hat_train = torch.cat((y_hat_train,torch.flatten(NN(x))))
  y_train = torch.cat((y_train,torch.flatten(y)))
print(y_hat_train.shape, y_train.shape)
plt.scatter(y_hat_train.detach(), y_train, s=2,c='y', label= "Train")

x_test, y_test = next(iter(validation_loader))
y_test = torch.flatten(y_test)
y_hat_test = torch.flatten(NN(x_test))
plt.scatter(y_hat_test.cpu().detach(), y_test, s=2,c='b', label="Validation")

plt.plot(np.arange(350),c='black')
plt.ylim(0, 350)
plt.xlim(0, 350)
plt.legend(loc="upper left")
plt.show()

## LOAD & EXECUTE

## Load, compute predictions.

NN = MLP_PyTorch()
NN.eval()
NN.load_state_dict(torch.load(results_path + 'MLP.ckpt'))
data_4, target_4 = next(iter(test_loader_4))
data_6, target_6 = next(iter(test_loader_6))
data_8, target_8 = next(iter(test_loader_8))
data_10, target_10 = next(iter(test_loader_10))

APs_4_predicted_throughput = NN.forward(data_4)
APs_6_predicted_throughput = NN.forward(data_6)
APs_8_predicted_throughput = NN.forward(data_8)
APs_10_predicted_throughput = NN.forward(data_10)

def print_plot_scenario(predictions, targets, num_of_scen, N_APs):
    results = np.zeros((num_of_scen))
    for i in range(num_of_scen):
        results[i] = criterion(predictions[i], targets[i]).item()
    print('\n %d APs :\n'%(N_APs), np.sort(results))

    y_test = torch.flatten(targets)
    y_hat_test = torch.flatten(predictions).detach().numpy()
    print(y_test.shape)
    plt.scatter(y_hat_test, y_test, s=2)
    plt.ylabel('Real Throughput')
    plt.xlabel('Predicted Throughput')
    plt.plot(np.arange(350),c='black')
    plt.ylim(0, 350)
    plt.xlim(0, 350)
    plt.show()

## Compute the MSE of each scenario and plot the predictions.

criterion = MSELoss()

print("4 TOTAL Mean squared error: %.2f" % criterion(APs_4_predicted_throughput,target_4).item())
print("6 TOTAL Mean squared error: %.2f" % criterion(APs_6_predicted_throughput, target_6).item())
print("8 TOTAL Mean squared error: %.2f" % criterion(APs_8_predicted_throughput, target_8).item())
print("10 TOTAL Mean squared error: %.2f" % criterion(APs_10_predicted_throughput, target_10).item())

print_plot_scenario(APs_4_predicted_throughput, target_4, 50, 4)
print_plot_scenario(APs_6_predicted_throughput, target_6, 50, 6)
print_plot_scenario(APs_8_predicted_throughput, target_8, 50, 8)
print_plot_scenario(APs_10_predicted_throughput, target_10, 50, 10)

print("TOTAL Mean squared error: %.2f" % criterion(target_4, APs_4_predicted_throughput))
print()
item = 0
item = randint(0, 49)

print(item,"Mean squared error: %.2f" % criterion(target_4[item], APs_4_predicted_throughput[item]))

print(APs_4_predicted_throughput[item])
print(target_4[item])

# GRAPHICAL REPRESENTATION:
plt.plot(APs_4_predicted_throughput[item].detach(), linestyle='--', marker='o', color='b')
plt.plot(target_4[item].detach(), linestyle='--', marker='o', color='y')
plt.ylim(0, 500)
plt.show()

print("TOTAL Mean squared error: %.2f" % criterion(target_6, APs_6_predicted_throughput))
print()

item = 33
# item = randint(0, 49)

print(item,"Mean squared error: %.2f" % criterion(target_6[item], APs_6_predicted_throughput[item]))

print(APs_6_predicted_throughput[item])
print(target_6[item])

# GRAPHICAL REPRESENTATION:
plt.plot(APs_6_predicted_throughput[item].detach(), linestyle='--', marker='o', color='b')
plt.plot(target_6[item].detach(), linestyle='--', marker='o', color='y')
plt.ylim(0, 500)
plt.show()

print("TOTAL Mean squared error: %.2f" % criterion(target_8, APs_8_predicted_throughput))
print()

item = 0
item = randint(0, 49)

print(item,"Mean squared error: %.2f" % criterion(target_8[item], APs_8_predicted_throughput[item]))

print(APs_8_predicted_throughput[item])
print(target_8[item])

# GRAPHICAL REPRESENTATION:
plt.plot(APs_8_predicted_throughput[item].detach(), linestyle='--', marker='o', color='b')
plt.plot(target_8[item].detach(), linestyle='--', marker='o', color='y')
plt.ylim(0, 500)
plt.show()

print("TOTAL Mean squared error: %.2f" % criterion(target_10, APs_10_predicted_throughput))
print()

item = 0
item = randint(0, 49)

print(item,"Mean squared error: %.2f" % criterion(target_10[item], APs_10_predicted_throughput[item]))

print(APs_10_predicted_throughput[item])
print(target_10[item])

# GRAPHICAL REPRESENTATION:
plt.plot(APs_10_predicted_throughput[item].detach(), linestyle='--', marker='o', color='b')
plt.plot(target_10[item].detach(), linestyle='--', marker='o', color='y')
plt.ylim(0, 500)
plt.show()

print('Saving...')
print('1')
for i in range(50):
    np.savetxt(results_path+'test scenario 1-4 APs/throughput_'+format(i+1,'1')+'.csv', [APs_4_predicted_throughput[i].detach().numpy().reshape(4)], delimiter=',',fmt='%.2f')
print('2')
for i in range(50):
    np.savetxt(results_path+'test scenario 2-6 APs/throughput_'+format(i+1,'1')+'.csv', [APs_6_predicted_throughput[i].detach().numpy().reshape(6)], delimiter=',',fmt='%.2f')
print('3')
for i in range(50):
    np.savetxt(results_path+'test scenario 3-8 APs/throughput_'+format(i+1,'1')+'.csv', [APs_8_predicted_throughput[i].detach().numpy().reshape(8)], delimiter=',',fmt='%.2f')
print('4')
for i in range(50):
    np.savetxt(results_path+'test scenario 4-10 APs/throughput_'+format(i+1,'1')+'.csv',[APs_10_predicted_throughput[i].detach().numpy().reshape(10)], delimiter=',',fmt='%.2f')
print('DONE!')
