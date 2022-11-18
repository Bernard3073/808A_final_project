import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from resnet import Bottleneck, ResNet, ResNet50

def ImportData(file):
    """Import data and group the lidar portion into 4. 
    The robot position/orientation and local goal position/orientation 
    were grouped into deltas. The data was then regularized by averaging 
    and dividing by the max in their respective columns to scale them 
    between 0 and 1.
    Args:
        file (str): Relative path to csv file
    Returns:
        [list]: Features and their respective Labels
    """

    # Define Training Data
    data = pd.read_csv(file)
    data = (data.to_numpy())

    # Define labels
    lin_vel = (data.T[-2]).T
    ang_vel = (data.T[-1]).T
    label = []
    for i in range(len(lin_vel)):
        label = np.append(label, [lin_vel[i], ang_vel[i]])
    label = np.reshape(label, (len(data), 2))

    # Define features vector (X_train/ X_test) pad with data set with a 1
    ld_end = 1080 # Last index of Lidar data
    ld_split = int(ld_end / 4) # Split lidar data into 67.5 degree segments
    lidar_1 = (data.T[:ld_split]).T # Lidar first 67.5 degree data
    lidar_2 = (data.T[ld_split:ld_split * 2]).T # Lidar next 67.5 degree to 135 degree data
    lidar_3 = (data.T[ld_split * 2:ld_split * 3]).T # Lidar first 135 degree to 202.5 degree data
    lidar_4 = (data.T[ld_split * 3:ld_split * 4]).T # Lidar first 202.5 degree to 270 degree data
    goal_l = [data.T[ld_end + 5], data.T[ld_end + 6]] # Local goal [x, y] feature
    goal_lq = [data.T[ld_end + 7], data.T[ld_end + 8]] # Local goal quaternion [qk, qr] feature
    robot_pos = [data.T[ld_end + 9], data.T[ld_end + 10]] # Robot pos [x, y] feature
    robot_q = [data.T[ld_end + 11], data.T[ld_end + 12]] # Robot orientation quaternion [qk, qr] feature
    
    pos_delta = np.subtract(goal_l, robot_pos) # Local goal - Robot pos [xl - xr, yl - yr] feature
    ori_delta = np.subtract(goal_lq, robot_q) # Local goal quaternion - Robot orientation quaternion [qk_l - qk_r, qr_l - qr_r] feature
    # Regularize data by taking the average
    l_1 = [np.average(l) for l in lidar_1]
    l_2 = [np.average(l) for l in lidar_2]
    l_3 = [np.average(l) for l in lidar_3]
    l_4 = [np.average(l) for l in lidar_4]
    # Regularize by dividing the max value of each set of lidar data
    l_1 = l_1 / np.amax(l_1)
    l_2 = l_2 / np.amax(l_2)
    l_3 = l_3 / np.amax(l_3)
    l_4 = l_4 / np.amax(l_4)

    feature = []
    for i in range(len(data)): # Link all data in an array and reshape to match the number of features padded data set with a 1
        feature = np.append(feature, (1, l_1[i], l_2[i], l_3[i], l_4[i], pos_delta[0][i], pos_delta[1][i], ori_delta[0][i], ori_delta[1][i]))
    feature = np.reshape(feature, (len(label), 9))

    return feature, label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ResNet', help='model name: test, lstm')
    args = parser.parse_args()

    train_feature, train_label = ImportData('corridor_CSV/July22_29.csv')
    test_feature, test_label = ImportData('July22_11.csv')

    # format to tensor
    train_feature = torch.from_numpy(train_feature).float()
    train_label = torch.from_numpy(train_label).float()
    test_feature = torch.from_numpy(test_feature).float()
    test_label = torch.from_numpy(test_label).float()

    # Define the model
    if args.model == 'test':
        model = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),  # activation
            nn.Linear(64, 64),
            nn.ReLU(),  # activation
            nn.Linear(64, 2),
        )
    elif args.model == 'lstm':
        model = nn.LSTM(9, 64, 2, batch_first=True)



    y_loss = {}  # loss history
    y_loss['train'] = []
    y_loss['val'] = []
    y_acc = {}
    y_acc['train'] = []
    y_acc['val'] = []

    x_epoch = []

    EPOCH = 1000
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    
    model.to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Define the loss function
    loss_function = nn.MSELoss()
    # Train the model
    for t in range(EPOCH):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(train_feature)
        # if t % 50 == 0:
        #     acc, loss = fwd_pass(net, train_feature, train_label, train=True)
        #     y_loss['train'].append(loss)
        #     y_acc['train'].append(acc)

        loss = loss_function(y_pred, train_label)
        # Compute loss and accuracy
        if t % 50 == 0:
            y_loss['train'].append(loss.item())
            # y_err['train'].append(torch.mean(torch.abs(y_pred - train_label)))
            # calculate accuracy
            matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(y_pred, train_label)]
            acc = matches.count(True) / len(matches)
            y_acc['train'].append(acc)
            x_epoch.append(t)
            print('Epoch: ', t, 'Loss: ', round(loss.item(), 4), 'Accuracy: ', round(acc, 2)*100)


        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # # Test the model
    # with torch.no_grad():
    #     y_pred = model(test_feature)
    #     loss = loss_fn(y_pred, test_label)
    #     # print(loss.item())
    #     y_loss['val'].append(loss.item())
    #     # y_err['val'].append(torch.mean(torch.abs(y_pred - test_label)))

    
    # # Save the model
    # torch.save(model.state_dict(), "model.pth")

    print("Done!")

    # plot loss curve
    
    plt.plot(x_epoch, y_loss['train'], label='loss')
    plt.plot(x_epoch, y_acc['train'], label='accuracy')
    plt.legend()
    plt.savefig(args.model + '.png')
    plt.show()



    # # plot the training result
    # plt.plot(train_label[:, 0].numpy(), label='ground truth')
    # plt.plot(y_pred[:, 0].numpy(), label='prediction')
    # plt.legend()
    # plt.show()


