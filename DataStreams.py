import numpy as np
import pickle
#import sklearn
import TrajWithNbrs
import random


def acceptable_length(traj):  # frame numbers of some trajectories are not high enough; remove the ones that has smaller number of frames
    obs_len = 50  # 50 frames  equivalent to 2 seconds
    pred_len = 125  # 125 frames equivalent to 5 seconds; 75 frames, equivalent to 3 seconds
    idx = []
    for i in range (len(traj)):
        if len(traj[i][1]) < obs_len + pred_len:
            idx.append(i)
    #idx = sorted(idx, inverse = True)
    for i in sorted(idx, reverse=True):
        del traj[i]
    return traj

def split_data(dataset):        # train and test split
    inputs = acceptable_length(dataset)
    num_samples = len(inputs)
    num_train_samples = int(num_samples*0.8)
    train = inputs[0:num_train_samples]
    test = inputs[num_train_samples:]

    return train, test

def segment_traj(data):     # create 2 segments for every trajectory: one for observation, another for prediction
    num_rows = 60
    num_samples = len(data)
    obs_len = 50    # 2 seconds
    label_len = 125  # 5 seconds; 75 for 3 seconds
    obs = np.zeros((num_samples, num_rows, obs_len))
    label = np.zeros((num_samples, num_rows, label_len))
    for i in range (len(data)):
        obs[i] = data[i][:, 0:obs_len]
        label[i] = data[i][:, obs_len:obs_len+label_len]

    return obs, label


#def load_batch():

def load_data():
    #dataset = TrajWithNbrs.get_data()
    #dataset = TrajWithNbrs.get_input_vector(tracks)
    with open('inpv_dirdri1.pickle', 'rb') as f: #  inpv_dirdri1.pickle, Wholeinpv_dirdri1.pickle, smallset.pickle
        dataset = pickle.load(f)
    train, test = split_data(dataset)

    train_obs, train_label = segment_traj(train)
    test_obs, test_label = segment_traj(test)

    return train_obs, train_label, test_obs, test_label
