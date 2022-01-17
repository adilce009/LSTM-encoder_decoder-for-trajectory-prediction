# normalize data
#imports
import sys
import os
import time
import torch.utils.data as utils
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as nppip
import random
from data_management import DataStreams
from model3 import *

import data_management.DataStreams
from torch.autograd import Variable

device = torch.device("cuda:0")


#import model


batch_size = 32
train_seq_length = 50 # 3s
pred_seq_length = 125 # 5s
num_epochs = 200
MODEL_LOC =  '/saved_model'  #'../../resources/trained_models/EncDec'

def shuffle_data(train_obs, train_label):
    mapIndexPosition = list(zip(train_obs,train_label))
    random.shuffle(mapIndexPosition)
    train_obs_shuffled, train_label_shuffled = zip(*mapIndexPosition)
    return train_obs_shuffled, train_label_shuffled

#def load_next_batch(batch_number):

def normalization(data):
    A = [list(i) for i in zip(*(data))]     # group feature values from all the batches together; gives feature_length * num_batch* sequence_len
    A = torch.tensor(A)
    mew = torch.mean(A, dim = (1,2))    # get mean and std of each feature separately
    sd = torch.std(A, dim = (1,2))

    data = [(torch.sub(torch.transpose(torch.tensor(item), 0,1), mew))/sd for item in data] #z scores / normalization
    data = [torch.transpose(data[i], 0, 1, ) for i in range(len(data))] # returning to the shape batch, features, sequence
    data = torch.stack(data)    # list to tensor


    return data

#def trainiters(n_epochs, train_dataloader, valid_dataloader, test1, test2, data, sufix, print_every=1, plot_every=1000, learning_rate=1e-3):
def trainitres():
    input_dim = 60
    decoder_input_dim = 12  # number of features of the agent
    hidden_dim = 128
    output_dim = 2     # number of features we want as output
    step_size = 125 # 25 = 1 second
    learning_rate = 0.001
    encoder = Encoder(input_dim, hidden_dim, hidden_dim).to(device)
    decoder = Decoder(output_dim, hidden_dim, hidden_dim, batch_size, step_size).to(device)
    encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=learning_rate)
    #load data
    train_obs, train_label, test_obs, test_label = DataStreams.load_data()
    train_obs = normalization(train_obs)

    for epoch in range(num_epochs):
        print_loss_total = 0
        train_obs_shuffled, train_label_shuffled = shuffle_data(train_obs, train_label)
        num_batches = int(len(train_obs) / batch_size)
        batch_start_index = 0
        batch_end_index = batch_size

        for bch in range(num_batches):
            #get batch
            obs_batch = train_obs_shuffled [batch_start_index: batch_end_index]
            label_batch = train_label_shuffled [batch_start_index: batch_end_index]
            target_label_batch = np.zeros(((len(label_batch), output_dim, step_size))) # step_size is sequence length

            for i in range(len(label_batch)):
                target_label_batch[i,:,:] = label_batch[i][0:2,:]
            target_label_batch = torch.tensor(target_label_batch).to(device)
            target_label_batch = target_label_batch.transpose(1,2)
            batch_start_index = batch_end_index
            batch_end_index = batch_end_index + batch_size
            obs_batch = torch.stack(obs_batch)
            input_tensor = torch.tensor(obs_batch).to(device)
            target_tensor = torch.tensor(target_label_batch).to(device)

            loss = train(input_tensor, target_tensor, encoder, decoder,
                                        encoder_optimizer, decoder_optimizer)
            print_loss_total += loss
        print('{}/{} epochs: average loss:'.format(epoch, num_epochs), print_loss_total / num_batches)
        print("end of this epoch")
        #calcualte_loss ()_ with_train_steam()

    #compute_accuracy_stream1(train_obs, train_label, encoder, decoder, num_epochs, batch_size, output_dim, step_size)
    save_model(encoder, decoder)
    compute_accuracy_stream1(test_obs, test_label, encoder, decoder, num_epochs, batch_size, output_dim, step_size)

    print("end")

    #save_model()
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer):
    Hidden_State, _, last_input_vector= encoder.loop(input_tensor)

    stream2_out, _ = decoder.loop(Hidden_State, last_input_vector)
    stream2_out = stream2_out.double()

    l = nn.MSELoss()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = l(stream2_out, target_tensor)
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


def compute_accuracy_stream1 ( obs , label , encoder , decoder , n_epochs,batch_size, output_dim, step_size):
    ade = 0
    fde = 0
    longitudinal_error = 0
    lateral_error = 0
    count = 0

    obs_shuffled, label_shuffled = shuffle_data(obs, label)
    num_batches = int(len(obs) / batch_size)
    batch_start_index = 0
    batch_end_index = batch_size

    for bch in range(num_batches):
        # get batch
        obs_batch = obs_shuffled[batch_start_index: batch_end_index]
        label_batch = label_shuffled[batch_start_index: batch_end_index]
        target_label_batch = np.zeros(((len(label_batch), output_dim, step_size)))

        for i in range(len(label_batch)):
            target_label_batch[i, :, :] = label_batch[i][0:2, :]    # 6: number of features predicting: x,y, xV,yV,xA,yA
        target_label_batch = torch.tensor(target_label_batch)
        target_label_batch = target_label_batch.transpose(1, 2)
        batch_start_index = batch_end_index
        batch_end_index = batch_end_index + batch_size

        input_tensor = torch.tensor(obs_batch).to(device)
        target_tensor = torch.tensor(target_label_batch).to(device)

        Hidden_State , _, last_input_vector = encoder.loop ( input_tensor )
        stream2_out , _  = decoder.loop ( Hidden_State, last_input_vector )
        #scaled_train = scale ( stream2_out , target_tensor )
        #mse = MSE ( scaled_train/torch.max(scaled_train) , target_tensor/torch.max(target_tensor) ) * (torch.max(target_tensor)).cpu().detach().numpy()
        mse, longi_error, lat_error= MSE(stream2_out, target_tensor)
        # mse = MSE ( scaled_train , label )
        mse = np.sqrt ( mse )
        # print(mse)
        longitudinal_error += longi_error
        lateral_error += lat_error
        ade += mse
        fde += mse[ -1 ]
        #count += testbatch_in_form.size ()[ 0 ]
        count += target_label_batch.size()[0]  #?
    ade = ade / count
    fde = fde / count
    longitudinal_error = longitudinal_error / num_batches
    lateral_error = lateral_error / num_batches
    print ( "ADE_1s: {}, ADE_2s: {}, ADE_3s: {}, ADE_4s: {}, ADE_5s: {},FDE: {}".format ( ade[24] , ade[49] , ade[74] , ade[99], ade[124], fde ) )
    print(" Longitudinal Errors: 1s: {}, 2s: {}, 3s:{},4s: {},5s: {},".format(longitudinal_error[24], longitudinal_error[49], longitudinal_error[74], longitudinal_error[99], longitudinal_error[124]))
    print(" Lateral Errors: 1s: {}, 2s: {}, 3s:{},4s: {},5s: {},".format(lateral_error[24], lateral_error[49], lateral_error[74], lateral_error[99], lateral_error[124]))
    print ( "average: ADE: {} FDE: {} LongitudinalE: {}, LateralE: {}".format ( np.mean(ade), np.mean(fde), np.mean(longitudinal_error), np.mean(lateral_error) ) )

def save_model( encoder, decoder,  loc=MODEL_LOC, data= 'highD', sufix = ' '):
    torch.save(encoder.state_dict(), os.path.join(loc, 'encoder_stream_EncDec_{}{}.pt'.format(data, sufix)))
    torch.save(decoder.state_dict(), os.path.join(loc, 'decoder_stream_EncDec_{}{}.pt'.format(data, sufix)))
    print('model saved at {}'.format(loc))


def MSE ( y_pred , y_gt):
    # y_pred = y_pred.numpy()
    y_pred = y_pred.cpu ().detach ().numpy ()
    y_gt = y_gt.cpu ().detach ().numpy ()
    acc = np.zeros ( np.shape ( y_pred )[ :-1 ] )
    muX = y_pred[ : , : , 0 ]   # x vlaues
    muY = y_pred[ : , : , 1 ]   # y values
    x = np.array ( y_gt[ : , : , 0 ] )
    y = np.array ( y_gt[ : , : , 1 ] )
    longitudinal_error = np.abs((muX - x))
    longitudinal_error = np.sum(longitudinal_error, axis = 0)/y_pred.shape[0]
    lateral_error = np.abs (muY - y)
    lateral_error = np.sum(lateral_error, axis = 0)/y_pred.shape[0]
    #print ( muX , x , muY , y )
    acc = np.power ( x - muX , 2 ) + np.power ( y - muY , 2 )
    lossVal = np.sum ( acc , axis=0 ) / len ( acc )
    return lossVal,longitudinal_error, lateral_error


trainitres ()

'''
def scale(train_tensor, target_tensor):
    train_tensor_x = train_tensor[:,:,0].clone()
    train_tensor_y= train_tensor[:,:,1].clone()
    train_tensor_xV = train_tensor[:,:,2].clone()
    train_tensor_yV = train_tensor[:, :, 3].clone()
    train_tensor_xA = train_tensor[:, :, 4].clone()
    train_tensor_yA = train_tensor[:, :, 5].clone()
    target_tensor_x= target_tensor[:, :, 0].clone()
    target_tensor_y= target_tensor[:, :, 1].clone()
    target_tensor_xV = target_tensor[:, :, 2].clone()
    target_tensor_yV = target_tensor[:, :, 3].clone()
    target_tensor_xA = target_tensor[:, :, 4].clone()
    target_tensor_yA = target_tensor[:, :, 5].clone()

    train_tensor[:, :, 0] = torch.mean(target_tensor_x) + (train_tensor_x - torch.mean(train_tensor_x))*( (torch.std(target_tensor_x)/ torch.std(train_tensor_x)))
    train_tensor[:, :, 1] = torch.mean(target_tensor_y) + (train_tensor_y- torch.mean(train_tensor_y))*( (torch.std(target_tensor_y)/ torch.std(train_tensor_y)))
    train_tensor[:, :, 2] = torch.mean(target_tensor_xV) + (train_tensor_xV - torch.mean(train_tensor_xV)) * ((torch.std(target_tensor_xV) / torch.std(train_tensor_xV)))
    train_tensor[:, :, 3] = torch.mean(target_tensor_yV) + (train_tensor_yV - torch.mean(train_tensor_yV)) * ((torch.std(target_tensor_yV) / torch.std(train_tensor_yV)))
    train_tensor[:, :, 4] = torch.mean(target_tensor_xA) + (train_tensor_xA - torch.mean(train_tensor_xA)) * ((torch.std(target_tensor_xA) / torch.std(train_tensor_xA)))
    train_tensor[:, :, 5] = torch.mean(target_tensor_yA) + (train_tensor_yA - torch.mean(train_tensor_yA)) * ((torch.std(target_tensor_yA) / torch.std(train_tensor_yA)))
    return train_tensor
'''


'''
# call eval()
_, _, test_obs, test_label = DataStreams.load_data()
eval(test_obs, test_label)
def eval(epochs, test_obs, test_label, data='highD', sufix=' ', learning_rate=1e-3, loc=MODEL_LOC):
    batch_size = 30
    output_dim = 6
    step_size = 50 # 2 seconds
    encoder_stream1 = None
    decoder_stream1 = None

    encoder1loc = os.path.join(loc, 'encoder_stream_EncDec_{}{}.pt'.format(data, sufix))
    decoder1loc = os.path.join(loc, 'decoder_stream_EncDec_{}{}.pt'.format(data, sufix))

    train_raw = test_obs
    pred_raw = test_label
    # Initialize encoder, decoders for both streams
    #batch = load_batch(0, BATCH_SIZE, 'pred', train_raw, pred_raw, train2_raw, pred2_raw)
    #batch, _ = batch
    num_batches = int(len(test_obs) / batch_size)
    #batch_in_form = np.asarray([batch[i]['sequence'] for i in range(BATCH_SIZE)])
    #batch_in_form = torch.Tensor(batch_in_form)
    batch_start_index = 0
    batch_end_index = batch_size
    for bch in range(num_batches):
        # get batch
        test_obs_batch = test_obs[batch_start_index: batch_end_index]
        test_label_batch = test_label[batch_start_index: batch_end_index]
        target_label_batch = np.zeros(((len(test_label_batch), output_dim, step_size)))
        for i in range(len(test_label_batch)):
            target_label_batch[i, :, :] = test_label_batch[i][0:6, :]
        target_label_batch = torch.tensor(target_label_batch)
        target_label_batch = target_label_batch.transpose(1, 2)
        batch_start_index = batch_end_index
        batch_end_index = batch_end_index + batch_size

        input_tensor = torch.tensor(test_obs_batch)  # .to(device)
        target_tensor = torch.tensor(target_label_batch)

    [batch_size, step_size, fea_size] = batch_in_form.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size

    encoder_stream1 = Encoder(input_dim, hidden_dim, output_dim).to(device)
    decoder_stream1 = Decoder('s1', input_dim, hidden_dim, output_dim, batch_size, step_size).to(device)
    encoder_stream1_optimizer = optim.RMSprop(encoder_stream1.parameters(), lr=learning_rate)
    decoder_stream1_optimizer = optim.RMSprop(decoder_stream1.parameters(), lr=learning_rate)
    encoder_stream1.load_state_dict(torch.load(encoder1loc))
    encoder_stream1.eval()
    decoder_stream1.load_state_dict(torch.load(decoder1loc))
    decoder_stream1.eval()

    compute_accuracy_stream1(tr_seq_1, pred_seq_1, encoder_stream1, decoder_stream1, epochs)
'''
