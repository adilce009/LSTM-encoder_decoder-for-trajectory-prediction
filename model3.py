#output features: only x, y locations

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

device = torch.device("cuda:0")

class Encoder ( nn.Module ):
    def __init__ ( self , input_size , cell_size , hidden_size ):
        """
        cell_size is the size of cell_state.
        hidden_size is the size of hidden_state, or say the output_state of each step
        """
        super ( Encoder , self ).__init__ ()

        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.fl = nn.Linear ( input_size + hidden_size , hidden_size )
        self.il = nn.Linear ( input_size + hidden_size , hidden_size )
        self.ol = nn.Linear ( input_size + hidden_size , hidden_size )
        self.Cl = nn.Linear ( input_size + hidden_size , hidden_size )

    def forward ( self , input , Hidden_State , Cell_State ):
        # print(input)
        combined = torch.cat ( (input.float(), Hidden_State.float()), 1)
        #combined = torch.cat((torch.unsqueeze(input, dim=0).float(), Hidden_State.float()), 1)
        f = torch.sigmoid ( self.fl ( combined ) )
        i = torch.sigmoid ( self.il ( combined ) )
        o = torch.sigmoid ( self.ol ( combined ) )
        C = torch.tanh ( self.Cl ( combined ) )
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * torch.tanh ( Cell_State )

        return Hidden_State , Cell_State

    def loop ( self , inputs ):
        batch_size = inputs.size ( 0 )
        time_step = inputs.size ( 2 )
        Hidden_State , Cell_State = self.initHidden ( batch_size )
        for i in range ( time_step ):
            Hidden_State , Cell_State = self.forward(torch.squeeze(inputs[:,:, i]), Hidden_State, Cell_State)
            #Hidden_State, Cell_State = self.forward(inputs, Hidden_State, Cell_State)
        return Hidden_State , Cell_State, inputs[:, 0:2, -1]

    def initHidden ( self , batch_size ):
        Hidden_State = Variable ( torch.zeros ( batch_size , self.hidden_size ).to( device ))
        Cell_State = Variable ( torch.zeros ( batch_size , self.hidden_size ).to ( device ))
        return Hidden_State , Cell_State


class Decoder(nn.Module):
    def __init__(self, input_size , cell_size , hidden_size, batchsize, timestep):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.batch_size = batchsize
        self.time_step = timestep
        self.out_features = 2  #location, velocity, acceleration
        self.fl = nn.Linear ( self.input_size + hidden_size , hidden_size )
        self.il = nn.Linear ( self.input_size + hidden_size , hidden_size )
        self.ol = nn.Linear ( self.input_size + hidden_size , hidden_size )
        self.Cl = nn.Linear ( self.input_size + hidden_size , hidden_size )
        self.linear1 = nn.Linear ( cell_size ,  self.out_features )  #?


    def forward(self, input , Hidden_State , Cell_State):

        combined = torch.cat ((input , Hidden_State), 1)
        f = torch.sigmoid ( self.fl ( combined ) )
        i = torch.sigmoid ( self.il ( combined ) )
        o = torch.sigmoid ( self.ol ( combined ) )
        C = torch.tanh ( self.Cl ( combined ) )
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * torch.tanh ( Cell_State )

        return Hidden_State , Cell_State

    def loop ( self, hidden_vec_from_encoder, last_input_vector):
        batch_size = self.batch_size
        time_step = self.time_step
        #if self.stream =='s2':
            #Cell_State, out, stream2_output = self.initHidden()

        #Cell_State , out  = self.initHidden ()
        Cell_State = self.initHidden()
        center_x_pred , center_y_pred = self.initOutParams()
        for i in range ( time_step ):
            if i == 0:
                Hidden_State = hidden_vec_from_encoder
                out = last_input_vector.float()

            Hidden_State , Cell_State = self.forward(out , Hidden_State , Cell_State )

            # print(Hidden_State.data)
            y_m = self.linear1(Hidden_State)
            #y_m1 = self.linear2(y_m)
            #y_m2 = self.linear3(y_m1)

            out = y_m
            center_x_, center_y_, = y_m.chunk(2, dim=-1) #?
            '''
            calculated_x = xVelocity_*0.04 + 0.5*xAcceleration_*0.04*0.04
            calculated_y = yVelocity_*0.04 + 0.5*yAcceleration_*0.04*0.04
            '''

            center_x_pred[:, i, :] = center_x_
            center_y_pred[:, i, :] = center_y_
            #xVelocity_pred[:, i, :] = xVelocity_
            #yVelocity_pred[:, i, :] = yVelocity_
            #xAcceleration_pred[:, i, :] = xAcceleration_
            #yAcceleration_pred[:, i, :] = yAcceleration_

        Stream_output = torch.cat((center_x_pred, center_y_pred), dim=2)
        return Stream_output , Cell_State

    def initHidden(self):
        #out = torch.randn(self.batch_size, self.out_features)
        #return torch.randn(self.batch_size, self.hidden_size), out
        #out = torch.zeros(self.batch_size, self.out_features)
        return torch.zeros(self.batch_size, self.hidden_size, device=device) #, out

    def initOutParams(self):
        center_x_pred = torch.randn(self.batch_size, self.time_step, 1, device=device)
        center_y_pred = torch.randn(self.batch_size, self.time_step, 1, device=device)
        #xVelocity_pred = torch.randn(self.batch_size, self.time_step, 1, device=device)
        #yVelocity_pred = torch.randn(self.batch_size, self.time_step, 1, device=device)
        #xAcceleration_pred = torch.randn(self.batch_size, self.time_step, 1, device=device)
        #yAcceleration_pred = torch.randn(self.batch_size, self.time_step, 1, device=device)

        return center_x_pred, center_y_pred
