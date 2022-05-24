import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from logging import getLogger
from libtraffic.model.abstract_traffic_state_model import AbstractTrafficStateModel
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import pandas as pd


def getedge():

    geofile = pd.read_csv('./raw_data/PEMSD4/PEMSD4.geo')
    geo_ids = list(geofile['geo_id'])
    geo_to_ind = {}
    for index, idx in enumerate(geo_ids):
        geo_to_ind[idx] = index
    r1 = np.loadtxt('./raw_data/PEMSD4/PEMSD4.rel', delimiter=',', usecols=2, skiprows=1)
    r2 = np.loadtxt('./raw_data/PEMSD4/PEMSD4.rel', delimiter=',', usecols=3, skiprows=1)
    for i in range(r1.size):
        r1[i] = geo_to_ind[r1[i]]
    for i in range(r2.size):
        r2[i] = geo_to_ind[r2[i]]
    return r1, r2

class GRULinear(nn.Module):
    def __init__(self, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(GRULinear, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 3, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        num_nodes=307
        batch_size = hidden_state.shape[0]
        inputs = inputs.reshape((batch_size, num_nodes, 3))
        # inputs (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 3))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [inputs, hidden_state] "[x, h]" (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (batch_size * num_nodes, gru_units + 1)
        concatenation = concatenation.reshape((-1, self._num_gru_units + 3))
        # [x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = concatenation @ self.weights + self.biases
        # [x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # [x, h]W + b (batch_size, num_nodes * output_dim)
        #outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }

class TGCNCell(nn.Module):
    def __init__(self, num_units, num_nodes, device, input_dim=1):
        super().__init__()
        self.num_units = num_units
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self._device = device
        self.act = torch.tanh
        # 这里提前构建好边集
        r1, r2 = getedge()
        self.r1 = r1
        self.r2 = r2
        self.init_params()
        self.GCN1 = GRULinear(100, 200)
        self.GCN2 = GRULinear(100, 100)
        self.GCN3 = GATv2Conv(103, 100)
    def init_params(self, bias_start=0.0):
        input_size = self.input_dim + self.num_units
        weight_0 = torch.nn.Parameter(torch.empty((input_size, 2 * self.num_units), device=self._device))
        bias_0 = torch.nn.Parameter(torch.empty(2 * self.num_units, device=self._device))
        weight_1 = torch.nn.Parameter(torch.empty((input_size, self.num_units), device=self._device))
        bias_1 = torch.nn.Parameter(torch.empty(self.num_units, device=self._device))

        torch.nn.init.xavier_normal_(weight_0)
        torch.nn.init.xavier_normal_(weight_1)
        torch.nn.init.constant_(bias_0, bias_start)
        torch.nn.init.constant_(bias_1, bias_start)

        self.register_parameter(name='weights_0', param=weight_0)
        self.register_parameter(name='weights_1', param=weight_1)
        self.register_parameter(name='bias_0', param=bias_0)
        self.register_parameter(name='bias_1', param=bias_1)

        self.weigts = {weight_0.shape: weight_0, weight_1.shape: weight_1}
        self.biases = {bias_0.shape: bias_0, bias_1.shape: bias_1}

    def forward(self, inputs, state):
        """
        Gated recurrent unit (GRU) with Graph Convolution.

        Args:
            inputs: shape (batch, self.num_nodes * self.dim)
            state: shape (batch, self.num_nodes * self.gru_units)

        Returns:
            torch.tensor: shape (B, num_nodes * gru_units)
        """
        batch_size = state.shape[0]
        state=self._gc3(state,inputs, self.num_units)
        output_size = 2 * self.num_units
        value = torch.sigmoid(
            self.GCN1(inputs, state))  # (batch_size, self.num_nodes, output_size)
        r, u = torch.split(tensor=value, split_size_or_sections=self.num_units, dim=-1)
        r = torch.reshape(r, (-1, self.num_nodes * self.num_units))  # (batch_size, self.num_nodes * self.gru_units)
        u = torch.reshape(u, (-1, self.num_nodes * self.num_units))
        c = self.act(self.GCN2(inputs, r * state))
        c = c.reshape(shape=(-1, self.num_nodes * self.num_units))
        new_state = u * state + (1.0 - u) * c
        return new_state




    def _gc3(self, state, inputs, output_size, bias_start=0.0):
        """
        GCN

        Args:
            inputs: (batch, self.num_nodes * self.dim)
            state: (batch, self.num_nodes * self.gru_units)
            output_size:
            bias_start:

        Returns:
            torch.tensor: (B, num_nodes , output_size)
        """
        batch_size = state.shape[0]

        state = torch.reshape(state, (batch_size, self.num_nodes, -1))  # (batch, self.num_nodes, self.gru_units)
        inputs = torch.reshape(inputs, (batch_size, self.num_nodes, -1))  
        inputs_and_state = torch.cat([state, inputs], dim=2)
        input_size = inputs_and_state.shape[2]
        x = inputs_and_state.cuda()
        edge_index = torch.tensor([self.r1, self.r2], dtype=torch.long).cuda()
        b=[]
        for i in x:
          x111=Data(x=i,edge_index=edge_index)
          xx=self.GCN3(x111.x,x111.edge_index)
          b.append(xx)
        x1=torch.stack(b)
        biases = self.biases[(output_size,)]
        x1 += biases

        x1 = x1.reshape(shape=(batch_size, self.num_nodes* output_size))
        return x1


class TGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        # self.adj_mx = data_feature.get('adj_mx')
        self.num_nodes = data_feature.get('num_nodes', 1)
        config['num_nodes'] = self.num_nodes
        self.input_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)
        self.gru_units = int(config.get('rnn_units', 64))
        self.lam = config.get('lambda', 0.0015)

        super().__init__(config, data_feature)

        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        #self.output_window = 6
        self.device = config.get('device', torch.device('cpu'))
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        # -------------------构造模型-----------------------------
        self.tgcn_model = TGCNCell(self.gru_units, self.num_nodes, self.device, self.input_dim)
        self.tgcn_model1 = TGCNCell(self.gru_units, self.num_nodes, self.device, self.input_dim)
        self.output_model = nn.Linear(self.gru_units*2, self.output_window * self.output_dim)
    def forward(self, batch):
        """
        Args:
            batch: a batch of input,
                batch['X']: shape (batch_size, input_window, num_nodes, input_dim) \n
                batch['y']: shape (batch_size, output_window, num_nodes, output_dim) \n

        Returns:
            torch.tensor: (batch_size, self.output_window, self.num_nodes, self.output_dim)
        """
        inputs = batch['X']
        # labels = batch['y']

        batch_size, input_window, num_nodes, input_dim = inputs.shape
        inputs = inputs.permute(1, 0, 2, 3)  # (input_window, batch_size, num_nodes, input_dim)
        inputs = inputs.view(self.input_window, batch_size, num_nodes * input_dim).to(self.device)
        state = torch.zeros(batch_size, self.num_nodes * self.gru_units).to(self.device)
        state1 = torch.zeros(batch_size, self.num_nodes * self.gru_units).to(self.device)

        for t in range(input_window):
              state = self.tgcn_model(inputs[t], state)
              state1 = self.tgcn_model1(inputs[input_window-t-1], state1)


        state = state.view(batch_size, self.num_nodes, self.gru_units)  # (batch_size, self.num_nodes, self.gru_units)
        state1 = state1.view(batch_size, self.num_nodes, self.gru_units)  # (batch_size, self.num_nodes, self.gru_units)
        #output1 = self.output_model(state)  # (batch_size, self.num_nodes, self.output_window * self.output_dim)

        state2 = torch.cat([state, state1], dim=2)
        output2=self.output_model(state2)
        #output1 = output1.view(batch_size, self.num_nodes, self.output_window, self.output_dim)
        output2 = output2.view(batch_size, self.num_nodes, self.output_window, self.output_dim)
        output2 = output2.permute(0, 2, 1, 3)
        #output2 = output2.permute(0, 2, 1, 3)
        #c = torch.cat([output1, output2], dim=1)
        return output2

    def calculate_loss(self, batch):
        lam = self.lam
        lreg = sum((torch.norm(param) ** 2 / 2) for param in self.parameters())

        labels = batch['y']
        y_predicted = self.predict(batch)

        y_true = self._scaler.inverse_transform(labels[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        loss = torch.mean(torch.norm(y_true - y_predicted) ** 2 / 2) + lam * lreg
        loss /= y_predicted.numel()
        # return loss.masked_mae_torch(y_predicted, y_true, 0)
        return loss

    def predict(self, batch):
        return self.forward(batch)
