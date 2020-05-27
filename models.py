import torch, math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import Iterable

######################## moe models #######################################
class Separate_MIMIC_Model(nn.Module):

    '''separate models for Jen's paper in PyTorch'''
    def __init__(self, experts):
        super(self.__class__, self).__init__()
        self.experts = experts

    def forward(self, x):
        return [expert(x) for expert in self.experts]

class Global_MIMIC_Cluster_Model(nn.Module):

    '''global model in Jen's paper in PyTorch used for clustering'''
    def __init__(self, input_dim, output_dim,
                 n_layers=1, units=16, num_dense_shared_layers=0,
                 dense_shared_layer_size=0, add_sigmoid=False):
        super(self.__class__, self).__init__()
        self.num_layers = n_layers
        self.hidden_size = units
        
        # global model
        self.lstm = nn.LSTM(input_dim, units, n_layers, batch_first=True)

        model = []
        # additional dense layers
        input_dim = units
        for l in range(num_dense_shared_layers):
            model.extend([nn.Linear(input_dim, dense_shared_layer_size),
                          nn.ReLU()])
            input_dim = dense_shared_layer_size

        # output layer
        model.append(nn.Linear(input_dim, output_dim))
        if add_sigmoid:
            model.append(nn.Sigmoid())
        self.rest = nn.Sequential(*model)

    def forward(self, x):
        '''assumes batch first'''
        batch_size = x.shape[0]
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        self.lstm.flatten_parameters()        
        o, (h, c) = self.lstm(x, (h, c))
        return self.rest(h[-1])

class Global_MIMIC_Model(nn.Module):

    '''global model in Jen's paper in PyTorch'''
    def __init__(self, n_layers, units, num_dense_shared_layers,
                 dense_shared_layer_size, input_dim, output_dim):
        super(self.__class__, self).__init__()
        self.num_layers = n_layers
        self.hidden_size = units
        
        # global model
        self.lstm = nn.LSTM(input_dim, units, n_layers, batch_first=True)

        model = []
        # additional dense layers
        input_dim = units
        for l in range(num_dense_shared_layers):
            model.extend([nn.Linear(input_dim, dense_shared_layer_size),
                          nn.ReLU()])
            input_dim = dense_shared_layer_size

        # output layer
        model.extend([nn.Linear(input_dim, output_dim),
                      nn.Sigmoid()])
        self.rest = nn.Sequential(*model)

    def forward(self, x):
        '''assumes batch first'''
        batch_size = x.shape[0]
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        self.lstm.flatten_parameters()        
        o, (h, c) = self.lstm(x, (h, c))
        return self.rest(h[-1])

class MoE_MIMIC_Model(nn.Module):

    ''' moe model for mimic dataset'''
    def __init__(self, input_dim, n_layers, units, num_dense_shared_layers,
                 dense_shared_layer_size, n_multi_layers, multi_units, output_dim, n_tasks):
        super(self.__class__, self).__init__()
        self.num_layers = n_layers
        self.hidden_size = units
        
        # shared part
        self.lstm = nn.LSTM(input_dim, units, n_layers, batch_first=True)
       
        model = []
        input_dim = units
        for l in range(num_dense_shared_layers):
            model.extend([nn.Linear(input_dim, dense_shared_layer_size),
                          nn.ReLU()])
            input_dim = dense_shared_layer_size
        self.shared = nn.Sequential(*model)

        # individual layers
        experts = nn.ModuleList()
        if n_multi_layers == 0:
            for task_num in range(n_tasks):
                experts.append(nn.Sequential(nn.Linear(input_dim, output_dim),
                                             nn.Sigmoid()))
            gating_function = MLP([input_dim, n_tasks])            
        else:
            for task_num in range(n_tasks):
                experts.append(nn.Sequential(MLP([input_dim, multi_units, output_dim]),
                                             nn.Sigmoid()))
            gating_function = MLP([input_dim, multi_units, n_tasks])                 

        gate = AdaptiveGate(input_dim, len(experts), forward_function=gating_function)
        self.moe = MoO(experts, gate)    
        
    def forward(self, x):
        '''assumes batch first'''
        batch_size = x.shape[0]
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        self.lstm.flatten_parameters()                
        o, (h, c) = self.lstm(x, (h, c))
        o = self.shared(h[-1])
        return torch.clamp(self.moe(o), 0, 1) # rarely need to be clamped, numerical issue in pytorch

class MTL_MIMIC_Model(nn.Module):

    ''' mtl model for mimic dataset'''
    def __init__(self, input_dim, n_layers, units, num_dense_shared_layers,
                 dense_shared_layer_size, n_multi_layers, multi_units, output_dim, tasks,
                 lstm=None, shared=None):
        super(self.__class__, self).__init__()
        self.num_layers = n_layers
        self.hidden_size = units
        n_tasks = len(tasks)
        
        # shared part
        if lstm is not None:
            self.lstm = lstm
        else:
            self.lstm = nn.LSTM(input_dim, units, n_layers, batch_first=True)
       
        model = []
        input_dim = units
        for l in range(num_dense_shared_layers):
            model.extend([nn.Linear(input_dim, dense_shared_layer_size),
                          nn.ReLU()])
            input_dim = dense_shared_layer_size

        if shared is not None:
            self.shared = shared
        else:
            self.shared = nn.Sequential(*model)

        # individual task layers: no need to learn gating function
        # later train using different sample weights defined by specific tasks/cohorts
        experts = nn.ModuleList()
        for task_num in range(n_tasks):
            mlp_layers = [input_dim] + [multi_units] * n_multi_layers + [output_dim]
            experts.append(nn.Sequential(MLP(mlp_layers),
                                         nn.Sigmoid()))

        self.experts = experts
        
    def forward(self, x):
        '''assumes batch first'''
        batch_size = x.shape[0]
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        self.lstm.flatten_parameters()                
        o, (h, c) = self.lstm(x, (h, c))
        o = self.shared(h[-1]) # use last layer
        
        return [expert(o) for expert in self.experts]

class Seq_AE_Model(nn.Module):

    ''' sequence autoencoder model for the mimic dataset'''
    def __init__(self, input_dim, units, n_layers=1):
        super(self.__class__, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = units
        self.encoder = nn.LSTM(input_dim, units, n_layers, batch_first=True)
        self.decoder = nn.LSTM(input_dim, units, n_layers, batch_first=True)

    def encoder_forward(self, x):
        '''assumes batch first'''
        batch_size = x.shape[0]
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        self.encoder.flatten_parameters()                
        o, (h, c) = self.encoder(x, (h, c))
        return h[-1] # use last layer
        
    def forward(self, x):
        '''assumes batch first and fix length, x (bs, T, d)'''
        batch_size, T, d = x.shape
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        encoded = self.encoder_forward(x).view(batch_size, 1, d)
        decoded = torch.expand(-1, T, -1)
        self.decoder.flatten_parameters()                
        o, (h, c) = self.decoder(decoded, (h, c))
        # o: (bs, T, num_directions*hidden_size)
        return o
    
######################## synthetic moe models #############################
class FIVNet(nn.Module):

    '''apply feature importance transformation before net '''
    def __init__(self, feature_importance, net):
        super(self.__class__, self).__init__()
        self.fi = feature_importance
        self.net = net
        
    def forward(self, x):
        return self.net(x*self.fi)

class FirstEntryGate(nn.Module):

    '''specific to my angle experiment'''
    def __init__(self, n_experts):
        super(self.__class__, self).__init__()
        self.n_experts = n_experts
        self.garbage = nn.Parameter(torch.tensor([1]).float()) # to satisfy pytorch
        
    def forward(self, x, coef=None):
        ret = torch.zeros(x.shape[0], self.n_experts)
        ret[torch.arange(x.shape[0]), x[:, 0].long()] = 1 # only assign to 1 expert
        return ret

class IgnoreFirstEntryModel(nn.Module):

    '''specific to my angle experiment, ignore the first entry'''
    def __init__(self, net):
        super(self.__class__, self).__init__()
        self.net = net
        
    def forward(self, x, coef=None):
        return self.net(x[:,1:])
    
class MLP(nn.Module):

    def __init__(self, neuron_sizes, activation=nn.LeakyReLU, bias=True): 
        super(MLP, self).__init__()
        self.neuron_sizes = neuron_sizes
        
        layers = []
        for s0, s1 in zip(neuron_sizes[:-1], neuron_sizes[1:]):
            layers.extend([
                nn.Linear(s0, s1, bias=bias),
                activation()
            ])
        
        self.classifier = nn.Sequential(*layers[:-1])

    def eval_forward(self, x, y):
        self.eval()
        return self.forward(x)
        
    def forward(self, x):
        x = x.contiguous()
        x = x.view(-1, self.neuron_sizes[0])
        return self.classifier(x)

class ID(nn.Module): # identity module

    def forward(self, x):
        if len(x.shape) < 2:
            return x.unsqueeze(0)
        return x

############################ main models ##################################
class MoE(nn.Module):

    ''' 
    This is a abstract base class for mixture of experts
    
    it supports:
    a) specifiying experts
    b) specifying the gating function (having parameter or not)

    it needs combining functions (either MoO or MoE)
    '''

    def __init__(self, experts, gate):
        super(MoE, self).__init__()        
        self.experts = experts
        self.gate = gate

class MoO(MoE):

    '''
    mixture of outputs
    '''
    def __init__(self, experts, gate, bs_dim=1, expert_dim=0):
        super(MoO, self).__init__(experts, gate)        
        # this is for RNN architecture: bs_dim = 2 for RNN
        self.bs_dim = bs_dim
        self.expert_dim = expert_dim

    def combine(self, o, coef):
        if type(o[0]) in [set, list]: # account for multi_output setting
            return [self.combine(o_, coef) for o_ in zip(*o)]
        else:
            o = torch.stack(o)
            # for RNN
            # wrong: (n_expert, bs, d) -> (d, bs, n_expert)
            # -> (d, n_expert, bs); this would be caught by
            # abc.Sequence for python3 as tensor is not a sequence
            # but tensor is iterable
            # correct: (n_expert, T, bs, d)) -> (d, T, bs, n_expert)
            # -> (d, T, bs, n_expert)
            # for others
            # (n_expert, bs, d) -> (d, bs, n_expert)
            # reshape o to (_, bs, n_expert)  b/c coef is (bs, n_expert)
            o = o.transpose(self.expert_dim, -1)
            o = o.transpose(self.bs_dim, -2) 

            # change back
            res = o * coef
            res = res.transpose(self.expert_dim, -1)
            res = res.transpose(self.bs_dim, -2)
            return res.sum(0)

    def ind_forward(self, coef, x): # for individual model training
        o = [expert(x) for expert in self.experts]
        return self.combine(o, coef)
        
    def forward(self, x, coef=None): # coef is previous coefficient: for IDGate
        coef = self.gate(x, coef) # (bs, n_expert) or n_expert
        self.last_coef = coef
        o = [expert(x) for expert in self.experts]
        return self.combine(o, coef)

class MoW(MoE):

    def forward(self, x, coef=None):
        # assume experts has already been assembled 
        coef = self.gate(x, coef)
        self.last_coef = coef
        return self.experts(x, coef)

################## sample gating functions for get_coefficients ###########
# todo:
# 1. play with gate initialization
# 2. play with regularization on gate output: total variation (smooth time transition) on similarity matrix, - sum(S) (encourage all shared), explicitly tying all weights: not helpful
# 3. play with whether adding softmax to coefficients makes sense: yes reduce variance

class Gate(nn.Module):

    '''
    gate function
    '''

    def forward(self, x, coef=None):
        raise NotImplementedError()

class AdaptiveGate(Gate):

    def __init__(self, input_size, num_experts,
                 normalize=True, requires_grad=True,
                 forward_function=None):
        super(self.__class__, self).__init__()

        if forward_function is None:
            self.forward_function = nn.Linear(input_size, num_experts)
        else:
            self.forward_function = forward_function

        if not requires_grad:
            self.forward_function = nn.Linear(input_size, num_experts, bias=False)
            # nn.init.orthogonal_(self.forward_function.weights)
            for param in self.forward_function.parameters():
                param.requires_grad = False
            
        self.normalize = normalize
        
    def forward(self, x, coef=None):
        o = self.forward_function(x) # (bs, num_experts)

        assert len(o.shape)==2, \
            "gate output must be (bs, num_experts), now {}".format(o.shape)        
        if self.normalize:
            return nn.functional.softmax(o, 1)
        else:
            return o

class AdaptiveLSTMGate(Gate):

    def __init__(self,
                 input_size, num_experts,
                 normalize=True, time_only=False,
                 use_h=True, use_c=False, use_t=True, use_x=True):
        super(self.__class__, self).__init__()
        self.forward_function = MLP([input_size, num_experts])
        self.normalize = normalize
        self.time_only = time_only

        self.use_h = use_h
        self.use_c = use_c
        self.use_t = use_t
        self.use_x = use_x
        
    def forward(self, x, coef=None):
        x, (h, c) = x # h (_, bs, d), x (_, bs, d)
        bs = h.shape[1]
        
        if self.time_only:
            t = torch.ones(bs, 1).cuda() * self.t            
            x = t
        else:
            to_cat = []
            if self.use_t:
                t = torch.ones(bs, 1).cuda() * self.t                            
                to_cat.append(t)
            if self.use_h:
                # print(h.shape, bs)                 
                h = h.transpose(0, 1).contiguous().view(bs, -1)
                to_cat.append(h)                
            if self.use_x:
                # print(x.shape, bs) 
                x = x.transpose(0, 1).view(bs, -1)
                to_cat.append(x)
            if self.use_c:
                c = c.transpose(0, 1).contiguous().view(bs, -1)
                to_cat.append(c)
            x = torch.cat(to_cat, 1)

        o = self.forward_function(x) # (bs, num_experts)
        assert len(o.shape)==2, \
            "gate output must be (bs, num_experts), now {}".format(o.shape)
        if self.normalize:
            # print(nn.functional.softmax(o, 1))
            return nn.functional.softmax(o, 1)
        else:
            return o
        
class NonAdaptiveGate(Gate):

    def __init__(self, num_experts, coef=None, fixed=False, normalize=True):
        '''
        fixed coefficient: resnet like with predefined not learnable gate values
        normalize: take softmax of the parameters
        '''
        super(self.__class__, self).__init__()
        self.normalize = normalize
        if coef is None: # initialization
            coef = torch.ones(num_experts)
            nn.init.uniform_(coef)
        if fixed:
            coef = nn.Parameter(coef, requires_grad=False)
        else:
            coef = nn.Parameter(coef)

        self.coefficients = coef

    def forward(self, x, coef=None):
        if self.normalize:
            return nn.functional.softmax(self.coefficients, 0)
        else:
            return self.coefficients

class IDGate(Gate): # identity gate

    def forward(self, x, coef): # coef is previous coefficient
        return coef

####### example networks ########
class ExampleAdaptive(nn.Module):
    # example usage of MoO and MoW (this case is the same)
    def __init__(self, input_size, hidden_size, num_classes):
        super(ExampleAdaptive, self).__init__()
        experts = nn.ModuleList([
            MLP([hidden_size, hidden_size]),
            MLP([hidden_size, hidden_size]),
            ID()
        ])

        self.layers = nn.Sequential(
            MLP([input_size, hidden_size]),
            nn.ReLU(),
            MoO(experts, AdaptiveGate(hidden_size, len(experts))),
            nn.ReLU(),
            MoO(experts, AdaptiveGate(hidden_size, len(experts))),
            nn.ReLU(),
            MLP([hidden_size, num_classes])
        )

    def forward(self, x):
        return self.layers(x)

class ExampleNonAdaptive(nn.Module):
    # example usage of MoO and MoW (this case is the same)
    def __init__(self, input_size, hidden_size, num_classes):
        super(ExampleNonAdaptive, self).__init__()
        experts = nn.ModuleList([
            MLP([hidden_size, hidden_size]),
            MLP([hidden_size, hidden_size]),
            ID()
        ])

        self.layers = nn.Sequential(
            MLP([input_size, hidden_size]),
            nn.ReLU(),
            MoO(experts, NonAdaptiveGate(len(experts))),
            nn.ReLU(),
            MoO(experts, NonAdaptiveGate(len(experts))),
            nn.ReLU(),
            MLP([hidden_size, num_classes])
        )

    def forward(self, x):
        return self.layers(x)

class ExampleFixed(nn.Module):
    # example usage of MoO and MoW (this case is the same)
    def __init__(self, input_size, hidden_size, num_classes):
        super(ExampleFixed, self).__init__()
        experts = nn.ModuleList([
            MLP([hidden_size, hidden_size]),
            MLP([hidden_size, hidden_size]),
            ID()
        ])

        self.layers = nn.Sequential(
            MLP([input_size, hidden_size]),
            nn.ReLU(),
            MoO(experts, NonAdaptiveGate(len(experts), fixed=True)),
            nn.ReLU(),
            MoO(experts, NonAdaptiveGate(len(experts), fixed=True)),
            nn.ReLU(),
            MLP([hidden_size, num_classes])
        )

    def forward(self, x):
        return self.layers(x)
    
class ExampleAdaptiveTie(nn.Module):
    # tying coefficients
    def __init__(self, input_size, hidden_size, num_classes):
        super(ExampleAdaptiveTie, self).__init__()
        experts = nn.ModuleList([
            MLP([hidden_size, hidden_size]),
            MLP([hidden_size, hidden_size]),
            ID()
        ])

        self.i2h = nn.Sequential(
            MLP([input_size, hidden_size]),
            nn.ReLU())

        self.h2h_1 = MoO(experts, AdaptiveGate(hidden_size, len(experts)))
        self.h2h_2 = MoO(experts, IDGate())

        self.h2o = nn.Sequential(
            MLP([hidden_size, num_classes])
        )

    def forward(self, x):
        x = self.i2h(x)
        x = self.h2h_1(x)
        coef = self.h2h_1.last_coef
        x = nn.functional.relu(x)
        x = self.h2h_2(x, coef)
        x = nn.functional.relu(x)        
        return self.h2o(x)

################ time series example models ################
class RNN_MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN_MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.model = MLP([input_size, hidden_size])

    def forward(self, x):
        x, hidden = x
        return self.model(x)

class BaseModelLSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 num_directions=1, dropout=0, activation=None):
        super(BaseModelLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        if dropout != 0:
            self.drop = nn.Dropout(p=dropout)
        self.bidirectional = (num_directions == 2)
        self.model = torch.nn.LSTM(self.input_size, self.hidden_size,
                                   num_layers=num_layers,
                                   bidirectional=self.bidirectional,
                                   dropout=dropout)
        self.h2o = nn.Linear(hidden_size * num_directions, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        if activation:
            self.activation = activation
        else:
            self.activation = lambda x: x

    def forward(self, x):
        x, hidden = x
        
        # x.shape: (seq_len, bs, _)
        o, hidden = self.model(x, hidden)

        if type(x) is torch.nn.utils.rnn.PackedSequence:
            # undo packing: second arg is input_lengths (we do not need)
            # o: seq_len x bs x d            
            o, _ = torch.nn.utils.rnn.pad_packed_sequence(o)
            
        seq_len, bs, d  = o.shape
        # dim transformation: (seq_len, bs, d) -> (seq_len x bs, d)
        o = o.view(-1, o.shape[2])

        if self.dropout != 0:
            o = self.drop(o)
        
        # run through prediction layer
        o = self.h2o(o)
        o = self.activation(o)

        # dim transformation
        o = o.view(seq_len, bs, self.output_size)

        return o, hidden
    
class ExampleMooLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes,
                 num_layers=1, num_directions=1, dropout=0, activation=None):
        super(ExampleMooLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.num_layers = num_layers
        self.num_directions = num_directions
        self.dropout = dropout
        self.activation = activation

    def setKT(self, k, t): # k models t steps
        '''k clusters with maximum of t time steps'''
        self.k = k
        self.T = t
        experts = nn.ModuleList()
        self.cells = nn.ModuleList()

        for _ in range(k):
            experts.append(BaseModelLSTM(self.input_size, self.hidden_size,
                                         self.num_classes, self.num_layers,
                                         self.num_directions, self.dropout,
                                         self.activation))

        for _ in range(t):
            gate = NonAdaptiveGate(self.k, normalize=True)
            self.cells.append(MoO(experts, gate,
                                  bs_dim=2, expert_dim=0))

    def forward(self, x, hidden, input_lengths):
        seq_len, bs, _ = x.shape
        o = []
        for t in range(seq_len):
            o_, hidden = self.cells[t]((x[t].view(1, bs, -1), hidden))
            o.append(o_)
            
        o = torch.cat(o, 0) # (seq_len, bs, _)

        return o, hidden

def moo_linear(in_features, out_features, num_experts, bs_dim=1,
               expert_dim=0, tie_weights=True, normalize=True):
    # repeat a linear model for self.num_experts times
    experts = nn.ModuleList()
    for _ in range(num_experts):
        experts.append(nn.Linear(in_features, out_features))

    # tie weights later
    if tie_weights:
        return MoO(experts, IDGate(), bs_dim=bs_dim, expert_dim=expert_dim)
    else:
        # linear expert, linear gate
        gate = AdaptiveGate(in_features, num_experts, normalize=normalize)
        return MoO(experts, gate, bs_dim=bs_dim, expert_dim=expert_dim)

class mowLSTM_(nn.Module):

    '''
    helper module for mowLSTM
    '''
    def __init__(self, input_size, hidden_size, num_experts=2, batch_first=False,
                 tie_weights=True, normalize=True):

        super(mowLSTM_, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.batch_first = batch_first

        # build cell
        self.input_weights = moo_linear(input_size, 4 * hidden_size,
                                        self.num_experts, bs_dim=1, # due to 1 step
                                        tie_weights=tie_weights,
                                        normalize=normalize) # i,f,g,o
        self.hidden_weights = moo_linear(hidden_size, 4 * hidden_size,
                                         self.num_experts, bs_dim=1, # due to 1 step
                                         tie_weights=tie_weights,
                                         normalize=normalize)
        # init same as pytorch version
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for m in self.input_weights.experts:
            for name, weight in m.named_parameters():
                nn.init.uniform_(weight, -stdv, stdv)
                # if 'weight' in name:
                #     nn.init.uniform_(weight)
        for m in self.hidden_weights.experts:
            for name, weight in m.named_parameters():
                nn.init.uniform_(weight, -stdv, stdv) 
                # if 'weight' in name:
                #     nn.init.orthogonal_(weight)
                
        # maybe: layer normalization: see jeeheh's code
        # maybe: orthogonal initialization: see jeeheh's code
        # note: pytorch implementation does neither

    def rnn_step(self, x, hidden, coef): # one step of rnn
        bs = x.shape[1]              
        h, c = hidden
        # need to squeeze to make sure adaptive_gate working correctly
        gates = self.input_weights(x.squeeze(0), coef) + \
                self.hidden_weights(h.squeeze(0), coef)
        # maybe: layer normalization: see jeeheh's code

        ingate, forgetgate, cellgate, outgate = gates.view(bs, -1).chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        c = forgetgate * c + ingate * cellgate
        h = outgate * torch.tanh(c) # maybe use layer norm here as well
        return h, c
    
    def forward(self, x, hidden, coef):
        if self.batch_first: # change to seq_len first
            x = x.transpose(0, 1)

        seq_len = x.shape[0]
        output = []
        for t in range(seq_len):
            hidden = self.rnn_step(x[t].unsqueeze(0), hidden, coef)
            output.append(hidden[0]) # seq_len x (_, bs, d)

        output = torch.cat(output, 0)
        return output, hidden

class mowLSTM(nn.Module):

    '''
    responsible for stacking and bidirectional LSTM
    stack according to 
    https://stackoverflow.com/questions/49224413/difference-between-1-lstm-with-num-layers-2-and-2-lstms-in-pytorch

    tie weights will use ID gate for each component
    '''
    def __init__(self, input_size, hidden_size, num_classes, num_experts=2,
                 num_layers=1, batch_first=False, dropout=0, bidirectional=False,
                 activation=None, tie_weights=True, normalize=True):
        
        super(mowLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.batch_first = batch_first
        self.dropouts = nn.ModuleList()

        self.h2o = moo_linear(self.num_directions * self.hidden_size,
                              self.num_classes, self.num_experts, bs_dim=1,
                              tie_weights=tie_weights, normalize=normalize)
        
        if activation:
            self.activation = activation
        else:
            self.activation = lambda x: x
        
        self.rnns = nn.ModuleList()
        for i in range(num_layers * self.num_directions):
            input_size = input_size if i == 0 else hidden_size
            self.rnns.append(mowLSTM_(input_size, hidden_size, num_experts,
                                      batch_first, tie_weights=tie_weights,
                                      normalize=normalize))
            self.dropouts.append(nn.Dropout(p=dropout))

    def forward(self, x, coef=None):
        x, hidden = x
        self.last_coef = coef
        
        h, c = hidden
        hs, cs = [], []
        for i in range(self.num_layers):
            if i != 0 and i != (self.num_layers - 1):
                x = self.dropouts[i](x) # waste 1 droput out but no problem
            x, hidden = self.rnns[i](x, (h[i].unsqueeze(0), c[i].unsqueeze(0)), coef)
            hs.append(hidden[0])
            cs.append(hidden[1])      

        # todo: bidirectional stacked LSTM, see reference here
        # https://github.com/allenai/allennlp/blob/master/allennlp/modules/stacked_bidirectional_lstm.py; it basically concat layer output

        h = torch.cat(hs, 0)
        c = torch.cat(cs, 0)
        o = x
        # run through prediction layer: o: (seq_len, bs, d)
        o = self.dropouts[0](o)

        # to have h2o work properly, i.e. each timestep has different h2o
        # we need seq_len to be 1
        seq_len = o.shape[0]
        assert seq_len == 1, "need seq len to be 1, now {}".format(seq_len)
        o = self.h2o(o.squeeze(0), coef).unsqueeze(0)
        o = self.activation(o)

        return o, (h, c)

class ExampleMowLSTM(nn.Module):

    '''
    recreate LSTM architectre
    then stack them according to 

    '''
    
    def __init__(self, input_size, hidden_size, num_classes,
                 num_layers=1, num_directions=1, dropout=0, activation=None):
        super(ExampleMowLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.dropout = dropout
        self.activation = activation

    def setKT(self, k, t): # k models t steps
        '''k experts with maximum of t time steps'''
        self.k = k
        self.T = t
        self.cells = nn.ModuleList()

        experts = mowLSTM(self.input_size, self.hidden_size,
                          self.num_classes, num_experts=self.k,
                          num_layers=self.num_layers, dropout=self.dropout,
                          bidirectional = (self.num_directions==2),
                          activation=self.activation)
        self.experts = experts
        
        for _ in range(t):
            gate = NonAdaptiveGate(self.k, normalize=True)
            self.cells.append(MoW(experts, gate))

    def forward(self, x, hidden, input_lengths):
        seq_len, bs, _ = x.shape
        o = []
        for t in range(seq_len):
            o_, hidden = self.cells[t]((x[t].view(1, bs, -1), hidden))
            o.append(o_)
            
        o = torch.cat(o, 0) # (seq_len, bs, d)
        return o, hidden

class LELG_LSTM(nn.Module): # linear expert linear gate

    '''
    don't tie weights, but currently shares gate for the entire layer, 
    not elementary wise
    '''
    def __init__(self, input_size, hidden_size, num_classes,
                 num_layers=1, num_directions=1, dropout=0, activation=None):
        super(LELG_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.dropout = dropout
        self.activation = activation

    def setKT(self, k, t): # k models t steps
        '''k experts with maximum of t time steps'''
        self.k = k
        self.T = t
        self.cells = nn.ModuleList()
        
        self.experts = mowLSTM(self.input_size, self.hidden_size,
                               self.num_classes, num_experts=self.k,
                               num_layers=self.num_layers, dropout=self.dropout,
                               bidirectional = (self.num_directions==2),
                               activation=self.activation,
                               tie_weights=False, normalize=True)

        for _ in range(t):
            experts = mowLSTM(self.input_size, self.hidden_size,
                              self.num_classes, num_experts=self.k,
                              num_layers=self.num_layers, dropout=self.dropout,
                              bidirectional = (self.num_directions==2),
                              activation=self.activation,
                              tie_weights=False, normalize=True)
            # replace moo_linear experts with the old ones
            experts.h2o.experts = self.experts.h2o.experts
            for m1, m2 in zip(experts.rnns, self.experts.rnns):
                m1.input_weights.experts = m2.input_weights.experts
                m1.hidden_weights.experts = m2.hidden_weights.experts          

            self.cells.append(experts)

    def forward(self, x, hidden, input_lengths):
        seq_len, bs, _ = x.shape
        o = []
        for t in range(seq_len):
            o_, hidden = self.cells[t]((x[t].view(1, bs, -1), hidden))
            o.append(o_)
            
        o = torch.cat(o, 0) # (seq_len, bs, d)
        return o, hidden
    
class ExampleME(nn.Module): # mixture of expert for LSTM setup 
    
    def __init__(self, input_size, hidden_size, num_classes,
                 num_layers=1, num_directions=1, dropout=0, activation=None):
        super(ExampleME, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.dropout = dropout
        self.activation = activation

    def setKT(self, k, t): # k models t steps
        '''k experts with maximum of t time steps'''
        self.k = k
        self.T = t

        experts = mowLSTM(self.input_size, self.hidden_size,
                          self.num_classes, num_experts=self.k,
                          num_layers=self.num_layers, dropout=self.dropout,
                          bidirectional = (self.num_directions==2),
                          activation=self.activation)
        self.experts = experts

        gate = AdaptiveLSTMGate(
            self.hidden_size *\
            self.num_layers *\
            self.num_directions + self.input_size + 1,
            self.k,
            normalize=True)
        self.cell = MoW(experts, gate)

    def forward(self, x, hidden, input_lengths):
        seq_len, bs, _ = x.shape
        o = []
        for t in range(seq_len):
            self.cell.gate.t = t # a hack 
            o_, hidden = self.cell((x[t].view(1, bs, -1), hidden))
            o.append(o_)
            
        o = torch.cat(o, 0) # (seq_len, bs, d)
        return o, hidden

class ExampleME2(nn.Module):

    '''
    only output is mixture of expert
    '''

    def __init__(self, input_size, hidden_size, num_classes,
                 num_layers=1, num_directions=1, dropout=0, activation=None):
        super(ExampleME2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.num_layers = num_layers
        self.num_directions = num_directions
        self.dropout = dropout
        self.activation = activation

    def setKT(self, k, t): # k models t steps
        '''k clusters with maximum of t time steps'''
        self.k = k
        self.T = t

        experts = nn.ModuleList()
        for _ in range(self.k):
            experts.append(nn.Linear(self.hidden_size * self.num_directions,
                                     self.num_classes))

        gate = AdaptiveGate(self.hidden_size * self.num_directions,
                            self.k,
                            normalize=True)
            
        self.h2o = MoO(experts, gate) # no need for bs_dim=2 b/c h2o will be reshaped
        self.cell = BaseModelLSTM(self.input_size, self.hidden_size,
                                  self.num_classes, self.num_layers,
                                  self.num_directions, self.dropout,
                                  self.activation)
        self.cell.h2o = self.h2o
        
    def forward(self, x, hidden, input_lengths):
        return self.cell((x, hidden))
    
class ExampleME3(nn.Module):

    '''
    each time step has a different gate
    '''
    def __init__(self, input_size, hidden_size, num_classes,
                 num_layers=1, num_directions=1, dropout=0, activation=None,
                 use_t=False, use_h=True, use_c=True, use_x=True):
        super(ExampleME3, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.dropout = dropout
        self.activation = activation
        self.use_h = use_h
        self.use_c = use_c
        self.use_t = use_t
        self.use_x = use_x
        
    def setKT(self, k, t): # k models t steps
        '''k experts with maximum of t time steps'''
        self.k = k
        self.T = t
        self.cells = nn.ModuleList()

        experts = mowLSTM(self.input_size, self.hidden_size,
                          self.num_classes, num_experts=self.k,
                          num_layers=self.num_layers, dropout=self.dropout,
                          bidirectional = (self.num_directions==2),
                          activation=self.activation)
        self.experts = experts

        for i in range(t):
            gate_size = 0
            if self.use_h:
                gate_size += self.hidden_size * self.num_layers * self.num_directions
            if self.use_c:
                gate_size += self.hidden_size * self.num_layers * self.num_directions
            if self.use_x:
                gate_size += self.input_size
            if self.use_t:
                gate_size += 1
            gate = AdaptiveLSTMGate(
                gate_size,
                self.k,
                use_x=self.use_x, use_c=self.use_c, use_h=self.use_h, use_t=self.use_t,
                normalize=True)
            self.cells.append(MoW(experts, gate))

    def forward(self, x, hidden, input_lengths):
        seq_len, bs, _ = x.shape
        o = []
        for t in range(seq_len):
            self.cells[t].gate.t = t # a hack
            o_, hidden = self.cells[t]((x[t].view(1, bs, -1), hidden))
            o.append(o_)
            
        o = torch.cat(o, 0) # (seq_len, bs, d)
        return o, hidden

class ExampleME4(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_classes,
                 num_layers=1, num_directions=1, dropout=0, activation=None):
        super(ExampleME4, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.dropout = dropout
        self.activation = activation

    def setKT(self, k, t): # k models t steps
        '''k experts with maximum of t time steps'''
        self.k = k
        self.T = t

        experts = mowLSTM(self.input_size, self.hidden_size,
                          self.num_classes, num_experts=self.k,
                          num_layers=self.num_layers, dropout=self.dropout,
                          bidirectional = (self.num_directions==2),
                          activation=self.activation)
        self.experts = experts

        gate = AdaptiveLSTMGate(
            1,
            self.k,
            normalize=True,
            time_only=True)
        self.cell = MoW(experts, gate)

    def forward(self, x, hidden, input_lengths):
        seq_len, bs, _ = x.shape
        o = []
        for t in range(seq_len):
            self.cell.gate.t = t # a hack 
            o_, hidden = self.cell((x[t].view(1, bs, -1), hidden))
            o.append(o_)
            
        o = torch.cat(o, 0) # (seq_len, bs, d)
        return o, hidden

class ExampleME5(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_classes,
                 num_layers=1, num_directions=1, dropout=0, activation=None):
        super(ExampleME5, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.dropout = dropout
        self.activation = activation

    def setKT(self, k, t): # k models t steps
        '''k experts with maximum of t time steps'''
        self.k = k
        self.T = t

        experts = mowLSTM(self.input_size, self.hidden_size,
                          self.num_classes, num_experts=self.k,
                          num_layers=self.num_layers, dropout=self.dropout,
                          bidirectional = (self.num_directions==2),
                          activation=self.activation)
        self.experts = experts

        gate = AdaptiveLSTMGate(
            1,
            self.k,
            normalize=True,
            time_only=True)
        gate.forward_function = MLP([1, 5, 5, self.k])
        self.cell = MoW(experts, gate)

    def forward(self, x, hidden, input_lengths):
        seq_len, bs, _ = x.shape
        o = []
        for t in range(seq_len):
            self.cell.gate.t = t # a hack 
            o_, hidden = self.cell((x[t].view(1, bs, -1), hidden))
            o.append(o_)
            
        o = torch.cat(o, 0) # (seq_len, bs, d)
        return o, hidden
    
    
######## for main_ff experiments
class Model_MoE(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, num_layers,
                 mode, normalize=True, use_relu=True, skip_threshold=1):
        '''
        number of layers are number of same structure layers
        skip_threshold determines whether or not to skip a layer
        '''
        modes = ['fixed', 'non_adaptive', 'adaptive', 'adaptive_fixed']
        assert mode in modes, 'wrong mode, must be in {}'.format(modes)
        nn.Module.__init__(self)

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for i in range(num_layers):
            experts = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size),
                ID(),
            ])
            if mode == 'fixed':
                gate = NonAdaptiveGate(len(experts),
                                       fixed=True,
                                       normalize=normalize)
            elif mode == 'non_adaptive':
                gate = NonAdaptiveGate(len(experts),
                                       normalize=normalize)
            elif mode == 'adaptive':
                gate = AdaptiveGate(hidden_size,
                                    len(experts),
                                    normalize=normalize)
            elif mode == 'adaptive_fixed':
                # remove this or previous mode later, is not really interesting
                gate = AdaptiveGate(hidden_size,
                                    len(experts),
                                    normalize=normalize,
                                    requires_grad=False)
                
            layers.append(MoO(experts, gate))
            if use_relu:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.BatchNorm1d(hidden_size))
                
        layers.extend([nn.Linear(hidden_size, num_classes)])
        self.classifier =  nn.Sequential(*layers)
        self.skip_threshold = skip_threshold
        self.mode = mode
        
    def forward(self, x):
        return self.classifier(x)

    def skip_forward(self, x):
        depth = 0
        for m in self.classifier:
            if type(m) == MoO:
                coef = m.gate(x) # (bs, num_experts)
                if self.mode in ['fixed', 'non_adaptive'] and \
                   coef[1].item() <= self.skip_threshold:
                    x = m(x)
                    depth += 1
                elif self.mode == 'adaptive':
                    o = m(x) # (bs, d), x: (bs, d)
                    ind = coef[:, 1] <= self.skip_threshold
                    x[ind] = o[ind]
                    depth += sum(ind).float().item() / x.shape[0]
            else:
                x = m(x)
        return x, depth

