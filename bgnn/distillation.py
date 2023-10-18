import copy
from selectors import EpollSelector
from tkinter import Y
from torch.nn.functional import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import math
import numpy as np
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)

class MLP_T(nn.Module):
    def __init__(self, dim):
        super(MLP_T, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class boost_kd(torch.nn.Module):
    r"""BGNN architecture for Graph representation learning.

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.

    .. note::
        `encoder` must have a `reset_parameters` method, as the weights of the target network will be initialized
        differently from the online network.
    """
    def __init__(self, encoder, teacher_encoder, train_size, FLAGS):
        super().__init__()
        self.dataset = FLAGS.dataset
        # online network
        self.student_encoder = encoder
        self.teacher_encoder = teacher_encoder

        self.gamma = FLAGS.gamma #weight of nll_loss
        self.alpha = FLAGS.alpha #weight of KD
        self.T = FLAGS.T #temperature

        self.boosting = FLAGS.boosting
        self.temp = FLAGS.temp

        self.train_size = train_size

        if self.temp:
            #### multi-teacher ####
            # self.MLP_T = MLP_T(FLAGS.n_classes+1)
           
            #### Single-teacher ####
            self.MLP_T = nn.Linear(self.train_size,self.train_size)

        self.n_classes = FLAGS.n_classes

        for param in self.teacher_encoder.parameters():
            param.requires_grad = False

    def reset_parameters(self):
        self.student_encoder.reset_parameters()

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        if self.temp:
            return list(self.student_encoder.parameters()) + list(self.MLP_T.parameters())
        else:
            return list(self.student_encoder.parameters())

    def cal_entropy(self, result):
        probs = F.softmax(result)
        return F.cross_entropy(probs, probs, reduction='none')

    def distillate_loss(self, s_1, s_2, t_1, t_2):
        return 2 - cosine_similarity(s_1, t_2.detach(), dim=-1).mean() - cosine_similarity(s_2, t_1.detach(), dim=-1).mean()

    def KD_loss(self, y_s, y_t, T):
    
        p_s = F.log_softmax(y_s/T, dim=1)
        p_t = F.softmax(y_t/T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (T**2) / y_s.shape[0]

        return loss

    def KD_loss_with_temp(self, y_s, y_t, T, max):
        p_s = F.softmax(y_s/T, dim=1)
        p_t = F.softmax(y_t/T, dim=1)
        loss = F.cross_entropy(p_s, p_t)
        return loss

    def forward(self, x, train_mask, sample_weights):
        s = self.student_encoder(x)

        with torch.no_grad():
            t = self.teacher_encoder(x).detach()
        
        if self.dataset != 'wiki-cs':
            s_result = s[train_mask]
            t_result = t[train_mask]
            true_result = x.y[train_mask].squeeze()
        else:
            s_result = s[train_mask[:, 0]]
            t_result = t[train_mask[:, 0]]
            true_result = x.y[train_mask[:, 0]]

        q_result = s_result.log_softmax(dim=-1)

        if self.boosting:
            tl_loss = (F.nll_loss(q_result, true_result, reduction='none') * sample_weights).sum()
        else:
            tl_loss = F.nll_loss(q_result, true_result)

        if self.temp:
            #### Multi-teacher ####
            # T = F.sigmoid(self.MLP_T(torch.cat((t_result, torch.reshape(self.cal_entropy(t_result), (t_result.size()[0],1))), 1)))
            # T = T*(self.T-1) + 1
            # T = T.repeat(1, self.n_classes)

            #### Single-teacher ####
            T = F.sigmoid(self.MLP_T(self.cal_entropy(t_result)))
            T = torch.reshape(T*(self.T-1) + 1, (T.size()[0],1))
            T = T.repeat(1, self.n_classes)

            kd_l = self.KD_loss_with_temp(s_result,t_result,T,self.T)
        else:
            kd_l = self.KD_loss(s_result,t_result, self.T)
 
        loss = self.gamma*tl_loss + self.alpha * kd_l

        return loss

    def evaluate(self, x, test_mask, train_masks, sample_weights):
        s = self.student_encoder(x)
        q_result = s[test_mask].log_softmax(dim=-1)
        true_result = x.y[test_mask].squeeze()
        
        _, indices = torch.max(q_result, dim=1)
        correct = torch.sum(indices == true_result)
        accs = correct.item() * 1.0 / len(true_result)

        wrong_result = []
        for indx, value in enumerate(indices):
            if value != true_result[indx]:
                wrong_result.append(indx)

         ## update weight
        output_logp = s.log_softmax(dim=-1)
        if self.dataset != 'wiki-cs':
            temp = F.nll_loss(output_logp[train_masks], x.y[train_masks], reduction='none')  # 140*1
        else: 
            temp = F.nll_loss(output_logp[train_masks[:,0]], x.y[train_masks[:, 0]], reduction='none')  # 140*1

        weight = sample_weights * torch.exp((1 - self.n_classes) / self.n_classes * temp)  # update weights
        weight = weight / weight.sum()
        weight.detach()

        return accs, wrong_result, weight

class Supervised_training(torch.nn.Module):
    r"""BGNN architecture for Graph representation learning.

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.

    .. note::
        `encoder` must have a `reset_parameters` method, as the weights of the target network will be initialized
        differently from the online network.
    """
    def __init__(self, encoder, FLAGS):
        super().__init__()
        self.dataset = FLAGS.dataset
        # online network
        self.student_encoder = encoder
        
        self.n_classes = FLAGS.n_classes

    def reset_parameters(self):
        self.student_encoder.reset_parameters()

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.student_encoder.parameters())

    def forward(self, x, train_mask):
        s = self.student_encoder(x)
        
        s_result = s[train_mask]
        true_result = x.y[train_mask].squeeze()

        q_result = s_result.log_softmax(dim=-1)
    
        loss = F.nll_loss(q_result, true_result)
        return loss

    def evaluate(self, x, test_mask, train_masks, sample_weights):
        s = self.student_encoder(x)
        ## calculate acc
        q_result = s[test_mask].log_softmax(dim=-1)
        true_result = x.y[test_mask].squeeze()
        
        _, indices = torch.max(q_result, dim=1)
        correct = torch.sum(indices == true_result)
        accs = correct.item() * 1.0 / len(true_result)

        ## update weight
        output_logp = s.log_softmax(dim=-1)
        if self.dataset != 'wiki-cs':
            temp = F.nll_loss(output_logp[train_masks], x.y[train_masks], reduction='none')  # 140*1
        else: 
            temp = F.nll_loss(output_logp[train_masks[:,0]], x.y[train_masks[:, 0]], reduction='none')  # 140*1
        weight = sample_weights * torch.exp((1 - self.n_classes) / self.n_classes * temp)  # update weights
        weight = weight / weight.sum()

        return accs, weight

#################################################################################
class boost_kd_link(torch.nn.Module):
    r"""BGNN architecture for Graph representation learning.

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.

    .. note::
        `encoder` must have a `reset_parameters` method, as the weights of the target network will be initialized
        differently from the online network.
    """
    def __init__(self, encoder, teacher_encoder, train_size, FLAGS):
        super().__init__()
        self.dataset = FLAGS.dataset
        # online network
        self.student_encoder = encoder
        self.teacher_encoder = teacher_encoder

        self.gamma = FLAGS.gamma #weight of nll_loss
        self.alpha = FLAGS.alpha #weight of KD
        self.T = FLAGS.T #temperature

        self.boosting = FLAGS.boosting
        self.temp = FLAGS.temp

        self.train_size = train_size

        if self.temp:
            #### multi-teacher ####
            # self.MLP_T = MLP_T(FLAGS.n_classes+1)
           
            #### Single-teacher ####
            self.MLP_T = nn.Linear(self.train_size,self.train_size)

        self.n_classes = FLAGS.n_classes

        for param in self.teacher_encoder.parameters():
            param.requires_grad = False

    def reset_parameters(self):
        self.student_encoder.reset_parameters()

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        if self.temp:
            return list(self.student_encoder.parameters()) + list(self.MLP_T.parameters())
        else:
            return list(self.student_encoder.parameters())

    def cal_entropy(self, result):
        #probs = F.softmax(result)
        #criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        #loss = criterion(result, result)
        probs = F.softmax(result, dim=1)
        # a = F.cross_entropy(probs, probs, reduction='none')
        return F.cross_entropy(probs, probs, reduction='none')
        #return loss
        #return F.binary_cross_entropy(result, result, reduction='none')

    def distillate_loss(self, s_1, s_2, t_1, t_2):
        return 2 - cosine_similarity(s_1, t_2.detach(), dim=-1).mean() - cosine_similarity(s_2, t_1.detach(), dim=-1).mean()

    def KD_loss(self, y_s, y_t, T):
    
        p_s = F.log_softmax(y_s/T, dim=1)
        p_t = F.softmax(y_t/T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (T**2) / y_s.shape[0]

        return loss

    def KD_loss_with_temp(self, y_s, y_t, T, max):
        # p_s = F.sigmoid(y_s/T)
        # p_t = F.sigmoid(y_t/T)
        # criterion = torch.nn.BCEWithLogitsLoss()
        # loss = criterion(p_s, p_t)
        # #loss = F.cross_entropy(p_s, p_t)
        # return loss
        #print(y_s.shape)
        #print(T.shape)
        p_s = F.softmax(y_s/T, dim=1)
        #print(p_s.shape)
        p_t = F.softmax(y_t/T, dim=1)
        #print(p_t.shape)
        loss = F.cross_entropy(p_s, p_t)
        return loss

    def forward(self, x, edge_label, edge_label_index):
        s = self.student_encoder(x, edge_label, edge_label_index)
        #print(s.shape)
        import torch
        with torch.no_grad():
            t = self.teacher_encoder(x, edge_label, edge_label_index).detach()
        
        if self.dataset != 'wiki-cs':
            #s_result = s[train_mask]
            #t_result = t[train_mask]
            s_result = self.student_encoder.decode(s,edge_label_index).view(-1)
            t_result = self.teacher_encoder.decode(t,edge_label_index).view(-1)
            #true_result = x.y[train_mask].squeeze()
        else:
            #s_result = s[train_mask[:, 0]]
            #t_result = t[train_mask[:, 0]]
            s_result = self.student_encoder.decode(s,edge_label_index).view(-1)
            t_result = self.teacher_encoder.decode(t,edge_label_index).view(-1)
            #true_result = x.y[train_mask[:, 0]]

        #q_result = s_result.log_softmax(dim=-1)
        #print((s_result>0.5).sum())
        p_s = torch.sigmoid(s_result)
        #print((p_s>0.5).sum())
        p_t = torch.sigmoid(t_result)

        if self.boosting:
            tl_loss = (F.nll_loss(q_result, true_result, reduction='none') * sample_weights).sum()
        else:
            #tl_loss = F.binary_cross_entropy(p_s, edge_label)
            criterion = torch.nn.BCEWithLogitsLoss()
            tl_loss = criterion(s_result, edge_label)

        if self.temp:
            #### Multi-teacher ####
            # T = F.sigmoid(self.MLP_T(torch.cat((t_result, torch.reshape(self.cal_entropy(t_result), (t_result.size()[0],1))), 1)))
            # T = T*(self.T-1) + 1
            # T = T.repeat(1, self.n_classes)

            #### Single-teacher ####
            #print("..............", self.cal_entropy(p_t))
            #T = F.sigmoid(self.MLP_T(self.cal_entropy(p_t)))
            #print("............", T.size())
            #T = torch.reshape(T*(self.T-1) + 1, (T.size()[0],1))
            #T = T.repeat(1, self.n_classes)
            T = F.sigmoid(self.MLP_T(self.cal_entropy(t)))
            T = torch.reshape(T*(self.T-1) + 1, (T.size()[0],1))
            T = T.repeat(1, 1)
            #print(T.shape)

            #kd_l = self.KD_loss_with_temp(s_result,t_result,T,self.T)
            kd_l = self.KD_loss_with_temp(s,t,T,self.T)
        else:
            kd_l = self.KD_loss(s_result,t_result, self.T)
 
        loss = self.gamma*tl_loss + self.alpha * kd_l
        #loss = tl_loss #+ self.alpha * kd_l

        return loss

    def evaluate(self, x, edge_label, edge_label_index):
        # s = self.student_encoder(x)
        # q_result = s[test_mask].log_softmax(dim=-1)
        # true_result = x.y[test_mask].squeeze()
        
        # _, indices = torch.max(q_result, dim=1)
        # correct = torch.sum(indices == true_result)
        # accs = correct.item() * 1.0 / len(true_result)

        # wrong_result = []
        # for indx, value in enumerate(indices):
        #     if value != true_result[indx]:
        #         wrong_result.append(indx)

        #  ## update weight
        # output_logp = s.log_softmax(dim=-1)
        # if self.dataset != 'wiki-cs':
        #     temp = F.nll_loss(output_logp[train_masks], x.y[train_masks], reduction='none')  # 140*1
        # else: 
        #     temp = F.nll_loss(output_logp[train_masks[:,0]], x.y[train_masks[:, 0]], reduction='none')  # 140*1

        # weight = sample_weights * torch.exp((1 - self.n_classes) / self.n_classes * temp)  # update weights
        # weight = weight / weight.sum()
        # weight.detach()

        #return accs, wrong_result, weight

        self.student_encoder.eval()
        z = self.student_encoder.forward(x, edge_label, edge_label_index)
        #print(z)
        out = self.student_encoder.decode(z, edge_label_index).view(-1).sigmoid()
        #print(out)
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(edge_label.cpu().detach().numpy(), out.cpu().detach().numpy())

    
    def decode(self, z, edge_label_index):
        out = self.student_encoder.decode(z, edge_label_index).view(-1)
        return out
#################################################################################



###########################################################
class Supervised_training_link(torch.nn.Module):
    r"""BGNN architecture for Graph representation learning.

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.

    .. note::
        `encoder` must have a `reset_parameters` method, as the weights of the target network will be initialized
        differently from the online network.
    """
    def __init__(self, encoder, FLAGS):
        super().__init__()
        self.dataset = FLAGS.dataset
        # online network
        self.student_encoder = encoder
        
        self.n_classes = FLAGS.n_classes

    def reset_parameters(self):
        self.student_encoder.reset_parameters()

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.student_encoder.parameters())

    def forward(self, x, edge_label, edge_label_index):
        s= self.student_encoder(x, edge_label, edge_label_index)
        out = self.decode(s, edge_label_index).view(-1)
        #print(out)
        
        # s_result = s[train_mask]
        # true_result = x.y[train_mask].squeeze()
        # s_result = s
        # true_result = edge_label

        # q_result = s_result.log_softmax(dim=-1)
    
        # loss = F.nll_loss(q_result, true_result)
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(out, edge_label)
        return s

    def evaluate(self, x, sample_weights):
        s = self.student_encoder(x)
        ## calculate acc
        # q_result = s[test_mask].log_softmax(dim=-1)
        # true_result = x.y[test_mask].squeeze()
        
        # _, indices = torch.max(q_result, dim=1)
        # correct = torch.sum(indices == true_result)
        # accs = correct.item() * 1.0 / len(true_result)

        out = self.decode(s, x.edge_label_index).view(-1).sigmoid()

        ## update weight
        output_logp = s.log_softmax(dim=-1)
        if self.dataset != 'wiki-cs':
            temp = F.nll_loss(output_logp[train_masks], x.y[train_masks], reduction='none')  # 140*1
        else: 
            temp = F.nll_loss(output_logp[train_masks[:,0]], x.y[train_masks[:, 0]], reduction='none')  # 140*1
        weight = sample_weights * torch.exp((1 - self.n_classes) / self.n_classes * temp)  # update weights
        weight = weight / weight.sum()

        return accs, weight
    
    def decode(self, z, edge_label_index):
        out = self.student_encoder.decode(z, edge_label_index).view(-1)
        return out

########################################################################
