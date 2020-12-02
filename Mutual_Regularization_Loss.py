# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import pprint
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms



class FocalLoss(nn.Module):
    
    def __init__(self,gamma=2.0,alpha=0.25, name='FocalLoss'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred, y_true):

        gt = y_true
        heat = y_pred

        logpt = - F.binary_cross_entropy(heat, gt,reduction='none')
        pt = torch.exp(logpt)
        focal_loss = -( (1-pt)**self.gamma ) * logpt * (self.alpha*gt+(1-self.alpha)*(1-gt))
        return  focal_loss


class IntraConsistencyLoss(nn.Module):
    #intracons_loss = self.intracons_loss_func((a_heat_gt, s_heat_gt, e_heat_gt), (a_heat, s_heat, e_heat))
    def __init__(self, alpha, name='IntraConsistencyLoss'):
        super(IntraConsistencyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        a_gt, s_gt, e_gt = y_true
        a_heat, s_heat, e_heat = y_pred

        def _intra_consistency_loss(heat, gt):

            # heat_fea -> [N,window,1]
            # gt -> [N,window,1]
            # mask -> [N,window,1]

            # a = np.array([[0,1,1,1,1,0,0,0]]) shape: (1,8)
            # matmul(a.T, a)
            # Out: 
            #        [0, 0, 0, 0, 0, 0, 0, 0]             [0, 0, 0, 0, 0, 0, 0, 0]            [0, 1, 1, 1, 1, 0, 0, 0]           [0, 0, 0, 0, 0, 1, 1, 1]
            #        [0, 1, 1, 1, 1, 0, 0, 0]             [0, 0, 1, 1, 1, 0, 0, 0]            [1, 0, 0, 0, 0, 1, 1, 1]           [0, 0, 0, 0, 0, 0, 0, 0]
            #        [0, 1, 1, 1, 1, 0, 0, 0]             [0, 1, 0, 1, 1, 0, 0, 0]            [1, 0, 0, 0, 0, 1, 1, 1]           [0, 0, 0, 0, 0, 0, 0, 0]
            #        [0, 1, 1, 1, 1, 0, 0, 0]    ---->    [0, 1, 1, 0, 1, 0, 0, 0]            [1, 0, 0, 0, 0, 1, 1, 1]           [0, 0, 0, 0, 0, 0, 0, 0]
            #        [0, 1, 1, 1, 1, 0, 0, 0]             [0, 1, 1, 1, 0, 0, 0, 0]      &     [1, 0, 0, 0, 0, 1, 1, 1]           [0, 0, 0, 0, 0, 0, 0, 0]
            #        [0, 0, 0, 0, 0, 0, 0, 0]             [0, 0, 0, 0, 0, 0, 0, 0]            [0, 1, 1, 1, 1, 0, 0, 0]           [1, 0, 0, 0, 0, 0, 1, 1]
            #        [0, 0, 0, 0, 0, 0, 0, 0]             [0, 0, 0, 0, 0, 0, 0, 0]            [0, 1, 1, 1, 1, 0, 0, 0]           [1, 0, 0, 0, 0, 1, 0, 1]
            #        [0, 0, 0, 0, 0, 0, 0, 0]             [0, 0, 0, 0, 0, 0, 0, 0] M_gt_1     [0, 1, 1, 1, 1, 0, 0, 0] M_gt_2    [1, 0, 0, 0, 0, 1, 1, 0] M_gt_3  
            N = gt.shape[0]
            window = gt.shape[1]
            distance  = (heat.unsqueeze(-1)-heat.unsqueeze(1)).abs() # [N,window,window]
            # gt -> [N,window,1]
            gt_1 = gt.unsqueeze(-1).unsqueeze(1)  # [N,1,window,1]
            gt_2 = gt.unsqueeze(1).unsqueeze(1)  # [N,1,1,window]
            eye_mat = torch.diag_embed(torch.ones_like(gt))

            M_gt_1 = F.relu(torch.matmul(gt_1, gt_2) - eye_mat.unsqueeze(1)) # [N,1,window,window]
            M_gt_2 = (gt_1 - gt_2) * (gt_1 - gt_2) # [N,1,window,window]
            M_gt_3 = torch.ones_like(M_gt_1) - eye_mat.unsqueeze(1) - M_gt_1 - M_gt_2 # [N,1,window,window]
            pairs_1 = M_gt_1.sum(dim=(1,2,3))+1 # [N]
            pairs_2 = M_gt_2.sum(dim=(1,2,3)) + 1 # [N]
            pairs_3 = M_gt_3.sum(dim=(1,2,3)) + 1 # [N]
            consistency_1 = (distance.unsqueeze(1) * M_gt_1).sum(dim=(1,2,3)) / pairs_1 # [N]
            consistency_2 = 1 - (distance.unsqueeze(1) * M_gt_2).sum(dim=(1,2,3)) / pairs_2 # [N]
            consistency_3 = (distance.unsqueeze(1) * M_gt_3).sum(dim=(1,2,3)) / pairs_3 # [N]
            consistency_loss = consistency_1 + consistency_2 + consistency_3
            return consistency_loss

        intra_loss_action = _intra_consistency_loss(a_heat, a_gt)
        intra_loss_start = _intra_consistency_loss(s_heat, s_gt)
        intra_loss_end = _intra_consistency_loss(e_heat, e_gt)
        intra_loss = self.alpha * (intra_loss_action + intra_loss_start + intra_loss_end)

        return intra_loss

class InterConsistencyLoss(nn.Module):

    def __init__(self, alpha, name='InterConsistencyLoss'):
        super(InterConsistencyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        a_gt, s_gt, e_gt = y_true
        a_heat, s_heat, e_heat = y_pred

        def _inter_consistency_loss(action_heat, start_heat, end_heat):

            # action_heat -> [N,window]
            # start_heat -> [N,window]
            # end_heat -> [N,window]

            diff = torch.cat([action_heat[:,1:,]-action_heat[:,:-1,], action_heat[:,-1:,]-action_heat[:,-2:-1,]], 1) # [N,window]
            # diff = diff / tf.reduce_max(diff, [1,2]) # [N,window,1]
            zeros_tmp = torch.zeros_like(diff)
            diff_1 =   torch.where(diff>=0, diff,zeros_tmp) #tf.where(tf.greater_equal(diff, 0), diff, tf.zeros_like(diff))
            diff_0 = - torch.where(diff<=0, diff,zeros_tmp) #tf.where(tf.less_equal(diff, 0), diff, tf.zeros_like(diff))
            # start_heat = start_heat / tf.reduce_max(start_heat, [1,2])
            # end_heat = end_heat / tf.reduce_max(end_heat, [1,2])
            start_diff_consistency = (diff_1 - start_heat).abs()
            end_diff_consistency = (diff_0 - end_heat).abs()       
            consistency_loss = end_diff_consistency + start_diff_consistency
            return consistency_loss

        inter_loss = self.alpha * (_inter_consistency_loss(a_heat, s_heat, e_heat))

        return inter_loss



class Creatloss_base(nn.Module):
    def __init__(self, args):  
        super(Creatloss_base, self).__init__()
        self.args=args
        self.clas_loss_func = nn.BCELoss(reduction='none')
        #self.clas_loss_func = FocalLoss()
        self.regr_loss_func = nn.SmoothL1Loss(reduction='none')
        self.intracons_loss_func = IntraConsistencyLoss(alpha=self.args.inner_alpha)
        self.intercons_loss_func = InterConsistencyLoss(alpha=self.args.external_alpha)

    def clas_reweight(self,ori_loss,heat_gt,mask):
        nmask = mask - heat_gt
        pos_num = heat_gt.sum(1)
        neg_num = nmask.sum(1)
        pos_loss = (ori_loss * heat_gt).sum(1)/ (pos_num+1e-7)
        neg_loss = (ori_loss * nmask).sum(1)/ (neg_num+1e-7) 
        reweight_loss = 0.5 * (pos_loss + neg_loss)
        return reweight_loss #size: batch_size

    def forward(self, action_heat,start_heat,end_heat,start_bias,end_bias,target):
        a_heat, s_heat, e_heat, s_bias, e_bias = action_heat,start_heat,end_heat,start_bias,end_bias
        a_heat_gt = target[:,0,:]
        s_heat_gt = target[:,1,:]
        e_heat_gt = target[:,2,:]
        s_bias_gt = target[:,3,:]
        e_bias_gt = target[:,4,:]
        mask = target[:,5,:]
        a_clas_loss = self.clas_loss_func(a_heat, a_heat_gt)
        s_clas_loss = self.clas_loss_func(s_heat, s_heat_gt)
        e_clas_loss = self.clas_loss_func(e_heat, e_heat_gt)
        #clas_loss = (a_clas_loss + s_clas_loss + e_clas_loss) * mask
        # 使用reweighting过的分类loss 
        clas_loss = self.clas_reweight(a_clas_loss,a_heat_gt,mask) \
                    + self.clas_reweight(s_clas_loss,s_heat_gt,mask) \
                    + self.clas_reweight(e_clas_loss,e_heat_gt,mask)
        s_regr_loss = self.regr_loss_func(s_bias, s_bias_gt) * s_heat_gt
        e_regr_loss = self.regr_loss_func(e_bias, e_bias_gt) * e_heat_gt
        #regr_loss = s_regr_loss.sum(1)/s_heat_gt.sum(1) + e_regr_loss.sum(1)/e_heat_gt.sum(1)
        regr_loss = (s_regr_loss + e_regr_loss) * mask
        intracons_loss = self.intracons_loss_func((a_heat_gt, s_heat_gt, e_heat_gt), (a_heat, s_heat, e_heat))
        intercons_loss = self.intercons_loss_func((a_heat_gt, s_heat_gt, e_heat_gt), (a_heat, s_heat, e_heat)).mean(1)
        cons_loss = intracons_loss + intercons_loss
        #loss = clas_loss.mean(1) + 0.05 * regr_loss + cons_loss
        loss = clas_loss + regr_loss.mean(1) + cons_loss

        return loss, a_clas_loss, s_clas_loss, e_clas_loss, regr_loss, intracons_loss, intercons_loss



      









