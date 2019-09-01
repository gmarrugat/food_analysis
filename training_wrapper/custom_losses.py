# -*- coding: utf-8 -*-
from keras import backend as K
import numpy as np

def deactivate_loss():
    def loss(y_true, y_pred):
        cost = K.variable(0)
        return cost
    return loss

def uncertainty_categorical_crossentropy(auto_weight=True, norm=False):
    def loss(y_true, y_pred):
        # categorical cross entropy loss (L(W) single task labels)
        y_pred_top = y_pred[:,:-1] 

        if auto_weight:
            logsigma2 = y_pred[:,-1]
        else:
            logsigma2 = K.variable(0.0)

        cost = K.categorical_crossentropy(y_true, y_pred_top)
        # L(W,s)
        cost = K.exp(-logsigma2)*cost + logsigma2
        # Ln(W,s)
        if norm:
            #max_loss = -K.log(K.exp(-1.0)/(K.exp(1.0)*(float(K.int_shape(y_pred_top)[-1]))))
            max_loss = -K.log(0.001)
            cost = cost/max_loss
        cost = K.mean(cost)
        return cost
    return loss

def uncertainty_binary_crossentropy(auto_weight=True, norm=False):
    def loss(y_true, y_pred):
        # categorical cross entropy loss (L(W) single task labels)
        y_pred_top = y_pred[:,:-1] 

        if auto_weight:
            logsigma2 = y_pred[:,-1]
        else:
            logsigma2 = K.variable(0.0)

        cost = K.sum(K.binary_crossentropy(y_true, y_pred_top), axis=-1)
        # L(W,s)
        nlabels = K.cast_to_floatx(K.int_shape(y_true)[-1])
        cost = K.exp(-logsigma2)*cost + nlabels * logsigma2
        # Ln(W,s)
        if norm:
            max_loss = nlabels
            cost = cost/max_loss       
        cost =  K.mean(cost)
        return cost
    return loss

def regularizer_mtl_loss(rs_matrix, num_attributes, output_types, use_rsm=False, alpha_r=1, beta_r=1):

    def loss(y_true, y_pred):
        if not use_rsm:
            return K.variable(0) 
        
        y_true_list = []
        y_pred_list = []
        
        ntask = len(num_attributes)
        for i in range(ntask):
            start_index = i if i ==0 else np.sum(num_attributes[:i])
            end_index = np.sum(num_attributes[:i+1])
            y_pred_list.append(K.expand_dims(y_pred[:,int(start_index):int(end_index)],0))
            y_true_list.append(K.expand_dims(y_true[:,int(start_index):int(end_index)],0))
            
        # C(ntask, 2) = ntask!/((ntask-2)!*2!)
        nrel = np.math.factorial(ntask)/(np.math.factorial(ntask-2)*np.math.factorial(2))
        penalizer = 0
        for i in range(ntask-1):
            for j in range(i+1, ntask):
                #print i, j
                aux_i = i
                aux_j = j
                rs_matrix_ij = rs_matrix
                for k in range(ntask-2):
                    if aux_i!=0: 
                        rs_matrix_ij = np.max(rs_matrix_ij, axis=0)
                        #print "max 0"
                        aux_i = i-1
                        aux_j = j-1
                    elif aux_j!=1:
                        rs_matrix_ij = np.max(rs_matrix_ij, axis=1)
                        #print "max 1"
                        aux_j = j-1
                    else:
                        rs_matrix_ij = np.max(rs_matrix_ij, axis=2)
                        #print "max 2"
                n_rs_matrix_ij = 1 - rs_matrix_ij
                # No related attributes (R-)
                pred_rel = K.permute_dimensions(y_pred_list[i],(1,2,0))*K.permute_dimensions(y_pred_list[j],(1,0,2))
                penalizer_aux = K.sum(K.sum(pred_rel*n_rs_matrix_ij, axis=-1), axis=-1)
                # normalization of the penalization.
                if output_types[i] == "categorical" and output_types[j] == "binary":
                    penalizer_aux /= (num_attributes[j]-K.sum(K.min(rs_matrix_ij, axis=1)))
                elif output_types[i] == "binary" and output_types[j] == "categorical":
                    penalizer_aux /= (num_attributes[i]-K.sum(K.min(rs_matrix_ij, axis=0)))
                elif output_types[i] == "binary" and output_types[j] == "binary":
                    penalizer_aux /= (num_attributes[i]*num_attributes[j]-K.sum(rs_matrix_ij))
                elif output_types[i] == "categorical" and output_types[j] == "categorical":
                    penalizer_aux /= 1.0
                penalizer += penalizer_aux*alpha_r
                
                # GT penalizer (R+)
                gt_rel = K.permute_dimensions(y_true_list[i],(1,2,0)) * K.permute_dimensions(y_true_list[j],(1,0,2))
                penalizer_aux = K.sum(K.sum((1-pred_rel)*gt_rel, axis=-1), axis=-1)/K.sum(K.sum(gt_rel, axis=-1))
                penalizer += penalizer_aux*beta_r
                
        penalizer /= nrel
        
        return penalizer
    return loss
