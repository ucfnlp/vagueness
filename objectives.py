from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np
import h5py
# from six.moves import range

# semantic_mat_r=h5py.File('/dataset/word_vec/81labels_l2norm_fixed.mat','r')
# semantic_mat=semantic_mat_r.get('semantic_mat')
# semantic_mat=np.array(semantic_mat).astype("float32")
#semantic_mat=T.shared.np.array(semantic_mat)

if theano.config.floatX == 'float64':
    epsilon = 1.0e-9
else:
    epsilon = 1.0e-7


def mean_squared_error(y_true, y_pred):
    return T.sqr(y_pred - y_true).mean(axis=-1)


def mean_absolute_error(y_true, y_pred):
    return T.abs_(y_pred - y_true).mean(axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    return T.abs_((y_true - y_pred) / T.clip(T.abs_(y_true), epsilon, np.inf)).mean(axis=-1) * 100.


def mean_squared_logarithmic_error(y_true, y_pred):
    return T.sqr(T.log(T.clip(y_pred, epsilon, np.inf) + 1.) - T.log(T.clip(y_true, epsilon, np.inf) + 1.)).mean(axis=-1)


def squared_hinge(y_true, y_pred):
    return T.sqr(T.maximum(1. - y_true * y_pred, 0.)).mean(axis=-1)


def hinge(y_true, y_pred):
    return T.maximum(1. - y_true * y_pred, 0.).mean(axis=-1)

def pairwise_ranking(y_true, y_pred):
    Input_shape=y_true.shape

    T1=T.reshape(y_true,T.stack(Input_shape[0],1,Input_shape[1]))
    T2=T.reshape(y_true,T.stack(Input_shape[0],Input_shape[1],1))
    
    IW_difference_mat=T.clip(-T.batched_dot(T2,T1),0.,1.)
    
    label_predict=y_pred
    
    pos_socre_mat=label_predict*T.clip(y_true,0.,1.)
    neg_socre_mat=label_predict*T.clip(-y_true,0.,1.)
    
    IW_pos3=IW_difference_mat*T.addbroadcast(T.reshape(pos_socre_mat,T.stack(Input_shape[0],1,Input_shape[1])),1)
    IW_neg3=IW_difference_mat*T.addbroadcast(T.reshape(neg_socre_mat,T.stack(Input_shape[0],Input_shape[1],1)),2)
    
    return T.maximum(1. - IW_pos3 + IW_neg3, 0.).mean()

def RankNet_mean(y_true, y_pred):
    Input_shape=y_true.shape
    
    # weights
    positive_tags=T.clip(y_true,0.,1.)
    negative_tags=T.clip(-y_true,0.,1.)
    positive_tags_per_im=positive_tags.sum(axis=1)
    negative_tags_per_im=negative_tags.sum(axis=1)
    weight_per_image=positive_tags_per_im*negative_tags_per_im
    weight_per_image=T.reshape(weight_per_image,T.stack(Input_shape[0],1,1))
    weight_per_image=T.addbroadcast(weight_per_image,1)
    weight_per_image=T.addbroadcast(weight_per_image,2)

#instance wise part
    T1=T.reshape(y_true,T.stack(Input_shape[0],1,Input_shape[1]))
    T2=T.reshape(y_true,T.stack(Input_shape[0],Input_shape[1],1))
    
    IW_difference_mat=T.clip(-T.batched_dot(T2,T1),0.,1.)
    
    label_predict=y_pred
    
    pos_socre_mat=label_predict*T.clip(y_true,0.,1.)
    neg_socre_mat=label_predict*T.clip(-y_true,0.,1.)
    
    IW_pos3=IW_difference_mat*T.addbroadcast(T.reshape(pos_socre_mat,T.stack(Input_shape[0],1,Input_shape[1])),1)
    IW_neg3=IW_difference_mat*T.addbroadcast(T.reshape(neg_socre_mat,T.stack(Input_shape[0],Input_shape[1],1)),2)
    
    O=(IW_neg3[IW_difference_mat.nonzero()]-IW_pos3[IW_difference_mat.nonzero()])
    
    weight_matrix=T.tile(weight_per_image,(1,Input_shape[1],Input_shape[1]))
    
    return (T.log(1.+T.exp(O))/weight_matrix[IW_difference_mat.nonzero()]).mean()

def WARP(y_true, y_pred):
    Input_shape=y_true.shape
    
    # weights
    positive_tags=T.clip(y_true,0.,1.)
    negative_tags=T.clip(-y_true,0.,1.)
    positive_tags_per_im=positive_tags.sum(axis=1)
    negative_tags_per_im=negative_tags.sum(axis=1)
    weight_per_image=positive_tags_per_im*negative_tags_per_im
    weight_per_image=T.reshape(weight_per_image,T.stack(Input_shape[0],1,1))
    weight_per_image=T.addbroadcast(weight_per_image,1)
    weight_per_image=T.addbroadcast(weight_per_image,2)
    
    #instance wise part
    T1=T.reshape(y_true,T.stack(Input_shape[0],1,Input_shape[1]))
    T2=T.reshape(y_true,T.stack(Input_shape[0],Input_shape[1],1))
    
    IW_difference_mat=T.clip(-T.batched_dot(T2,T1),0.,1.)
    
    label_predict=y_pred
    
    pos_socre_mat=label_predict*T.clip(y_true,0.,1.)
    neg_socre_mat=label_predict*T.clip(-y_true,0.,1.)
    
    IW_pos3=IW_difference_mat*T.addbroadcast(T.reshape(pos_socre_mat,T.stack(Input_shape[0],1,Input_shape[1])),1)
    IW_neg3=IW_difference_mat*T.addbroadcast(T.reshape(neg_socre_mat,T.stack(Input_shape[0],Input_shape[1],1)),2)
    
    pos_neg_dif_mat=IW_neg3-IW_pos3
    

    #adaptive weights
    
    adapative_rank_matrix=T.clip(pos_neg_dif_mat+1.,0.,1.)
    adapative_rank_matrix=T.ceil(adapative_rank_matrix)
    neg_vol_count=T.sum(adapative_rank_matrix,axis=1)
    
    positive_tags_per_im=T.addbroadcast(T.reshape(positive_tags_per_im,T.stack(Input_shape[0],1)),1)
    rank_weight1=neg_vol_count*(1./positive_tags_per_im+neg_vol_count/positive_tags_per_im)/2
    rank_weight=T.addbroadcast(T.reshape(rank_weight1,T.stack(Input_shape[0],1,Input_shape[1])),1)
    pos_neg_dif_mat=pos_neg_dif_mat*rank_weight
    
    pos_neg_dif_vec=pos_neg_dif_mat[IW_difference_mat.nonzero()]
    return (T.maximum(1.+pos_neg_dif_vec,0.)).mean()    
        
def global_pairwise(y_true, y_pred):
    flatted_prediction=y_pred
    flatted_GT=y_true
    diffe_label_mat=-T.outer(flatted_GT,flatted_GT)# same is -1 otherwise 1
    diffe_label_mat=T.clip(diffe_label_mat,0.,1.)
    pos_to_add=-T.outer(T.clip(flatted_GT,0.,1.),flatted_prediction)
    neg_to_sub=T.outer(flatted_prediction,T.clip(-flatted_GT,0.,1.))
    return T.maximum(0.,1.+(pos_to_add+neg_to_sub)*diffe_label_mat).mean()


def categorical_crossentropy(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes
    '''
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    # scale preds so that the class probas of each sample sum to 1
    y_pred /= y_pred.sum(axis=-1, keepdims=True)
    cce = T.nnet.categorical_crossentropy(y_pred, y_true)
    return cce


def binary_crossentropy(y_true, y_pred):
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    bce = T.nnet.binary_crossentropy(y_pred, y_true).mean(axis=-1)
    return bce


def poisson_loss(y_true, y_pred):
    return T.mean(y_pred - y_true * T.log(y_pred + epsilon), axis=-1)

# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error

# from .utils.generic_utils import get_from_module
# def get(identifier):
#     return get_from_module(identifier, globals(), 'objective')
