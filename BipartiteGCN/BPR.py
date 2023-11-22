from KnowledgeTracing.Constant import Constants as C
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import collections

user_num = C.user_num
item_num = C.item_num

class BPR(nn.Module):
    def __init__(self, user_num,item_num,factor_num,user_item_matrix,item_user_matrix):
        super(BPR, self).__init__()

        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = item_user_matrix

    def forward(self, embed_user, embed_item, stu=None, que_i=None, que_j=None):

        users_embedding=embed_user
        items_embedding=embed_item

        gcn1_users_embedding = torch.sparse.mm(self.user_item_matrix, items_embedding) #+ users_embedding.mul(self.d_i_train))#*2. #+ users_embedding
        gcn1_items_embedding = torch.sparse.mm(self.item_user_matrix, users_embedding) #+ items_embedding.mul(self.d_j_train))#*2. #+ items_embedding

        gcn2_users_embedding = torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) #+ gcn1_users_embedding.mul(self.d_i_train))#*2. + users_embedding
        gcn2_items_embedding = torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) #+ gcn1_items_embedding.mul(self.d_j_train))#*2. + items_embedding
        
        gcn3_users_embedding = torch.sparse.mm(self.user_item_matrix, gcn2_items_embedding) #+ gcn2_users_embedding.mul(self.d_i_train))#*2. + gcn1_users_embedding
        gcn3_items_embedding = torch.sparse.mm(self.item_user_matrix, gcn2_users_embedding) #+ gcn2_items_embedding.mul(self.d_j_train))#*2. + gcn1_items_embedding

        gcn_users_embedding = users_embedding + (1/2)*gcn1_users_embedding + (1/3)*gcn2_users_embedding + (1/4)*gcn3_users_embedding
        gcn_items_embedding = items_embedding + (1/2)*gcn1_items_embedding + (1/3)*gcn2_items_embedding + (1/4)*gcn3_items_embedding

        # stu = F.embedding(stu,gcn_users_embedding)
        # que_i = F.embedding(que_i,gcn_items_embedding)
        # que_j = F.embedding(que_j,gcn_items_embedding)  
        # # # pdb.set_trace()
        # prediction_i = (stu * que_i).sum(dim=-1)
        # prediction_j = (user * que_j).sum(dim=-1)

        # l2_regulization = 0.0001*(user**2+item_i**2+item_j**2).sum(dim=-1)

        # loss_ = -((prediction_i - prediction_j).sigmoid().log().mean())
        # loss = -((prediction_i - prediction_j)).sigmoid().log().mean() + l2_regulization.mean()

        return gcn_items_embedding

def readD(set_matrix,num_):
    user_d=[]
    for i in range(num_):
        len_set=1.0/(len(set_matrix[i])+1)  
        user_d.append(len_set)
    return user_d

def readTrainSparseMatrix(set_matrix,u_d,i_d,is_user,is_train=1):
    user_items_matrix_i=[]
    user_items_matrix_v=[]
    if is_user:
        d_i=u_d
        d_j=i_d
    else:
        d_i=i_d
        d_j=u_d
    for i in set_matrix:
        # len_set=len(set_matrix[i])
        for j in set_matrix[i]:
            user_items_matrix_i.append([i,j])
            if is_train:
                d_i_j=np.sqrt(d_i[i]*d_j[j])
            else:
                d_i_j=d_i[i]*d_j[j]
            #1/sqrt((d_i+1)(d_j+1))
            user_items_matrix_v.append(d_i_j)#(1./len_set) 

    # user_items_matrix_i=torch.cuda.LongTensor(user_items_matrix_i)
    # user_items_matrix_v=torch.cuda.FloatTensor(user_items_matrix_v)
    user_items_matrix_i=torch.tensor(user_items_matrix_i, dtype=torch.long, device='cuda')
    user_items_matrix_v=torch.tensor(user_items_matrix_v, dtype=torch.float, device='cuda')
    # return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)
    return torch.sparse_coo_tensor(user_items_matrix_i.t(), user_items_matrix_v, dtype=torch.float, device='cuda')

def bipartite(trainLoaders, testLoaders):

    Q,A = trainLoaders[1],trainLoaders[2]
    Q_,A_ = testLoaders[1],testLoaders[2]

    Q = np.concatenate((Q, Q_), axis=0)
    A = np.concatenate((A, A_), axis=0)

    training_user_set = collections.defaultdict(set)
    training_item_set = collections.defaultdict(set)
    training_user_set_ = collections.defaultdict(set)
    training_item_set_ = collections.defaultdict(set)

    for i in range(Q.shape[0]):
        for j in range(C.MAX_STEP):
            if A[i][j] > 0:
                training_user_set[i].add(Q[i][j]-1)
                training_item_set[Q[i][j]-1].add(i)
            elif A[i][j] == 0:
                training_user_set_[i].add(C.NUM_OF_QUESTIONS + Q[i][j]-1)
                training_item_set_[C.NUM_OF_QUESTIONS + Q[i][j]-1].add(i)

    training_user_set[user_num-1].add(item_num-1)
    training_item_set[item_num-1].add(user_num-1)
    training_user_set_[user_num-1].add(item_num-1)
    training_item_set_[item_num-1].add(user_num-1)

    u_d=readD(training_user_set,user_num)
    i_d=readD(training_item_set,item_num)
    u_d_=readD(training_user_set_,user_num)
    i_d_=readD(training_item_set_,item_num)

    sparse_u_i=readTrainSparseMatrix(training_user_set,u_d,i_d,True)
    sparse_i_u=readTrainSparseMatrix(training_item_set,u_d,i_d,False)
    sparse_u_i_=readTrainSparseMatrix(training_user_set_,u_d_,i_d_,True)
    sparse_i_u_=readTrainSparseMatrix(training_item_set_,u_d_,i_d_,False)

    return [sparse_u_i, sparse_i_u, sparse_u_i_, sparse_i_u_]
