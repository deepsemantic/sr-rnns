#
#       <State-Regularized Recurrent Neural Networks>
#
#   File:     <SR-RNNs>
#   Authors:  <Cheng Wang (cheng.wang@neclab.eu)> 
#             <Mathias Niepert (niepert.mathias@neclab.eu)>
#
# NEC Laboratories Europe GmbH, Copyright (c) <2019>, All rights reserved.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#        PROPRIETARY INFORMATION ---
#
# SOFTWARE LICENSE AGREEMENT
#
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
#
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
#
# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor.
#
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
#
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
#
# COPYRIGHT: The Software is owned by Licensor.
#
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
#
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
#
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
#
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
#
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
#
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
#
# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.
#
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
#
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
#
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
#
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.
#
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
#
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
#
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
#
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
#
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
#
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#


from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time

import numpy

import theano
import theano.typed_list
from theano import config
import theano.tensor as tensor

from graphviz import Digraph
import Queue

#from theano.tensor.shared_randomstreams import RandomStreams
import matplotlib.pyplot as plt

import pdb

from Optimizers import Optimizer
from Data_Untils import Data_Until

import argparse
parser = argparse.ArgumentParser()

class ParameterInitializator:

    def __init__(self, options):
     
        self.options = options
        
    def global_init_params(self):

        options = self.options
        params = OrderedDict()

        weight_filler = FillWeight()
        params['Wemb'] = weight_filler.uniform_weight(options['n_words'], options['dim_hidden']).astype(config.floatX)
        params = self.srgru_init_params(options, params, weight_filler, prefix='srgru')
        params['U'] = weight_filler.uniform_weight(options['dim_hidden'], options['n_classes']).astype(config.floatX)
        params['b'] = numpy.zeros((options['n_classes'],)).astype(config.floatX)
        params['centriods'] = weight_filler.uniform_centroids(options['n_centriods'], options['dim_hidden']).astype('float32')
        
        return params
        
    def srgru_init_params(self, options, params, weight_filler, prefix='SR_SRU'):
        '''
        Init the SR-GRU parameter:
        '''
        W = numpy.concatenate([weight_filler.ortho_weight(options['dim_hidden']),
                               weight_filler.ortho_weight(options['dim_hidden']),
                               weight_filler.ortho_weight(options['dim_hidden'])], axis=1)            
        params['srgru_W'] = W
        
        U = numpy.concatenate([weight_filler.ortho_weight(options['dim_hidden']),
                               weight_filler.ortho_weight(options['dim_hidden']),
                               weight_filler.ortho_weight(options['dim_hidden'])], axis=1)  
        params['srgru_U'] = U

        return params

    def tensor_params(self, params):
        tparams = OrderedDict()
        for kk, pp in params.items():
            tparams[kk] = theano.shared(params[kk], name=kk)
        return tparams
        
    def load_params(self, path, params):
        pp = numpy.load(path)
        for kk, vv in params.items():
            if kk not in pp:
                raise Warning('%s is not in the archive' % kk)
            params[kk] = pp[kk]

        return params
   
    
class FillWeight:

    def uniform_weight(self, n_in, n_out=None, fix=True):
        if n_out is None:
            n_out = n_in
        W = numpy.random.uniform(low=-numpy.sqrt(6. / (n_in + n_out)),
        high=numpy.sqrt(6. / (n_in + n_out)), size=(n_in, n_out))   
        #pdb.set_trace()        
        if fix:    
            W = numpy.random.uniform(low=-0.05, high=0.05, size=(n_in, n_out))
        return W.astype('float32')
        
    def uniform_centroids(self, n_in, n_out=None):
        if n_out is None:
            n_out = n_in           
        W = numpy.random.uniform(low=-0.5, high=0.5, size=(n_in, n_out))
        return W.astype('float32')
    
    def ortho_weight(self, ndim):
        W = numpy.random.randn(ndim, ndim)
        u, s, v = numpy.linalg.svd(W)
        return u.astype(config.floatX)
 
class Layers:

    def embedding_layer(self, tparams, x_in):
        x_out = tparams['Wemb'][x_in.flatten()]
        return x_out

    def srgru_layer(self, tparams, x, options, prefix='srgru', mask=None):

        n_timesteps = x.shape[0]
        n_batchsize = x.shape[1]
        
        n_centriods = options['n_centriods']
        dim_hidden = options['dim_hidden']
        

        def srgru_cell(m_, x_,  h_, transition_probs_, centroids):

            z = tensor.nnet.hard_sigmoid(tensor.dot(x_, tparams['srgru_U'][:,0:dim_hidden]) + tensor.dot(h_, tparams['srgru_W'][:,0:dim_hidden]))
            r = tensor.nnet.hard_sigmoid(tensor.dot(x_, tparams['srgru_U'][:,dim_hidden:2*dim_hidden]) + tensor.dot(h_, tparams['srgru_W'][:,dim_hidden:2*dim_hidden]))
            c = tensor.tanh(tensor.dot(x_, tparams['srgru_U'][:,2*dim_hidden:3*dim_hidden]) + tensor.dot(h_ * r, tparams['srgru_W'][:,2*dim_hidden:3*dim_hidden]))
            h = (tensor.ones_like(z) - z) * c + z * h_
          
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            #[batch_size x dim_hidden] dot [dim_hidden x n_centroids]--> [batch_size x n_centroids] 
            transition_probs = tensor.dot(h, centroids.T)    
            
            #[batch_size x n_centroids]
            transition_probs = tensor.nnet.softmax(transition_probs) 
            
            #[batch_size x n_centroids] dot [n_centroids x dim_hidden]--> [batch_size x dim_hidden] 
            h_new = tensor.dot(transition_probs, centroids) 
            
            h_new = m_[:, None] * h_new + (1. - m_)[:, None] * h
            #h_new = m2_[:, None] * h_new + (1. - m2_)[:, None] * h
            #transition_probs = m2_[:, None] * transition_probs + (1. - m2_)[:, None] * transition_probs_
            
            return h_new, transition_probs
 
        h = tensor.alloc(0., n_batchsize, dim_hidden)
        transition_probs = tensor.alloc(0., n_batchsize, options['n_centriods'])

        centriods = tparams['centriods']
      
        rval, updates = theano.scan(srgru_cell,
                                    sequences=[mask, x],
                                    outputs_info=[h,  transition_probs],
                                    name='srgru_layer',
                                    n_steps=n_timesteps,
                                    non_sequences = [centriods], 
                                    truncate_gradient=-1)
        return rval[0], rval[1]

def build_model(tparams, options):

    x = tensor.matrix('x', dtype='int64')

    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_batchsize = x.shape[1]

    layer = Layers()
    embedding = layer.embedding_layer(tparams, x).reshape([n_timesteps, n_batchsize, options['dim_hidden']])
    
    hidden_state, transition_probs = layer.srgru_layer(tparams, embedding, options, prefix='srgru', mask=mask)
    last_state = hidden_state[-1] 

    pred = tensor.nnet.softmax(tensor.dot(last_state, tparams['U']) + tparams['b'])
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')
    
    f_pred_states = theano.function([last_state], pred.argmax(axis=1), name='f_pred_state')

 
    return x, mask, y,  f_pred, f_pred_states, transition_probs

def train_srgru(
    dim_hidden=100,  # srgru number of hidden units.
    n_words=3,  # Vocabulary size 
    n_classes =2, # accept or reject
    maxlen=None,  # Sequence longer then this get ignored
    n_centriods=50,
    tomita_grammar='1'
):

    # Model options
    model_options = locals().copy()

    model_path = "models/srgru_tomita_%s.npz"%(tomita_grammar)
   
    print('Loading tomita_data') 
    data_until = Data_Until()
    train, valid = data_until.load_data(tomita_grammar)

    
    print('Loading pre-trained SR-GRU model')
    #pdb.set_trace()
    Parameter = ParameterInitializator(model_options)
    params = Parameter.global_init_params()
    params = Parameter.load_params(model_path, params)
    tparams = Parameter.tensor_params(params)
    
    print('Building Inference Models')
    x, mask,   y, f_pred, f_pred_states, transition_probs = build_model(tparams, model_options)

    f_weight_print = theano.function([x, mask], transition_probs)
    

    centriods_states=0
    for kk, vv in tparams.items():
        if kk == 'centriods':
            centriods_states = vv.get_value()
    
    pred_centriod_states=dict()
    centriod_index=1
    for centriod in centriods_states:
        #pdb.set_trace()
        state_predition = f_pred_states(centriod.reshape(1,-1))
        pred_centriod_states[centriod_index]=state_predition[0]
        centriod_index +=1
        
            
    #pdb.set_trace()
    print('Extracting DFAs by querying valid samples')
    transition_records=[]
    transition_types = []
    transition_instance=dict()
    correct_num = 0
    for starting_example in valid:
        
        active_states=[]

        x_, mask_ = data_until.prepare_data_inference(['2']+list(starting_example[0])) 
        predition = f_pred (x_, mask_)

        
        centriod_weights =f_weight_print(x_, mask_)
        seqs = []
        states =[]
        for i, w in zip(x_, centriod_weights):
            
            #pdb.set_trace()
            current_state = w[0].argmax(axis=0)+1
            #print (str(i[0])+'   '+str(current_state))
            w_ = [ '%.2f' % elem for elem in w[0] ]
            #print (str(i[0])+'   '+str(current_state)+'  '+str(w_))
            if len(active_states)==0:
                active_states.append(current_state)
            trans = (active_states[-1], current_state, str(i[0]))
            if trans not in transition_types:
                transition_types.append(trans)
                transition_instance[trans]=0
            else:
                transition_instance[trans] +=1

            active_states.append(current_state)
            #pdb.set_trace()
            #states.append(current_state)
            transition_records.append(transition_instance)

    return transition_instance, pred_centriod_states


    
def compare_state_transition(tran1,tran2):

    (s1, e1, l1), n1=tran1.items()[0]
    (s2, e2, l2), n2=tran2.items()[0]
    
    if n1>n2:
        return tran1
    else:
        return tran2

def construct_transition(transition_records):

    q = Queue.Queue()
    trans_num = dict()
    trans_states =[]   
    visited_nodes=[]

    start_state = get_start_node(transition_records)
    # check if the transition are same but with different label
    start_node = start_state.keys()[0][1]
    
    #pdb.set_trace()
    if start_state.keys()[0][1] not in visited_nodes:
        
        q.put(start_state.keys()[0][1])
        visited_nodes.append(start_state.keys()[0][1])
        
    transition_records.pop(start_state.keys()[0], None)
    while not q.empty():

        current_node = q.get()
        current_0_state, current_1_state  = get_nodes(transition_records, current_node)

        if  bool(current_0_state)==False and bool(current_1_state)==False:
            break
            
        if  bool(current_0_state)==False or bool(current_1_state)==False:
            current_1_state= current_0_state
            
        (s0, e0, l0), n0 = current_0_state.items()[0]
        
        (s1, e1, l1), n1 = current_1_state.items()[0]
        
        
        if s0== s1 and l0== l1 :

            current_state = compare_state_transition(current_0_state,current_1_state)
            (s_, e_, l_), n_ =current_state.items()[0]
            next_node = e_
            if next_node not in q.queue and next_node not in visited_nodes:
                    q.put(next_node)
                    visited_nodes.append(next_node)
            trans_states =check_and_add_states(trans_states, current_state)

            
        elif (s0,e0)== (s1,e1):
            if s0==e0:
                next_node = e0
                if next_node not in q.queue and next_node not in visited_nodes:
                    q.put(next_node)
                    visited_nodes.append(next_node)
                trans_states =check_and_add_states(trans_states, current_0_state)
                trans_states =check_and_add_states(trans_states, current_1_state)
            else:
                current_state = compare_state_transition(current_0_state,current_1_state)
                next_node = current_state.keys()[0][1]
                if next_node not in q.queue and next_node not in visited_nodes:
                    q.put(next_node)
                    visited_nodes.append(next_node)
                trans_states =check_and_add_states(trans_states, current_state)
        else:
            next_node = e0
            if next_node not in q.queue and next_node not in visited_nodes:
                    q.put(next_node)
                    visited_nodes.append(next_node)
                
            next_node = e1
            if next_node not in q.queue and next_node not in visited_nodes:
                    q.put(next_node)
                    visited_nodes.append(next_node)
            trans_states =check_and_add_states(trans_states, current_0_state)
            trans_states =check_and_add_states(trans_states, current_1_state)
        
        transition_records.pop(current_0_state.keys()[0], None)
        transition_records.pop(current_1_state.keys()[0], None)
        
        #print(trans_states)

    return trans_states, start_node

def check_and_add_states(trans_states, new_state):

    if len(trans_states)==0:
        trans_states.append(new_state)
    else:
        (s_, e_, l_), n_ =new_state.items()[0]
        length = len(trans_states)
        for t in range(length):
            (s, e, l), n =trans_states[t].items()[0]
            if s_==s and l_==l and n<n_:
                trans_states[t] = new_state
        trans_states.append(new_state)
    return trans_states
    
def get_start_node(transition_records):
    start_state=dict()
    
    for (s, e, l), n in transition_records.items(): 
        if l=='2':
             start_state[s, e, l]=n
            
    
    return start_state

def get_nodes(transition_records, node):
    trans_to_0 = []
    trans_to_1 = []
    S_0_state=dict()
    S_1_state=dict()
    
    for (s, e, l), n in transition_records.items(): 
        
        #pdb.set_trace()
        if s==node and l=='0':
            temp=dict()
            temp[s, e, l]=n
            trans_to_0.append(temp)
            #del transition_records[(s, e, l)]
        if s==node and l=='1':
            temp=dict()
            temp[s, e, l]=n
            trans_to_1.append(temp)
            #del transition_records[(s, e, l)]
    
    if len(trans_to_0)>1:
        #pdb.set_trace()
        S_0_state=select_one_state(trans_to_0)
    else:
        S_0_state=trans_to_0[0]
    if len(trans_to_1)>1:
        S_1_state=select_one_state(trans_to_1)  
    else:
        S_1_state=trans_to_1[0]      
    
    return S_0_state, S_1_state

def select_one_state(trans):
    #pdb.set_trace()
    state=dict()
    temp_max=0
    for t in trans:
        (s, e, l), n= t.items()[0]
        if  n > temp_max:
            temp_max =n
            state.clear()
            state[s,e,l] = n
    return state

def draw(dfa, pred_states, start_node, tomita_grammar='1'):
    
    dfa_nodes =[]
    for tran in dfa:
        (s, e, l), n=tran.items()[0]
        if s not in dfa_nodes:
            dfa_nodes.append(s)
        if e not in dfa_nodes:
            dfa_nodes.append(e)
        #dfa_edges.append((s, e,l))
    print(dfa)

    filename='DFAs/SRGRU_tomita_%s_dfa'%(tomita_grammar)
    g = Digraph('G', filename=filename, format='pdf')
    g.attr(rankdir='LR', size='8,5')


    for node in dfa_nodes:

        if pred_states[node]==1:
            if node==start_node:
                g.attr('node', shape='doublecircle', style='filled', color='gray80')
            else:
                g.attr('node', shape='doublecircle', style='filled', color='gray80')
            g.node(str(node))
        else:
            if node==start_node:
                g.attr('node', shape='doubleoctagon', style='filled', color='gray80')
            else:
                g.attr('node', shape='circle', style='filled', color='gray80')
            g.node(str(node))
    
   
    #pdb.set_trace()
    g.attr('node', shape='point', color="black")
    
    g.edge('', str(start_node), label='', color="black")
    for tran in dfa:
        (s, e, l), n=tran.items()[0]
        g.edge(str(s), str(e), label=str(l))

    g.render() 
    print('DFA extracted at: '+str(filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tomita_grammar', type=str, default='1', required=True,  help='1|2|3|4|7')
    args = parser.parse_args()
    
    transition_records, pred_centriod_states = train_srgru(tomita_grammar=args.tomita_grammar)
    dfa, start_node = construct_transition(transition_records)
    print('Generating the visualization of DFA ')
    draw(dfa, pred_centriod_states, start_node, args.tomita_grammar)
    
    
    
    
