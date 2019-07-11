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
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import matplotlib.pyplot as plt

import pdb

from Optimizers import Optimizer
from Data_Untils import Data_Until
from plot_transitions import *

SEED = 1234
numpy.random.seed(SEED)

import argparse
parser = argparse.ArgumentParser()


def _p(pp, name):
    return '%s_%s' % (pp, name)
    
class ParameterInitializator:

    def __init__(self, options):
     
        self.options = options
        
    def global_init_params(self):

        options = self.options
        params = OrderedDict()

        weight_filler = FillWeight()
        params['Wemb'] = weight_filler.uniform_weight(options['n_words'], options['dim_hidden']).astype(config.floatX)
        params = self.srlstmp_init_params(options, params, weight_filler, prefix='srlstmp')
        params['U'] = weight_filler.uniform_weight(options['dim_hidden'], options['n_classes']).astype(config.floatX)
        params['b'] = numpy.zeros((options['n_classes'],)).astype(config.floatX)
        params['srlstmp_centriods'] = weight_filler.uniform_centroids(options['n_centriods'], options['dim_hidden']).astype('float32')
        
        return params
        
    def srlstmp_init_params(self, options, params, weight_filler, prefix='srlstmp'):
        '''
        Init the srlstmp parameter:
        '''
        W = numpy.concatenate([weight_filler.ortho_weight(options['dim_hidden']),
                               weight_filler.ortho_weight(options['dim_hidden']),
                               weight_filler.ortho_weight(options['dim_hidden']),
                               weight_filler.ortho_weight(options['dim_hidden'])], axis=1)
        params[_p(prefix, 'W')] = W
        U = numpy.concatenate([weight_filler.ortho_weight(options['dim_hidden']),
                               weight_filler.ortho_weight(options['dim_hidden']),
                               weight_filler.ortho_weight(options['dim_hidden']),
                               weight_filler.ortho_weight(options['dim_hidden'])], axis=1)
        params[_p(prefix, 'U')] = U
        
        P = numpy.concatenate([weight_filler.uniform_centroids(options['dim_hidden']),
                           weight_filler.uniform_centroids(options['dim_hidden']),
                           weight_filler.uniform_centroids(options['dim_hidden'])], axis=1)
    #
        params[_p(prefix, 'P')] = P
        
        b_i = numpy.zeros((options['dim_hidden'],))
        b_f = numpy.ones((options['dim_hidden'],))
        b_o = numpy.zeros((options['dim_hidden'],))
        b_c = numpy.zeros((options['dim_hidden'],))
        #pdb.set_trace()
        b = numpy.concatenate([b_i, b_f, b_o, b_c],axis=0)
        params[_p(prefix, 'b')] = b.astype(config.floatX)

        return params
    def load_params(self, path, params):
        pp = numpy.load(path)
        for kk, vv in params.items():
            if kk not in pp:
                raise Warning('%s is not in the archive' % kk)
            params[kk] = pp[kk]

        return params
    def tensor_params(self, params):
        tparams = OrderedDict()
        for kk, pp in params.items():
            tparams[kk] = theano.shared(params[kk], name=kk)
        return tparams
   
    
class FillWeight:

    def uniform_weight(self, n_in, n_out=None, fix=False):
        if n_out is None:
            n_out = n_in
        W = numpy.random.uniform(low=-0.01*numpy.sqrt(6. / (n_in + n_out)),
        high=0.01*numpy.sqrt(6. / (n_in + n_out)), size=(n_in, n_out))   
        #pdb.set_trace()        
        if fix:    
            W = numpy.random.uniform(low=-0.05, high=0.05, size=(n_in, n_out))
        return W.astype('float32')
        
    def uniform_weight1(self, n_in, n_out=None):
        if n_out is None:
            n_out = n_in
        W = 0.01 * numpy.random.rand(n_in,n_out)
        
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

    def srlstmp_layer(self, tparams, x_in, options, prefix='srlstmp', mask=None, mask2=None):

        n_timesteps = x_in.shape[0]
        n_centriods = options['n_centriods']
        if x_in.ndim == 3:
            n_batchsize = x_in.shape[1]
        else:
            n_batchsize = 1

        
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]
        
            
        def _srlstmp_cell(m_, m2_, x_, h_, c_, transition_probs):
            #pdb.set_trace()
            preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
            preact += x_
            peephone_i = c_ * (tparams[_p(prefix, 'P')].T)[0] ## BxD * 1xD
            peephone_f = c_ * (tparams[_p(prefix, 'P')].T)[1]

            # gates
            i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_hidden'])+ peephone_i)
            f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_hidden'])+ peephone_i)
            
            
            c = tensor.tanh(_slice(preact, 3, options['dim_hidden']))
            # cell state
            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_
            
            peephone_o = c * (tparams[_p(prefix, 'P')].T)[2]
            o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_hidden'])+peephone_o)
            
            """ state-regularization """
            h = o * tensor.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_
            
            transition_probs = tensor.dot(h, tparams[_p(prefix, 'centriods')].T)  ## BxD dot DxK--> BxK
            transition_probs = tensor.nnet.softmax(transition_probs) # BxK
            h_new = tensor.dot(transition_probs, tparams[_p(prefix, 'centriods')])
            
            h_new = m2_[:, None] * h_new + (1. - m2_)[:, None] * h
 
            return h_new, c, transition_probs
                 

        x_in = (tensor.dot(x_in, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')])

        dim_hidden = options['dim_hidden']

        h = tensor.alloc(0., n_batchsize, dim_hidden)
        h_new = tensor.alloc(0., n_batchsize, dim_hidden)
        c = tensor.alloc(0., n_batchsize, dim_hidden)
        t = tensor.alloc(0., n_batchsize, options['n_centriods'])

        [h, c, transition_probs], updates = theano.scan(_srlstmp_cell,
                                    sequences=[mask, mask2, x_in],
                                    outputs_info=[h, c, t],
                                    name=_p(prefix, 'layers'),
                                    n_steps=n_timesteps, 
                                    truncate_gradient=options['truncate_grad'])

        return h, c, transition_probs

def build_model(tparams, options):

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    mask2 = tensor.matrix('mask2', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_batchsize = x.shape[1]

    layer = Layers()
    embedding = layer.embedding_layer(tparams, x).reshape([n_timesteps, n_batchsize, options['dim_hidden']])
    
    hidden_state, cell_state, transition_probs = layer.srlstmp_layer(tparams, embedding, options, prefix='srlstmp', mask=mask, mask2=mask2)
    f_vis = theano.function([x, mask, mask2], [hidden_state, cell_state, transition_probs], name='f_vis')
    
    return x, mask, mask2, f_vis

def train_srlstmp(
    dim_hidden=10,  # srlstmp number of hidden units.
    max_epochs=1000,  # The maximum number of epoch to run
    curr_epochs=10,
    dispFreq=100,  # Display to stdout the training progress every N updates
    lrate=0.01,  
    decay_parameter=0.0005,  # Weight decay
    n_words=30,  # Vocabulary size 
    n_classes =2, # accept or reject
    validFreq=1000,  # Compute the validation error after this number of update.
    saveFreq=100000,  # Save the parameters after every saveFreq updates
    maxlen=None,  # Sequence longer then this get ignored
    truncate_grad=-1,
    n_centriods=5, # the number of centroids
    optimizer="rmsprop",
    model_path = "models/srlstmp_bp.npz",
    log_path = "logs/srlstmp_bp.log",
):

    # Model options
    model_options = locals().copy()

    print('Weights Initialization')
    #pdb.set_trace()
    Parameter = ParameterInitializator(model_options)
    params = Parameter.global_init_params()
    
    params = Parameter.load_params(model_path, params)
    
    tparams = Parameter.tensor_params(params)
    
    print('Building Inference Models')
    x, mask, mask2, f_vis = build_model(tparams, model_options)
    
    
    example_x = '(a(b(f))))'
    example_y = '1'
    print(example_x)
    
    alphabet = "#abcdefghijklmnopqrstuvwxyz()$" ## '#' start token, '$' end token
    alphabet_dict = dict()
    for i in range(len(alphabet)):
        alphabet_dict[alphabet[i]] = i
    data_until = Data_Until(alphabet_dict)
    x_, mask_, mask2_ =data_until.prepare_data_inference([['#']+list(example_x)+['$']], maxlen=None)
    h, c, t = f_vis(x_, mask_, mask2_)
    plot_transition(example_x, h, c)


if __name__ == '__main__':
    train_srlstmp()
    
    
    
    
