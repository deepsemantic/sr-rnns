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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pdb, numpy
from collections import defaultdict
from random import choice, random

class BP(object):


    depth_counter = 0
    depth = 5

    def __init__(self, depth):
        self.prod = defaultdict(list)
        self.depth = depth

    def add_prod(self, lhs, rhs):
        """ Add production to the grammar. 'rhs' can
            be several productions separated by '|'.
            Each production is a sequence of symbols
            separated by whitespace.

            Usage:
                grammar.add_prod('NT', 'VP PP')
                grammar.add_prod('Digit', '1|2|3|4')
        """
        prods = rhs.split('|')
        for prod in prods:
            self.prod[lhs].append(tuple(prod.split()))

    def gen_random(self, symbol, curr_len = 0):
        """ Generate a random sentence from the
            grammar, starting with the given
            symbol.
        """
       
        sentence = ' '
        #s = ''
        is_valid = True
        while (is_valid):
            # select one production of this symbol randomly
            rand_prod = choice(self.prod[symbol])
             #print('rand_prod:'+ str(rand_prod))
            if rand_prod == "( S )":
                if self.depth_counter > 5:
                    continue
                else:
                    self.depth_counter += 1
            is_valid=False

        for sym in rand_prod:
            # for non-terminals, recurse
            if sym in self.prod:
                sentence += self.gen_random(sym, curr_len)[0]
            else:
                sentence += sym + ' '
            curr_len = len(sentence.replace(' ', ''))
            if curr_len>100:
                break
                
        self.depth_counter = 1
        #print(sentence)
        
        
        return sentence, curr_len

def get_depth(S):
    current_max = 0
    max_depth = 0
    n = len(S)
 
    # Traverse the input string
    for i in xrange(n):
        if S[i] == '(':
            current_max += 1
 
            if current_max > max_depth:
                max_depth = current_max
        elif S[i] == ')':
            if current_max > 0:
                current_max -= 1
            else:
                return -1
 
    # finally check for unbalanced string
    if current_max != 0:
        return -1
 
    return max_depth


def check_label(bp_str):
    open_number = 0
    close_number = 0
    
    for i in bp_str:
        if i == '(':
            open_number += 1
        if i == ')':
            close_number += 1
    label = 1 if open_number - close_number == 0 else 0
    return label


def sample_letter():
    alphabet = 'abcdefghijklmnopqrstuvwxyz()'
    index = numpy.random.randint(0, len(alphabet))
    return alphabet[index]
    
def labeling_samples(samples, min_depth, max_depth, max_len):
    
    pos_samples = []
    neg_samples = []
    
    for sample in samples:
        label = check_label(sample)
        depth = get_depth(sample)
        length = len(sample)
        if label == 1 and depth>=min_depth and depth <= max_depth and length <= max_len:
            pos_samples.append((sample, label))
        if label == 0 and depth>=min_depth and depth <= max_depth and length <= max_len:
            neg_samples.append((sample, label))
    
    '''
    modify positive samples to negative ones
    
    '''
    for sample, label in pos_samples:

        ## do randomly sample other letters or symbols to replace orginal ones
        mask_sym = numpy.random.binomial(1, 0.5, len(sample))
        new_sample=''
        #print(sample)
        for m_, s_ in zip(mask_sym, sample):
            if m_ == 0:
               new_sample += s_
            else:  # make some changes on original sample
               new_sample += sample_letter()   
                
            
        new_label = check_label(new_sample)
        new_depth = get_depth(new_sample)
        new_length = len(new_sample)

        if new_label == 0 and new_length <= max_len:
            neg_samples.append((new_sample, new_label))
    
    return pos_samples, neg_samples
            
def sample_generator(max_number, max_len, min_depth, max_depth, dataset=''): 
 
    bp = BP(max_depth)
    bp.add_prod('S', '( S )|S S|V')
    bp.add_prod('V', 'a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z')
    
    samples=[]
    for i in xrange(max_number):   
       sample, _ = bp.gen_random('S')
       #print(sample)
       sample = sample.replace(' ', '')
       if sample not in samples:
           samples.append(sample)
    pos_samples, neg_samples = labeling_samples(samples, min_depth, max_depth, max_len) 
    
    train = pos_samples[:]
    train.extend(neg_samples)
    
    with open('data/dataset_info.txt', 'a') as the_file:
        the_file.write('generated ' +dataset + ' '+ str(len(train))+ ' samples ' + ' positive: '+ str(len(pos_samples))+ ' negative: '+ str(len(neg_samples))+'\n')
        
    print('generated ' +dataset + ' '+ str(len(train))+ ' samples ' + ' positive: '+ str(len(pos_samples))+ ' negative: '+ str(len(neg_samples)))
    
    
    return train
        
if __name__ == '__main__':
    min_depth = 1
    max_depth = 5
    max_number = 2000
    max_len = 100
  
    train = sample_generator(max_number, max_len, min_depth, max_depth, 'train') 

    min_depth = 6
    max_depth = 10
    max_number = 2000
    max_len = 100
    valid = sample_generator( max_number, max_len, min_depth, max_depth, 'valid') 
    #pdb.set_trace()
    numpy.save('data/bp_train_small.npy', train)
    numpy.save('data/bp_valid_small.npy', valid)
    


