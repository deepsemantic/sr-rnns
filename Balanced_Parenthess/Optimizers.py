
from collections import OrderedDict
import numpy

import theano
import theano.typed_list
from theano import config
import theano.tensor as tensor

class Optimizer:
    '''
    original code from "http://deeplearning.net/tutorial/lstm.html"
    '''
    def adadelta(self, lr, tparams, grads, x, mask, mask2, y, cost):
        """
        An adaptive learning rate optimizer

        Parameters
        ----------
        lr : Theano SharedVariable
            Initial learning rate
        tpramas: Theano SharedVariable
            Model parameters
        grads: Theano variable
            Gradients of cost w.r.t to parameres
        x: Theano variable
            Model inputs
        mask: Theano variable
            Sequence mask
        y: Theano variable
            Targets
        cost: Theano variable
            Objective fucntion to minimize

        Notes
        -----
        For more information, see [ADADELTA]_.

        .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
           Rate Method*, arXiv:1212.5701.
        """
        
     

        zipped_grads = [theano.shared(p.get_value() * 0.,
                                      name='%s_grad' % k)
                        for k, p in tparams.items()]
        running_up2 = [theano.shared(p.get_value() * 0.,
                                     name='%s_rup2' % k)
                       for k, p in tparams.items()]
        running_grads2 = [theano.shared(p.get_value() * 0.,
                                        name='%s_rgrad2' % k)
                          for k, p in tparams.items()]

        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
                 for rg2, g in zip(running_grads2, grads)]
                 
        '''   Mergre updates          ''' 
        updates = zgup + rg2up
        #pdb.set_trace()
       
        #updates.append(updates_1.items()[0])

        f_grad_shared = theano.function([x, mask, mask2,  y], cost, updates=updates,
                                        name='adadelta_f_grad_shared')

        updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
                 for zg, ru2, rg2 in zip(zipped_grads,
                                         running_up2,
                                         running_grads2)]
        ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
                 for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]
        
        '''   Mergre updates          ''' 
        updates = ru2up + param_up
        #pdb.set_trace()
        #updates.append(updates_1.items()[0])

        f_update = theano.function([lr], [], updates=updates,
                                   on_unused_input='ignore',
                                   name='adadelta_f_update')

        return f_grad_shared, f_update
    
    def rmsprop(self, lr, tparams, grads, x, mask, mask2, y, cost):
        """
        A variant of  SGD that scales the step size by running average of the
        recent step norms.

        Parameters
        ----------
        lr : Theano SharedVariable
            Initial learning rate
        tpramas: Theano SharedVariable
            Model parameters
        grads: Theano variable
            Gradients of cost w.r.t to parameres
        x: Theano variable
            Model inputs
        mask: Theano variable
            Sequence mask
        y: Theano variable
            Targets
        cost: Theano variable
            Objective fucntion to minimize

        Notes
        -----
        For more information, see [Hint2014]_.

        .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
           lecture 6a,
           http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        """

        zipped_grads = [theano.shared(p.get_value() * 0.,
                                      name='%s_grad' % k)
                        for k, p in tparams.items()]
        running_grads = [theano.shared(p.get_value() * 0.,
                                       name='%s_rgrad' % k)
                         for k, p in tparams.items()]
        running_grads2 = [theano.shared(p.get_value() * 0.,
                                        name='%s_rgrad2' % k)
                          for k, p in tparams.items()]

        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
                 for rg2, g in zip(running_grads2, grads)]

        f_grad_shared = theano.function([x, mask, mask2, y], cost,
                                        updates=zgup + rgup + rg2up,
                                        name='rmsprop_f_grad_shared')

        updir = [theano.shared(p.get_value() * 0.,
                               name='%s_updir' % k)
                 for k, p in tparams.items()]
        updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                     for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                                running_grads2)]
        param_up = [(p, p + udn[1])
                    for p, udn in zip(tparams.values(), updir_new)]
        f_update = theano.function([lr], [], updates=updir_new + param_up,
                                   on_unused_input='ignore',
                                   name='rmsprop_f_update')

        return f_grad_shared, f_update
        
    
    
