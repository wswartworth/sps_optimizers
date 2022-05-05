import numpy as np
import torch
import time
import copy

class SP2_base(torch.optim.Optimizer):

    def __init__(self, params, stepsize=1, beta=0):

        params = list(params)
        super().__init__(params, {})
        self.params = params

        self.state['step'] = 0

        self.stepsize = stepsize

        #momentum
        zero_fill = [torch.zeros_like(w_i) for w_i in params]
        self.last_two_steps = [zero_fill, zero_fill] #most recent step, followed by the previous step
        self.beta = beta

    def update_momentum(self, step):
        self.last_two_steps[0], self.last_two_steps[1] = step, self.last_two_steps[0]

    def get_momentum(self):
        return [self.beta * (a-b) for a,b in zip(self.last_two_steps[0], self.last_two_steps[1])]

    def step(self, closure=None, loss=None):

        '''compute and perform a step, typically pass in a loss'''

        if loss is None and closure is None:
            raise ValueError('specify either closure or loss')

        if loss is not None:
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss)

        self.state['step'] += 1

        if loss is None:
            loss = closure()
        else:
            assert closure is None, 'if loss is provided then closure should be None'

        self.compute_grad_info()
        step = self.compute_step(float(loss))

        inertia_term = self.get_momentum() #add on the momentum term
        for step_i, m_i in zip(step, inertia_term):
          step_i.add_(m_i)

        update_params(self.params, step, stepsize=self.stepsize)

        self.update_momentum(step) #update the last two steps used to compute momentum

        if torch.isnan(self.params[0]).sum() > 0:
            raise ValueError('Got NaNs')

        return float(loss)

    def compute_grad_info(self):
        '''Compute current gradient and hessian gradient product'''
        grads = [p.grad for p in self.params]
        hessian_grad = torch.autograd.grad(grads, self.params, grad_outputs=grads)
        self.grads = grads
        self.hessian_grad = hessian_grad


    def compute_step(self,loss):
        pass


class SGD_test(SP2_base):

    '''Implements SGD by subclassing SP2_base for testing purposes'''

    def __init__(self, params, stepsize=1):
        super().__init__(params, stepsize=stepsize)

    def compute_step(self, loss):
        grads, hessian_grad = self.grads, self.hessian_grad
        step = [0.001*g for g in grads]
        return step

class SP2_plus(SP2_base):

    def __init__(self, params, stepsize=1, beta=0):
        super().__init__(params, stepsize=stepsize, beta=beta)

    def compute_step(self, loss):
        grads, hessian_grad = self.grads, self.hessian_grad



        return step



class SP2L1_plus(SP2_base):

    def __init__(self, params, lmda, init_s, stepsize=1, beta=0):
        super().__init__(params, stepsize=stepsize, beta=beta)
        self.lmda = lmda
        self.s = init_s
    
    def compute_step(self, loss):

        '''Computes the next step, also updates s'''

        grad, hessian_grad = self.grads, self.hessian_grad
        grad_norm_sq = norm(grad)**2
        lmda, s = self.lmda, self.s

        G3 = pos(loss - (s - (lmda/(2*(1-lmda))) ) ) / (1 + grad_norm_sq)

        G4 = min(G3, loss/grad_norm_sq)

        L1 = loss - G4*grad_norm_sq + 0.5*(G4**2)*inner_prod(hessian_grad,grad)

        Q = norm([g - G4*hg for g,hg in zip(grad,hessian_grad)])**2 #used again below
        denom_5 = 1 + Q
        G5 = pos(L1 - (s-(lmda/(2*(1-lmda))) ) ) / denom_5

        denom_6 = Q
        G6 = min(G5, L1/denom_6)

        w_step = [(G4 + G6)*g - G6*G4*hg for g,hg in zip(grad,hessian_grad)]

        new_s = pos(
                    pos(s - (lmda/(2*(1-lmda))) + G3 )
                        - (lmda/(2*(1-lmda))) + G5
                    )

        self.s = new_s

        return w_step

class SP2L2_plus(SP2_base):

    def __init__(self, params, lmda, init_s, stepsize=1, beta=0):
        super().__init__(params, stepsize=stepsize, beta=beta)
        self.lmda = lmda
        self.s = init_s
    
    def compute_step(self, loss):

        '''Computes the next step, also updates s'''

        grad, hessian_grad = self.grads, self.hessian_grad
        grad_norm_sq = norm(grad)**2
        lmda, s = self.lmda, self.s

        G1 = pos(loss - (1-lmda)*s ) / (1-lmda + grad_norm_sq)

        denom_2 = (1-lmda) \
                    + norm([g - G1*hg for g,hg in zip(grad,hessian_grad)])**2
        G2 = pos(
                (loss - G1*grad_norm_sq - ((1-lmda)**2)*(s+G1)
                + 0.5*(G1**2)*inner_prod(hessian_grad,grad))
                / denom_2
                )

        w_step = [G1*g + G2*(g - G1*hg) for g,hg in zip(grad,hessian_grad)]
        new_s = (1-lmda) * ((1-lmda)*(s+G1) + G2)

        self.s = new_s
        return w_step


class SP2max_plus(SP2_base):

    def __init__(self, params, lmda, stepsize=1, beta=0):
        super().__init__(params, stepsize=stepsize, beta=beta)
        self.lmda = lmda

    def compute_step(self, loss):
        grad, hessian_grad = self.grads, self.hessian_grad
        grad_norm_sq = norm(grad)**2
        lmda = self.lmda

        G1 = min(loss/grad_norm_sq,
                lmda/(2*(1-lmda))
                )

        G2 = loss - G1 * grad_norm_sq \
                + 0.5 * (G1**2) * inner_prod(hessian_grad, grad)

        denom_3 = norm([g - G1*hg for g,hg in zip(grad,hessian_grad)])**2
        G3 = min(G2/denom_3,
                lmda/(2*(1-lmda))
                )
        
        w_step = [(G1 + G3)*g - G3*G1*hg for g,hg in zip(grad,hessian_grad)]

        return w_step

pos = lambda x: x if x>0 else 0

def norm(v):
    '''Used to compute the norms of gradients'''
    return torch.sqrt(torch.sum(torch.stack([torch.norm(vi)**2 for vi in v])))

def inner_prod(v, w):
    '''used to compute inner products with the gradients'''
    return torch.sum(torch.stack([ torch.dot(torch.flatten(v_i), torch.flatten(w_i)) 
              for v_i,w_i in zip(v,w)]))

def update_params(params, step, stepsize=1):

    for p,g in zip(params, step):
        p.data.add_(other = -stepsize*g) #note the minus sign

