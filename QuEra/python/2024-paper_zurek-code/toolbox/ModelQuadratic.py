__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import lmfit
import numpy as np


#...!...!....................
def quad_f(x, A,B,C):
    return A * x**2 + B * x + C

###########################################################
# The standard practice of defining an ``lmfit`` model

class ModelQuadratic(lmfit.model.Model):
    __doc__ = "parabola model" 

#...!...!..................
    def __init__(self, *args, **kwargs):
        super().__init__(quad_f, *args, **kwargs)

#...!...!..................        
    def guess(self, data, x=None, **kwargs):
        verbose = kwargs.pop('verbose', None)
        if x is None:   return
        
        if verbose :
            print('ModLin Guess: 0,1,0')
           
        params = self.make_params(A=0., B=1., C=0.)        
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)

