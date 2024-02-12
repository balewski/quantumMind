__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import lmfit
import numpy as np


#...!...!....................
def linear_f(x,A,B): # constant bias
    return  A*x +B 

###########################################################
# The standard practice of defining an ``lmfit`` model

class ModelLinear(lmfit.model.Model):
    __doc__ = "linear model" 

#...!...!..................
    def __init__(self, *args, **kwargs):
        super().__init__(linear_f, *args, **kwargs)

#...!...!..................        
    def guess(self, data, x=None, **kwargs):
        verbose = kwargs.pop('verbose', None)
        if x is None:   return
        
        if verbose :
            print('ModLin Guess: 0,0')
           
        params = self.make_params(A=0., B=0.)        
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)

