__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import lmfit
import numpy as np

#...!...!....................
def power_f(x,A,MU): # constant bias
    return    A*(x**MU)

###########################################################
# The standard practice of defining an ``lmfit`` model

class ModelPower(lmfit.model.Model):
    __doc__ = "power function"

#...!...!..................
    def __init__(self, *args, **kwargs):
        super().__init__(power_f, *args, **kwargs)
        #self.set_param_hint('A', min=0.01)  # Enforce A is positive
       
#...!...!..................        
    def guess(self, data, x=None, **kwargs):
        verb = kwargs.pop('verbose', None)
        if x is None:   return

        #[badSeed, period0, phase0, ampl0, offset0 ]= seed_sin_fit(x,data)
                
        params = self.make_params(A=6, MU=0.3)
        #params['P'].set(min=-2.1*np.pi, max=2.1*np.pi)    
        #params['A'].vary = False
        
        
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)

