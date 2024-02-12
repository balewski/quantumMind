__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import lmfit
import numpy as np

from toolbox.Util_Fitter import seed_sin_fit

############################################################
''' Since ``scipy.optimize`` and ``lmfit`` require real parameters
  we use the following empiric formula to fit the resonance

'''

#...!...!....................
# input, amplitude, decay parameter, angular freq, constant offset
def sin_f(x,A,F,P,E): # constant bias
    return   (1-E)*( 0.5+A*np.sin(2*np.pi*F*x+P) )


###########################################################
# The standard practice of defining an ``lmfit`` model

class ModelSinMit(lmfit.model.Model):
    __doc__ = "damped oscilator modelXXX"# + lmfit.models

#...!...!..................
    def __init__(self, *args, **kwargs):
        super().__init__(sin_f, *args, **kwargs)
        self.set_param_hint('E', min=0.001)  # Enforce E is positive
       

#...!...!..................        
    def guess(self, data, x=None, **kwargs):

        verb = kwargs.pop('verbose', None)
        if x is None:   return

        [badSeed, period0, phase0, ampl0, offset0 ]= seed_sin_fit(x,data)
        if verb: print('badSinSeed=%r'%badSeed)
        if 'period' in kwargs: period0=kwargs['period']
        if 'ampl' in kwargs: ampl0=kwargs['ampl']
                
        freq0=1./period0
        
        params = self.make_params(A=ampl0, F=freq0, P=phase0, E=0.,T=100.) #?C=offset0)
        #?params = self.make_params(A=-0.5, F=1.8, P=0., E=0.) #?C=offset0)
        params['P'].set(min=-1.1*np.pi, max=1.1*np.pi)    

        # fix  some params to test sensitivity
        #params['C1'].vary = False
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)

