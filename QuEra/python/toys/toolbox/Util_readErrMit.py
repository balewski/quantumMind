__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
import numpy as np


#...!...!....................
def mitigate_probs_readErr_QuEra_1bit(Y,eps,forceProb=True):
    # vectorized, computed by Mathematica
    # forceProb assures:  p_i in [0,1] & sum p_i=1
    print('mit: Yinp\n',Y, '\nYsum=',np.sum(Y,axis=-1),Y.shape)
    assert Y.ndim>=3  # conform with shape[PAY,...,NB] 
    assert Y.shape[-1]==2 # hardcoded 1-bit mittigation
    assert Y.shape[0]==3 # PAY expected
        
    em1=1 - eps
    em2=em1**2
    eps2=eps**2
    fact=em1    
    #print('1bfact=',fact,'eps=',eps)

    X=np.zeros_like(Y) #,dtype=np.float32)

    #......... mitigate probabilities .........
    k=0  # -->P in PEY
    X[k,...,0]=em1*(Y[k,...,0]) - eps *(Y[k,...,1])
    X[k,...,1]=(Y[k,...,1])

    X[k]/=fact

    #........ compute variance ......
    k=1  # -->E in PEY
    X[k,...,0]=em2*np.power((Y[k,...,0]),2) + eps2*np.power((Y[k,...,1]),2)
    X[k,...,1]=np.power((Y[k,...,1]),2)
    
    X[k]=np.sqrt(X[k])/fact

    _common_mitigate_probs_completion(X,Y,forceProb)
            
    #print('mit: Xout\n',X, '\nXsum=',np.sum(X,axis=-1))
    
    return X



#...!...!....................
def mitigate_probs_readErr_QuEra_2bits(Y,eps,forceProb=True):
    # vectorized, computed by Mathematica
    # forceProb assures:  p_i in [0,1] & sum p_i=1
    #print('mit: Yinp\n',Y, '\nYsum=',np.sum(Y,axis=-1),Y.shape)
    assert Y.ndim>=3  # conform with shape[PAY,...,NB] 
    assert Y.shape[-1]==4 # hardcoded 2-bit mittigation
    assert Y.shape[0]==3 # PAY expected
        
    em1=1 - eps
    em2=em1**2
    em3=em1**3
    em4=em1**4
    em6=em1**6
    eps2=eps**2

    fact=em3    
    #print('2bfact=',fact,'eps=',eps)

    X=np.zeros_like(Y) #,dtype=np.float32)

    #......... mitigate probabilities .........
    k=0  # -->P in PEY
    X[k,...,0]=em3*(Y[k,...,0]) + eps *(-(em2*(Y[k,...,1])) - em2*(Y[k,...,2]) + 2*eps *(Y[k,...,3]))
    X[k,...,1]=em2*(Y[k,...,1]) - eps *(Y[k,...,3])
    X[k,...,2]=em2*(Y[k,...,2]) - eps *(Y[k,...,3])
    X[k,...,3]=em1*(Y[k,...,3])
                    
    X[k]/=fact

    #........ compute variance ......
    k=1  # -->E in PEY
    X[k,...,0]=em6*np.power((Y[k,...,0]),2) + eps2*(em4*np.power((Y[k,...,1]),2) + em4*np.power((Y[k,...,2]),2) + 4*eps2*np.power((Y[k,...,3]),2))
    X[k,...,1]=em4*np.power((Y[k,...,1]),2) + eps2*np.power((Y[k,...,3]),2)
    X[k,...,2]=em4*np.power((Y[k,...,2]),2) + eps2*np.power((Y[k,...,3]),2)
    X[k,...,3]=em2*np.power((Y[k,...,3]),2)
    
    X[k]=np.sqrt(X[k])/fact  # it is sqrt(var) so the same 'fact' is used

    _common_mitigate_probs_completion(X,Y,forceProb)

    #print('mit: Xout\n',X, '\nXsum=',np.sum(X,axis=-1))
    
    return X

#...!...!....................
def mitigate_probs_readErr_QuEra_3bits(Y,eps,forceProb=True):
    # vectorized, computed by Mathematica
    # forceProb assures:  p_i in [0,1] & sum p_i=1
    #print('mit: Yinp\n',Y, '\nYsum=',np.sum(Y,axis=-1),Y.shape)
    assert Y.ndim>=3  # conform with shape[PAY,...,NB] 
    assert Y.shape[-1]==8 # hardcoded 3-bit mittigation
    assert Y.shape[0]==3 # PAY expected
        
    em1=1 - eps
    em2=em1**2
    em3=em1**3
    em4=em1**4
    em5=em1**5
    em6=em1**6
    em12=em1**12
    em10=em1**10
    em8=em1**8

    eps2=eps**2
    eps4=eps**4

    fact=em6    
    #print('2bfact=',fact,'eps=',eps)

    X=np.zeros_like(Y) #,dtype=np.float32)

    #......... mitigate probabilities .........
    k=0  # -->P in PEY
    X[k,...,0]=em6*(Y[k,...,0]) - eps *(em5*(Y[k,...,1]) + em5*(Y[k,...,2]) - 2*em3*eps *(Y[k,...,3]) + em5*(Y[k,...,4]) - 2*em3*eps *(Y[k,...,5]) - 2*em3*eps *(Y[k,...,6]) + 6*np.power(eps ,2)*(Y[k,...,7]))
    X[k,...,1]=em5*(Y[k,...,1]) + eps *(-(em3*(Y[k,...,3])) - em3*(Y[k,...,5]) + 2*eps *(Y[k,...,7]))
    X[k,...,2]=em5*(Y[k,...,2]) + eps *(-(em3*(Y[k,...,3])) - em3*(Y[k,...,6]) + 2*eps *(Y[k,...,7]))
    X[k,...,3]=em4*(Y[k,...,3]) - em1*eps *(Y[k,...,7])
    X[k,...,4]=em5*(Y[k,...,4]) + eps *(-(em3*(Y[k,...,5])) - em3*(Y[k,...,6]) + 2*eps *(Y[k,...,7]))
    X[k,...,5]=em4*(Y[k,...,5]) - em1*eps *(Y[k,...,7])
    X[k,...,6]=em4*(Y[k,...,6]) - em1*eps *(Y[k,...,7])
    X[k,...,7]=em3*(Y[k,...,7])
                    
    X[k]/=fact

    #........ compute variance ......
    k=1  # -->E in PEY
    X[k,...,0]=em12*np.power((Y[k,...,0]),2) + eps2*(em10*np.power((Y[k,...,1]),2) + em10*np.power((Y[k,...,2]),2) + 4*em6*eps2*np.power((Y[k,...,3]),2) + em10*np.power((Y[k,...,4]),2) + 4*em6*eps2*np.power((Y[k,...,5]),2) + 4*em6*eps2*np.power((Y[k,...,6]),2) + 36*eps4*np.power((Y[k,...,7]),2))
    X[k,...,1]=em10*np.power((Y[k,...,1]),2) + eps2*(em6*np.power((Y[k,...,3]),2) + em6*np.power((Y[k,...,5]),2) + 4*eps2*np.power((Y[k,...,7]),2))
    X[k,...,2]=em10*np.power((Y[k,...,2]),2) + eps2*(em6*np.power((Y[k,...,3]),2) + em6*np.power((Y[k,...,6]),2) + 4*eps2*np.power((Y[k,...,7]),2))
    X[k,...,3]=em8*np.power((Y[k,...,3]),2) + em2*eps2*np.power((Y[k,...,7]),2)
    X[k,...,4]=em10*np.power((Y[k,...,4]),2) + eps2*(em6*np.power((Y[k,...,5]),2) + em6*np.power((Y[k,...,6]),2) + 4*eps2*np.power((Y[k,...,7]),2))
    X[k,...,5]=em8*np.power((Y[k,...,5]),2) + em2*eps2*np.power((Y[k,...,7]),2)
    X[k,...,6]=em8*np.power((Y[k,...,6]),2) + em2*eps2*np.power((Y[k,...,7]),2)
    X[k,...,7]=em6*np.power((Y[k,...,7]),2)
    X[k]=np.sqrt(X[k])/fact  # it is sqrt(var) so the same 'fact' is used

    _common_mitigate_probs_completion(X,Y,forceProb)

    #print('mit: Xout\n',X, '\nXsum=',np.sum(X,axis=-1))
    
    return X


#...!...!....................
def _common_mitigate_probs_completion(X,Y,forceProb):
    #...... (hack) re-normalize probabilities .....
    if forceProb:
        k=0
        X[k]=np.clip(X[0],0.,1.)  # force prob in [0,1]
        S=np.sum(X[k],axis=-1)  # sum of probs should be 1.0
        #print('prob S',S)
        #?assert S.ndim==1 # not expanded for mutiple hidden axis
        Sres=S[:,np.newaxis]
        Sbrd=np.broadcast_to(Sres,X[k].shape)
        #print('prob Sbrd',Sbrd)
        X[k]/=Sbrd
        S=np.sum(X[k],axis=-1)  # sum of probs should be 1.0
        #print('prob Scheck',S)

    #...... estimate mitigated counst ........
    k=2  # -->Y in PEY
    S=np.sum(Y[k],axis=-1)  # sum of yields=shots
    #print('yield S',S)
    #?assert S.ndim==1 # not expanded for mutiple hidden axis
    Sres=S[:,np.newaxis]
    Sbrd=np.broadcast_to(Sres,X[k].shape)
    X[k]=Sbrd*X[0]  # shots * new probabilities
                                      
