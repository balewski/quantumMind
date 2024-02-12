__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


import numpy as np

#...!...!....................
def compute_wall_moments(probV,tshot):
    # INPUT: numpy array representing the probability distribution

    sum_p= np.sum( probV)
    # Generate the values of X corresponding to the distribution
    x_values = np.arange(len(probV))

    # Compute the mean (expected value) of X distribution
    mean_x = np.sum(x_values * probV)/sum_p
        
    # Compute the standard deviation of X
    std_x = np.sqrt(np.sum((x_values - mean_x)**2 * probV)/sum_p )

    # std of observables
    mean_err= std_x/np.sqrt(tshot)
    std_std =std_x/np.sqrt(2*(tshot-1))
    
    print("wall_num:  mean X=%.2f +/-%.2f,  std X=%.2f  var X=%.2f"%( mean_x,mean_err, std_x,std_x**2))
    print("AWDH,%d,%.3f,%.3f,%.3f "%(tshot,mean_x,mean_err, std_x))  
    return [ mean_x, mean_err, std_x, std_std, tshot]
    
#...!...!....................
def accumulate_wallDens_histo( mbitL,isLoop=False):
    nAtom=len(mbitL[0])
    tshot=len(mbitL)
    cntH=[0 for i in range(nAtom+1) ]
    for binpatt in mbitL:
        if isLoop:  binpatt=binpatt+binpatt[0]  # chain-->loop
        xV = np.array(list(binpatt))
        # Count all occurrences of '11' or '00' patterns
        n11 = np.sum(np.logical_and(xV[:-1] == '1', xV[1:] == '1'))
        n00 = np.sum(np.logical_and(xV[:-1] == '0', xV[1:] == '0'))
        nw=n00+n11        
        cntH[nw]+=1
    wallDensV=np.array(cntH)/tshot
    #print(' wallDensV dump:', wallDensV, 'tshot=',tshot)
    wallMoments=compute_wall_moments(wallDensV,tshot)
    return wallDensV,wallMoments

#...!...!....................
def XXzurek_observables_one(bigD,md):
    amd=md['analyzis']
    #pmd=md['payload']
    nAtom= amd['num_atom']
    hexpattV=bigD['ranked_hexpatt']
    mshotV=bigD['ranked_counts']
    nSol=hexpattV.shape[0]
    print('zurek_observables START t_detune=%.1f, numSol=%d'%(detT,nSol))    

    
    tshot=0
    w=4 # maximal wall width
    wwidthH=[0 for i in range(w)]  # accumulator

    for k in range(nSol):
        mshot=mshotV[k]
        hexpatt=hexpattV[k].decode("utf-8")
        A=BitArray(hex=hexpatt)[-nAtom:]  # clip leading 0s
        binpatt=A.bin
        
        
        #print('\nhex:',hexpatt,nAtom,binpatt, len(binpatt),'nw=',nw,mshot)
        count_wall_depth(binpatt,wwidthH,mshot)
        #break    
    wwidthV=np.array(wwidthH)/tshot
    print('wwidthV:',wwidthV)
    
    return  wallDensV,wallMoments,wwidthV
