import numpy as np
from qiskit.pulse.instructions.phase import ShiftPhase
from qiskit.pulse.instructions.delay import Delay

#...!...!..................
def qiskitSched_unpack(sched):
    instrL=sched.instructions
    print('sched instruct len=',len(instrL))
    stepL=[]
    pulseD={}
    pulseidx=0
    for (t,op) in instrL:
        print('t=%d, op='%t,op)#chan.index) #,type(op)
        one=None
        if type(op)==ShiftPhase:
            one=(t,'Phase',float('%.5f'%op.phase),op.channel.name)
        elif  type(op)==Delay:
            one=(t,'Delay',op.duration,op.channel.name)
        else: # those are real pulses
                        
            #print('drive_ampl:',drv_ampls.shape, drv_ampls.dtype)#,'\n',drv_ampls[::10])
            name='%d_%s'%(pulseidx,op.name)
            pulseidx+=1
            one=(t,'Waveform',name,op.channel.name)
            try:
                samples=op.pulse.samples  # Waveform
            except:
                samples=op.pulse.get_waveform().samples # Drag
            pulseD[name]=samples
        stepL.append(one)
    return stepL,pulseD

#...!...!..................
def qiskitSched_dump(sched):
    instrL=sched.instructions
    print('sched instruct len=',len(instrL))
    for (t,op) in instrL:
        chan=op.channel
        print('t=%d, op='%t,op)#chan.index) #,type(op)
        continue
    if  type(op)==ShiftPhase:
        phase=op.phase
        print('just phase=%.3f'%phase)
    elif  type(op)==Delay:
        print('just delay=%d'%op.duration)
    else:
        drv_ampls=op.pulse.samples
        print('drive_ampl:',drv_ampls.shape, drv_ampls.dtype)#,'\n',drv_ampls[::10])


