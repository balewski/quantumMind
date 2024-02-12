#!/usr/bin/env python3
# generate Pulses for a fake 2Q device
# My ticket: https://quantumcomputing.stackexchange.com/questions/17410/configuration-of-fakeopenpulse2q

import qiskit
print(qiskit.__qiskit_version__)

import matplotlib.pyplot as plt
from qiskit import IBMQ
from qiskit import QuantumCircuit, transpile,schedule
from qiskit.test.mock import FakeOpenPulse2Q
# to build newU3 pule definition
from qiskit.pulse import  play, build, shift_phase, ControlChannel, DriveChannel, GaussianSquare, Drag, Gaussian

# - - - - - - - - - - - -
def transfer_U3_calib(back1, back2, trgtQ):
    instrL=back1.defaults().instruction_schedule_map.get('u3',qubits=[0]).instructions
    print('back1 instr:')#,instr)
    for (t,op) in instrL:
        print('t=%d, op='%t,op)#chan.index) #,type(op)

    #Create your calibration pulse program using the pulse builder

    P2=0.90
    P0=0.40
    P1=0.60
    with build(back2) as pulse_prog:
        shift_phase(-1.0*P2, DriveChannel(0)),
        play(Drag(duration=320, amp=(0.370467155866812-0.0762912966131609j), sigma=80, beta=-1.095928909587139, name='X90p_d0'), DriveChannel(0), name='X90p_d0'),
        shift_phase(-1.0*P0, DriveChannel(0)),
        play(Drag(duration=320, amp=(-0.370467155866812+0.07629129661316089j), sigma=80, beta=-1.095928909587139, name='X90m_d0'), DriveChannel(0), name='X90m_d0')
        shift_phase(-1.0*P1, DriveChannel(0))

        pulse_prog.draw('IQXDebugging',  show_waveform_info = True)
    ok1

#=================================
#=================================
#  M A I N 
#=================================
#=================================

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend1 = provider.get_backend('ibmq_armonk')
backend2 = FakeOpenPulse2Q()

transfer_U3_calib(backend1, backend2, 0)


circ1 = QuantumCircuit(1,name='1Qcirc')
circ1.u(0.4,0.6,0.9,0)
qc1 = transpile(circ1, basis_gates=['u1','u2','u3'], optimization_level=1)

circ2 = QuantumCircuit(2,name='2Qcirc')
circ2.u(0.4,0.6,0.9,0)
circ2.sx(1)
qc2 = transpile(circ2, basis_gates=['u1','u2','u3'], optimization_level=1)

sched1=schedule(qc1,backend1)
sched2=schedule(qc2,backend2)

fig=plt.figure(1,facecolor='white', figsize=(14, 8))
ax1=plt.subplot2grid((3,1), (0,0), rowspan=1)#,sharex=ax1 )
ax2=plt.subplot2grid((3,1), (1,0), rowspan=2)#,sharex=ax1 )

sched1.draw('IQXDebugging', axis = ax1, show_waveform_info = True)#,plot_range=[0, 60])
sched2.draw('IQXDebugging', axis = ax2, show_waveform_info = True)#,plot_range=[0, 60])


print('circuit1:'); print(circ1); print(qc1)
print('\ncircuit2:'); print(circ2); print(qc2)
plt.tight_layout()
plt.show()
