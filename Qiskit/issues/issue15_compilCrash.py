
#!/usr/bin/env python3
''' problem: crash for large qubit count

'''
import qiskit as qk
from qiskit.tools.monitor import job_monitor
from qiskit_ibm_provider import IBMProvider
from pprint import pprint

# for remap QQQ
from qiskit.transpiler import PassManager, Layout
from qiskit.transpiler.passes import SetLayout, ApplyLayout

#...!...!....................
def remap_qubits(qc,targetMap):
    # input: new target order of the current qubit IDs
    # access quantum register
    qr=qc.qregs[0]
    nq=len(qr)
    #print('rmr',qr,nq)
    assert len(targetMap)==nq
    #print('registers has %d qubist'%nq,qr)
    regMap={}
    for i,j in enumerate(targetMap):
        #print(i,'do',j)
        regMap[qr[j]]=i

    #print('remap qubits:'); print(regMap)
    layout = Layout(regMap)
    #print('layout:'); print(layout)

    # Create the PassManager that will allow remapping of certain qubits
    pass_manager = PassManager()

    # Create the passes that will remap your circuit to the layout specified above
    set_layout = SetLayout(layout)
    apply_layout = ApplyLayout()

    # Add passes to the PassManager. (order matters, set_layout should be appended first)
    pass_manager.append(set_layout)
    pass_manager.append(apply_layout)

    # Execute the passes on your circuit
    remapped_circ = pass_manager.run(qc)

    return remapped_circ



#...!...!....................
def circMcondM(cond):

        if cond:
            name='MCM_q0'
        else:
            name='MXM_q0'
        
        crPre = qk.ClassicalRegister(1, name="m_pre")
        crPost = qk.ClassicalRegister(1, name="m_post")
        qr = qk.QuantumRegister(1, name="q")
        qc= qk.QuantumCircuit(qr, crPre,crPost,name=name)
        qc.h(0)
        
        qc.measure(0,crPre)
        if cond:
            with qc.if_test((crPre, 0)):
                qc.x(0)                
        else:
            qc.x(0)
        qc.measure(0,crPost[0])        
        return qc


#=================================
#=================================
#  M A I N 
#=================================
#=================================

# -------Create a Quantum Circuit list for 1 qubit
qcL=[]
for c in [False,True]:
    qc=circMcondM(cond=c)
    print('BC',qc.name); print(qc)
    qcL.append(qc)

# ---- transpile circ for backen
backName='ibmq_jakarta'
backName='ibmq_qasm_simulator'
backName='ibm_washington'
#backName='ibm_sherbrooke'
#backName='ibm_auckland'
print('M:IBMProvider()...')
provider = IBMProvider()

backend = provider.get_backend(backName)
tot_qubits=backend.configuration().num_qubits
print('my backend=',backend,' num phys qubits:',tot_qubits)

qcTL = qk.transpile(qcL, backend, initial_layout=[0])

print('qcTL0',qcTL[0].draw(output="text", idle_wires=False))

shots=10000
#....  expand circ list to be executed over all qubits
print('M: clone circuits over all %d qubits'%tot_qubits)
qcTAL=qcTL.copy()
for qid in range(1,tot_qubits):
    #if len(qcTAL)>=150: break
    #if qid%2==1: continue
    if qid<50: continue
    qMap=[ i for i in range(tot_qubits) ]
    qMap[0]=qid; qMap[qid]=0  # swap  measured qubit
    for qc in qcTL:
        qc1=remap_qubits(qc,qMap)
        qc1.name=qc1.name.replace('_q0','_q%d'%qid)
        #print('rem',qid,qc1.name)
        #print(qc1.draw(output="text", idle_wires=False))
        qcTAL.append(qc1)
        
        
print('qcTAL[-1]',qcTAL[-1].draw(output="text", idle_wires=False))
print('num circ:',len(qcTAL),'shots=',shots)

job =  backend.run(qcTAL,shots=shots, dynamic=True)
jid=job.job_id()

print('submitted JID=',jid,backend ,'\n now wait for execution of your circuit ...')
 
job_monitor(job)
counts = job.result().get_counts(0)
pprint(counts)

try:
    job = backend.retrieve_job(jid)
except:
    print('job=%s  is NOT found, quit\n'%job)
    exit(99)
    
print('job IS found, retrieving it ...')

job_status = job.status()
print('Status  %s , queue  position: %s ' %(job_status.name,str(job.queue_position())))
print('M:ok')

