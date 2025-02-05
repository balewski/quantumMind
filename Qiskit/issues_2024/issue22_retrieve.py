from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()

# https://quantum.ibm.com/jobs/cntn93cb08x0008y0vhg
jid='cntn93cb08x0008y0vhg'  # hanoi

# https://quantum.ibm.com/jobs/clrilr9ptmuf3ee8jeng
jid='clrilr9ptmuf3ee8jeng' # sampler, ibmq_qasm_simulator , works

# https://quantum.ibm.com/jobs/cns2cfhqygeg00879yv0
jid='cns2cfhqygeg00879yv0' # smapler,  cairo, works

print('M: retrieve jid:',jid)
job=service.job(jid)
print('M: found job')
result=job.result()

jobMD=result.metadata    
print('M: job-meta[0]:',jobMD[0])
