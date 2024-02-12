from pyquil import Program, get_qc
qprg=Program("DECLARE ro BIT[2]\nH 0\nY 1\nXY(pi) 1 2\nFENCE 0\nRY(0.34) 3\nMEASURE 1 ro[1]")
qprg.wrap_in_numshots_loop(5)
print('M:qprg\n',qprg)
device_name = '6q-qvm'
qcom = get_qc(device_name, as_qvm=True)  #backend
shot_meas = qcom.run(qprg).readout_data.get("ro")
print('M:shot_meas',shot_meas)
