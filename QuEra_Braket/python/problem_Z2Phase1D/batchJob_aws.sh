#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -x ;  # print out the statements as they are being executed
set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value

#1stop_me

nAtom=13   # 47 min on laptop
#dist=21.4; omega=1.8; detune=0.0
shots=10000 ;  backN=cpu   # CPU emulator
#shots=10 ; nAtom=20; backN=qpu   # real HW

basePath=/dataVault/dataQuEra_2023julyQMBS_$backN 

if [ ! -d "$basePath" ]; then
    echo create $basePath
    mkdir $basePath
    cd $basePath
    mkdir ana  jobs  meas calib
    cd -
fi

# .... configure initial state Hamiltonian
commonConf="   --chain_shape  chain $nAtom  --evol_time_us 2.0  --rabi_omega_MHz 2.5  --atom_dist_um  5.7  "
if [ $nAtom -eq 13 ]; then  #WRONG, for 11, it is 13-atoms config
    iniStateConf=" --rabi_ramp_time_us  0.29 0.31 --detune_shape   0.41 0.61 0.68 0.71 0.78 1.05  --add_ancilla  8 4.8 "
    
fi
fullConf=" $commonConf  $iniStateConf "
echo fullConf: $fullConf

# ......  run jobs .......
k=0
for i in {0..18}; do  # vary scare time
    t_us=`python3 -c "print('%.1f'%($i * 0.1))"`  
    k=$[ $k +1 ]
    expN=scar${nAtom}atom_${backN}_t$t_us
    
    echo $k  expN: $expN
    
    ./submit_Z2Phase_job.py  --scare_time_us 0.05  $t_us --numShots ${shots} --backendName ${backN}  --expName ${expN} --basePath ${basePath}  $fullConf >& out/Ls.$expN.txt
    
    #./retrieve_awsQuEra_job.py  --expName ${expN} --basePath ${basePath} 
    ./postproc_Z2Phase.py --expName ${expN} --basePath ${basePath}  -p r s e  d c m f -X >& out/La.$expN.txt
    #break
done



exit
