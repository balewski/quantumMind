#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -x ;  # print out the statements as they are being executed
set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value


dist=24
#shots=7 ; nAtom=9; backN=cpu   # CPU emulator
shots=5 ; nAtom=16; backN=qpu   # real HW

basePath=/dataVault/dataQuEra_2023may1AtB_$backN

if [ ! -d "$basePath" ]; then
    echo create $basePath
    mkdir $basePath
    cd $basePath
    mkdir ana  jobs  meas calib
    cd -
fi
    
k=0

for i in {0..29}; do
#for i in {0..15}; do    
    #t_us=`python3 -c "print('%.1f'%(0.1+ $i * 0.1))"`  # range 0.1 - 4.0 for 41 bins
    #t_us=`python3 -c "print('%.2f'%(0.3 + $i * 0.04))"`  # range 0.1 - 1.4 for 30 bins
    t_us=`python3 -c "print('%.2f'%(2.8 + $i * 0.04))"`  # range 2.8 - 4.0 for 30 bins
        
    k=$[ $k +1 ]
    expN=rabi_${backN}_t$t_us
    
    echo $k  expN=$expN
    #continue
    #time ./submit_awsQuEra_job.py  --numShots ${shots} --backendName ${backN}  --expName ${expN} --basePath ${basePath}  --num_atom $nAtom --atom_in_column  4 --atom_dist_um $dist   --rabi_omega_MHz 1.8 --evol_time_us $t_us -E  >& out/Ls.$expN
    
    ./retrieve_awsQuEra_job.py  --expName ${expN} --basePath ${basePath} 
    ./ana_AGS.py  --expName ${expN} --basePath ${basePath}  -X >& out/La.$expN
    #break
done



exit
