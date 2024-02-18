#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -x ;  # print out the statements as they are being executed
set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value

#1stop_me

dist=21.4; omega=1.8; detune=0.0
shots=10 ; nAtom=8; backN=cpu   # CPU emulator
shots=10 ; nAtom=20; backN=qpu   # real HW

#basePath=/dataVault/dataQuEra_2023may1B_$backN  # omega=1.8 MHz
basePath=/dataVault/dataQuEra_2023may1C_$backN ;  detune=-$omega; omega=0.0001

if [ ! -d "$basePath" ]; then
    echo create $basePath
    mkdir $basePath
    cd $basePath
    mkdir ana  jobs  meas calib
    cd -
fi
    
k=0

#for i in {0..29}; do  # focus rabi 1 atom
    #t_us=`python3 -c "print('%.2f'%(0.3 + $i * 0.04))"`  # range 0.1 - 1.4 for 30 bins
#    t_us=`python3 -c "print('%.2f'%(2.8 + $i * 0.04))"`  # range 2.8 - 4.0 for 30 bins

#for t_us in 0.30 0.34 0.38 0.58 0.62 0.66; do  # readErrMit  1-to-0
for t_us in 0.30 0.40 0.50 1.00 1.50 2.0 2.1 2.2 2.5 3.0 3.5 4.0 ; do  # readErrMit  only 0s

    k=$[ $k +1 ]
    expN=rabi_${backN}_t$t_us
    
    echo $k  expN=$expN
    #continue
    
    #time ./submit_awsQuEra_job.py  --numShots ${shots} --backendName ${backN}  --expName ${expN} --basePath ${basePath}  --num_atom $nAtom --atom_in_column 4  --gridType hex --atom_dist_um $dist   --rabi_omega_MHz $omega --detune_MHz $detune --evol_time_us $t_us -E  >& out/Ls.$expN
    
    ./retrieve_awsQuEra_job.py  --expName ${expN} --basePath ${basePath} 
    ./ana_AGS.py  --expName ${expN} --basePath ${basePath}  -X >& out/La.$expN
    #break
done



exit
