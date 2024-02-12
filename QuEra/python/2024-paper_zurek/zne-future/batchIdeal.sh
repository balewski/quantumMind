#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -x ;  # print out the statements as they are being executed
set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value

shots=10000 
noiseStr=" 1.2 1.4 1.6 1.8 2.0 "
resample=40

# assert nAtom in [13, 17, 33, 39, 47, 50, 58] # the only viable option

#basePath=/dataVault/dataQuEra_2023paper_take1
basePath='env'
codePath=../problem_Z2Phase1D


k=0

     
for pfbit in 0.01 0.02 0.05 0.10 0.15 0.20  0.25 ; do
#for pfbit in ; do           
    expN=ideal10k_o58pf${pfbit}  
    echo
    echo $k  expN=$expN
    
    #echo $expN,${t_de}
    #continue
    
    ./ideal_samples_zurek.py  --numShots $shots --bitInfidelity $pfbit  --expName   $expN
    ./apply_zneWall.py      --expName  $expN   -X  --noiseScale  $noiseStr --sampleReuse $resample
    cp /dataVault/dataQuEra_2023julyA/rezne/${expN}.zneWall.h5 4milo
    #break  # do just the first
done




echo done $k jobs
exit
