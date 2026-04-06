#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value

#stop-me

basePath="/dataVault2026/qtuum_march"
#basePath="out"

shots=1000  
iniState=3  

nqL=" 6 10 14 18 22 24 " # 26 28
nqL="   30  32"
#nqL=" 24 26 28 "

#  ./submit_qtuum_zoo.py   --basePath  $basePath --taskType qft 17 --num_qubits 10  --num_shots 500 -b Helios-1E -E



echo bp=$basePath

# Define array of qubit configurations

for nq in $nqL ; do
    echo 'nq='$nq
    expName=qft_nq${nq}_heliosEmu
    echo expName=$expName shots=$shots
    
    #./submit_qtuum_zoo.py  --basePath  $basePath   --num_shots $shots --taskType qft $iniState  --num_qubits  $nq  -b Helios-1E   --expName $expName  -E ; continue
       
    ./retrieve_qtuum_job.py --basePath  $basePath  --expName $expName   # --cancelJob         
        
        echo
done

echo all DONE
 
