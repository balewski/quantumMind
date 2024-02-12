#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -x ;  # print out the statements as they are being executed
set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value

#1stop_me


shots=1000 ; nAtom=58; backN=qpu ; atm=o$nAtom   # real HW

# assert nAtom in [13, 17, 33, 39, 47, 50, 58] # the only viable option

basePath=/dataVault/dataQuEra_2023paper_confA
codePath=../problem_Z2Phase1D



k=0

     
#for t_de in  0.1 0.2 0.3 0.4 0.5 0.7 1.0 1.3 1.7 2.1 3.0  ; do  # t_detune scan
for t_de in  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.5 1.7 1.9 2.1 2.4 2.7 3.0  ; do  # t_detune scan

    
    k=$[ $k +1 ]
    
    expN=zurek_${backN}_At${t_de}${atm}
    echo
    echo $k  expN=$expN
    
    echo $expN
    expL=''
    for ser in a b c d ; do
	expL=$expL" ${expN}$ser"
    done

    expM=${expN}aM4
    echo expL=$expL   expM=$expM
    
    #continue
    
    $codePath/merge_measZ2Phase.py  --basePath ${basePath}   --expName $expL >& out/Lm.$expM
    #exit
    $codePath/postproc_Z2Phase.py --basePath ${basePath}  --expName ${expM}  -p r s e  d c m  f -X   >& out/La.$expM
    
    #break  # do just the first
done

# spare , to split data on halves :  --useHalfShots 0
echo done $k jobs
exit
