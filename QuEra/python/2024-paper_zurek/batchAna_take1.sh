#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -x ;  # print out the statements as they are being executed
set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value

#1stop_me

ser=a
shots=1000 ; nAtom=58; backN=qpu ; atm=o$nAtom   # real HW

# assert nAtom in [13, 17, 33, 39, 47, 50, 58] # the only viable option

basePath=/dataVault/dataQuEra_2023paper_take1
codePath=../problem_Z2Phase1D



k=0

     
for t_de in  0.1 0.2 0.3 0.4 0.5 0.7 1.0 1.3 1.7 2.1 3.0  ; do  # t_detune scan
    
    t_us=`python3 -c "print('%.1f'%(1. + $t_de ))"`
    k=$[ $k +1 ]
    
    expN=zurek_${backN}_td${t_de}${atm}$ser    
    echo
    echo $k  expN=$expN
    
    #echo $expN,${t_de}
    #continue
    
   
   ./apply_zneWall.py   --expName ${expN} --basePath ${basePath} -X   --noiseScale  1.1 1.2 1.3 1.4  |grep M:gold
    
    #$codePath/postproc_Z2Phase.py  --expName ${expN} --basePath ${basePath}  -p r s e  d c m   -X   >& out/La.$expN
    
    #break  # do just the firts
done

# spare , to split data on halves :  --useHalfShots 0
echo done $k jobs
exit
