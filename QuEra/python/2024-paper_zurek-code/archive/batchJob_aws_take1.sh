#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -x ;  # print out the statements as they are being executed
set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value

#1stop_me

ser=a
#shots=5000 ; nAtom=13; backN=cpu; atm=c$nAtom   # CPU emulator
shots=1000 ; nAtom=58; backN=qpu ; atm=o$nAtom   # real HW

# assert nAtom in [13, 17, 33, 39, 47, 50, 58] # the only viable option

basePath=/dataVault/dataQuEra_2023paper_${backN}_t1
codePath=../problem_Z2Phase1D


#;  detune=-$omega; omega=0.0001

if [ ! -d "$basePath" ]; then
    echo create $basePath
    mkdir $basePath
    cd $basePath
    mkdir ana  jobs  meas 
    cd -
fi
    
k=0
detune_max=4.0

#for nAtom in 17 33 47  ; do
#    t_de=3.0; atm=c$nAtom 
#    if [ $nAtom -eq 58 ] ; then  atm=o$nAtom  ; fi

#for t_de in  0.3   0.5  1.0  2.1 3.0  ; do  # t_detune scan
#    detune_max=3.0 ; ser=e
#    detune_max=10.0 ; ser=d
#    #detune_max=18.0 ; ser=c

     
for t_de in  0.1 0.2 0.3 0.4 0.5 0.7 1.0 1.3 1.7 2.1 3.0  ; do  # t_detune scan
    
    t_us=`python3 -c "print('%.1f'%(1. + $t_de ))"`
    k=$[ $k +1 ]
    
    expN=zurek_${backN}_td${t_de}${atm}$ser    
    echo $k  expN=$expN
    #echo ${t_de},$expN
    #continue
    
    #time $codePath/submit_Z2Phase_job.py   --rabi_ramp_time_us 0.5   --rabi_omega_MHz 2.5  --detune_shape 0.5 --detune_delta_MH -2.6 $detune_max --atom_dist_um  6.2   --chain_shape  loop  $nAtom  --evol_time_us  $t_us  --numShots ${shots} --backendName ${backN}  --expName ${expN} --basePath ${basePath}  -E  #>& out/Ls.$expN

    $codePath/retrieve_awsQuEra_job.py  --expName ${expN} --basePath ${basePath}
    
    $codePath/postproc_Z2Phase.py  --expName ${expN} --basePath ${basePath}  -p r s e  d c m   -X   >& out/La.$expN
    
    #break  # do just the firts
done

# spare , to split data on halves :  --useHalfShots 0
echo done $k jobs
exit
