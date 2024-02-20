#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -x ;  # print out the statements as they are being executed
set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value

stop-me
conf=C ; atomDist=6.2
#conf=C ; atomDist=5.7

batch=d

#shots=5000 ; nAtom=13; backN=cpu; atm=c$nAtom   # CPU emulator
shots=500 ; nAtom=58; backN=qpu ; atm=o$nAtom   # real HW

# assert nAtom in [13, 17, 33, 39, 47, 50, 58] # the only viable option

basePath=/dataVault/dataQuEra_2023paper_conf$conf
codePath=../problem_Z2Phase1D

if [ "$backN" == "cpu" ] ; then
   basePath='env'
fi


if [ ! -d "$basePath" ]; then
    echo create $basePath
    mkdir $basePath
    cd $basePath
    mkdir post  jobs  meas 
    cd -
fi
    
k=0
   
for t_detune in  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.5 1.7 1.9 2.1 2.4 2.7 3.0  ; do
#for t_detune in  0.1  0.8  3.0  ; do 
    
    t_evol=`python3 -c "print('%.1f'%(1. + $t_detune ))"`
    t_hold=`python3 -c "print('%.1f'%(4. - $t_evol ))"`
    k=$[ $k +1 ]
    
    expN=zurek_${backN}_${conf}t${t_detune}${atm}$batch   
    #echo $k  expN=$expN  #$t_evol $t_hold
    
    #continue
    
    #echo $codePath/submit_Z2Phase_job.py   --rabi_ramp_time_us 0.5   --rabi_omega_MHz 2.5  --detune_shape 0.5 --detune_delta_MH -2.6 4.0  --atom_dist_um  $atomDist   --chain_shape  loop  $nAtom  --evol_time_us  $t_evol  ${shots} --backendName ${backN}  --expName ${expN} --basePath ${basePath} -E  #>& out/Ls.$expN

    # SPARE:  --hold_time_us  $t_hold  --numShots 
    
    $codePath/retrieve_awsQuEra_job.py  --expName ${expN} --basePath ${basePath}
    
    $codePath/postproc_Z2Phase.py  --expName ${expN} --basePath ${basePath}  -p r s e  d c m   -X   >& out/La.$expN
    
    #break  # do just the first
done

# spare , to split data on halves :  --useHalfShots 0
#echo done $k jobs
exit
