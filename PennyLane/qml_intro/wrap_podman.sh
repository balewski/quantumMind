#!/bin/bash
echo W:myRank is $SLURM_PROCID
IMG=$1
CMD=$2
outPath=$3
DATA_VAULT=$4

if [ $SLURM_PROCID -eq 0 ] ; then 
   echo W:IMG=$IMG 
   echo W:CMD=$CMD
   #echo Q:fire $
fi

podman-hpc run -it \
	   --volume ${DATA_VAULT}//dataPennyLane_tmp:/dataPennyLane_tmp \
	   --volume $outPath:/wrk \
	   --workdir /wrk \
	   -e HDF5_USE_FILE_LOCKING='FALSE' \
	   $IMG $CMD --myRank $SLURM_PROCID  --expName abcRank$SLURM_PROCID
