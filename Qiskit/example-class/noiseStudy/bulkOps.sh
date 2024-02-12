#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#./retr_ciruitArray.py  --transpName grover_3Qas01.q20b-Q3+2+1  --measCalib sim -d simjobs -j $1


for i in      j159c82 jb08f95  j159c84  j91894c j159c86 j159c88  j1ccbd8 jd0dba6 jd0dba8 j159c8a j032608  ; do
    echo bulk=$i
    ./retr_ciruitArray.py  --transpName   grover_3Qas01.q20j-Q5+6+0 --measCalib  j1ccbc8 -j $i

done

exit

for i in `seq 1 18`; do
    echo bulk=$i
    ./submit_ciruitArray.py --shots 1024 --transpName grover_3Qas01.q20j-Q5+6+0  --noiseModel 25 1 1.0
done

exit




