#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable

# ./submit_circ.py -t ghz0s_3qm.q20b-Q16+15+17

for circ in 0 0r 0s 0t 0u ; do
    for q in 15+17 17+15 ; do
	name=ghz${circ}_3qm.q20b-Q16+$q
	log=out/log.$name
	echo $log

	./submit_circ.py -t $name -r 50 --noErrCorr >& $log
	grep Submitted $log
	#exit
    done
done

exit

for i in `seq 1 18`; do
    echo bulk=$i
    ./submit_ciruitArray.py --shots 1024 --transpName grover_3Qas01.q20j-Q5+6+0  --noiseModel 25 1 1.0
done

exit




