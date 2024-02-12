#!/bin/bash

PODMAN_IMG=balewski/deb10-rigetti:p2b
#PODMAN_IMG=balewski/deb10-rigetti:x1
#PODMAN_IMG=rigetti/forest:3.3.2

# after mac reboot activate XQuartz
xhost + 127.0.0.1 

echo This instance allows to work on Rigetti, Jupyter-NB
echo you are launching Docker image ...  remeber to exit

JNB_PORT=' '
WORK_DIR=/quantumMind/Rigetti
echo "The number of arguments is: $#"
#  encoded variables:   jnb
for var in "$@"; do
  echo "The length of argument '$var' is: ${#var}"
  if [[ "jnb" ==  $var ]];  then
     JNB_PORT="    --publish 8844:8844 "
     echo added  $JNB_PORT
     echo sudo balewski
     echo cd /quantumMind/Rigetti/notebooks/
     echo jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --port  8844
  fi
done


#
#     --publish 8844:8844  \

echo "wait for qvm+quilc daemons to start ..."
podman run -it  \
    -e QCrank_dataVault=/dataVault/dataQCrank_august16 \
    -e QBArt_dataVault=/dataVault/dataQBArt_sep7 \
    --workdir $WORK_DIR \
    --volume /Users/balewski/docker_volumes/dataVault:/dataVault \
    --volume /Users/balewski/docker_volumes/qcrank_wrk:/qcrank_wrk \
    --volume /Users/balewski/docker_volumes/quantumMind:/quantumMind \
    $JNB_PORT \
    --user $(id -u):$(id -g) \
    $PODMAN_IMG /bin/bash

# 
#
#      -e DISPLAY=host.docker.internal:0 \

#/src/pyquil/entrypoint.sh
# test JNB
# cd    /qtuum_wrk/hqs-api-examples/notebooks
# jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --port 8844
# apt install   texlive texlive-latex-extra texlive-fonts-recommended dvipng
