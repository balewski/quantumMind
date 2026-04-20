FROM ubuntu:24.04

#  podman build  --network=host -f ubu24-quera-tsim.dockerfile -t ubu24-quera-tsim:p1f   --platform linux/arm64
# --no-cache tells Podman not to use any cached layers
# on PM use 'podman-hpc' instead of 'podman' and all should work
# additionaly do 1 time: podman-hpc migrate balewski/ubuXX-qiskit-qml:p1


# Set non-interactive mode for apt-get
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

# Update the OS and install required packages
RUN echo "1a-AAAAAAAAAAAAAAAAAAAAAAAAAAAAA OS update" && \
    apt-get update && \
    apt-get install -y locales autoconf automake gcc g++ make vim wget ssh openssh-server sudo git emacs aptitude build-essential xterm python3-pip python3-tk python3-scipy python3-dev iputils-ping net-tools screen feh hdf5-tools python3-bitstring plocate graphviz tzdata x11-apps python3-venv dnsutils iputils-ping && \
    apt-get clean


# Create a virtual environment for Python packages to avoid the externally managed environment issue
RUN python3 -m venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Install additional Python libraries
RUN echo "2a-AAAAAAAAAAAAAAAAAAAAAAAAAAAAA python libs" && \
    pip install --upgrade pip && \
    pip install matplotlib h5py scipy jupyter notebook bitstring lmfit pytest scikit-learn pytz networkx[default] pandas

# Install Qiskit and its related packages
RUN echo "2b-AAAAAAAAAAAAAAAAAAAAAAAAAAAAA Qiskit libs" && \
    pip install --upgrade "qiskit[visualization,ibm]" qiskit-ibm-runtime qiskit-aer qiskit-experiments

# Tsim
RUN pip install -U bloqade-tsim pymatching sinter galois python-sat
RUN pip install --pre -U "stim>=1.16.dev0" "sinter>=1.16.dev0"

# Build tesseract-decoder from source (no aarch64 wheel on PyPI)
RUN echo "2c-AAAAAAAAAAAAAAAAAAAAAAAAAAAAA CMAKE tessarac" && \
    apt-get update && apt-get install -y cmake libboost-dev && apt-get clean && \
    pip install pybind11 && \
    git clone https://github.com/quantumlib/tesseract-decoder.git /tmp/tesseract && \
    cd /tmp/tesseract && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CXX_FLAGS="-fPIC -O2" -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())") && \
    make -j$(nproc) tesseract_decoder && \
    SITE=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    cp /tmp/tesseract/src/tesseract_decoder*.so $SITE/ && \
    cp -r /tmp/tesseract/src/py/_tesseract_py_util $SITE/ && \
    rm -rf /tmp/tesseract


# Final cleanup
RUN apt-get clean

# Set the default command to bash
CMD ["/bin/bash"]

# check the latest version:  pip index versions qiskit
