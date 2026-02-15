FROM ubuntu:24.04
# Quantinuum - x86_64 version

# podman-hpc build -f ubu24-x86-qtuum.dockerfile -t balewski/ubu24-x86-qtuum:p4a .
# >> real	7m51.364s
# podman-hpc migrate balewski/ubu24-x86-qtuum:p4a

# Set non-interactive mode for apt-get
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

# Update the OS and install required packages
RUN echo "1a-OS update" && \
    apt-get update && \
    apt-get install -y locales autoconf automake gcc g++ make vim wget ssh openssh-server sudo git emacs aptitude build-essential xterm python3-pip python3-tk python3-scipy python3-dev iputils-ping net-tools screen feh hdf5-tools python3-bitstring plocate graphviz tzdata x11-apps python3-venv dnsutils libgomp1 cmake ninja-build && \
    apt-get clean

# Create a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install math libs
RUN pip install --upgrade pip wheel setuptools && \
    pip install scikit-learn pandas seaborn[stats] networkx[default] matplotlib h5py scipy jupyter notebook bitstring lmfit pytest pylatexenc

# Quantinuum libs
# Explicitly install wasmtime and guppylang-internals to workaround dependency spec issue
RUN pip install wasmtime==38.0.0
RUN pip install guppylang-internals==0.27 --no-deps

# Patch the installed metadata to remove the 'v' prefix from the requirement
RUN find /opt/venv -name "METADATA" -exec sed -i 's/wasmtime~=v38.0.0/wasmtime~=38.0.0/g' {} +

# Install Quantinuum & Tket suite
# On x86_64, these have pre-built wheels, so no conan/rust compilation is needed
RUN pip install pytket pytket-quantinuum qnexus pytket-qiskit qiskit-aer guppylang==0.21.8 selene_sim tket2 tket hugr-qir

# Final cleanup
RUN apt-get clean

# Set the default command to bash
CMD ["/bin/bash"]
