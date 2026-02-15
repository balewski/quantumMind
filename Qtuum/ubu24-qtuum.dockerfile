FROM ubuntu:24.04
# Quantinuum

# podman build   -f ubu24-qtuum.dockerfile -t balewski/ubu24-qtuum:p3e   --platform linux/arm64   
# PM: real      7m42.406s
# for omp_get_num_threads:  #      -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 \


# Set non-interactive mode for apt-get
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

# Update the OS and install required packages
RUN echo "1a-AAAAAAAAAAAAAAAAAAAAAAAAAAAAA OS update" && \
    apt-get update && \
    apt-get install -y locales autoconf automake gcc g++ make vim wget ssh openssh-server sudo git emacs aptitude build-essential xterm python3-pip python3-tk python3-scipy python3-dev iputils-ping net-tools screen feh hdf5-tools python3-bitstring plocate graphviz tzdata x11-apps python3-venv dnsutils iputils-ping libgomp1 && \
    apt-get clean

# Create a virtual environment for Python packages to avoid the externally managed environment issue
RUN python3 -m venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"


# Install ML  libraries
RUN echo "2c-AAAAAAAAAAAAAAAAAAAAAAAAAAAAA math libs" && \
    /opt/venv/bin/pip install scikit-learn pandas seaborn[stats] networkx[default]

# Install additional Python libraries
RUN echo "2d-AAAAAAAAAAAAAAAAAAAAAAAAAAAAA python libs" && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install matplotlib h5py scipy jupyter notebook bitstring lmfit pytest pylatexenc

# Quantinuum libs
# Upgrade pip and install necessary Python packages
RUN pip install --upgrade pip wheel setuptools

# Explicitly install wasmtime and guppylang-internals to workaround dependency spec issue
# guppylang-internals 0.27 has a bad dependency on wasmtime~=v38.0.0 (v prefix issue)
# We pre-install wasmtime and force-install guppylang-internals without verifying deps
RUN pip install wasmtime==38.0.0
RUN pip install guppylang-internals==0.27 --no-deps

# Patch the installed metadata to remove the 'v' prefix from the requirement
RUN find /opt/venv -name "METADATA" -exec sed -i 's/wasmtime~=v38.0.0/wasmtime~=38.0.0/g' {} +

# Install pytket and the Quantinuum extension
RUN pip install pytket pytket-quantinuum qnexus pytket-qiskit qiskit-aer guppylang==0.21.8 selene_sim

# Install pyqir
RUN pip install pyqir

# hugr-qir has no pre-built arm64 wheel, must compile from source
# Requires LLVM 14 dev headers and Rust toolchain
RUN apt-get update && apt-get install -y llvm-14-dev libclang-14-dev libpolly-14-dev curl && apt-get clean
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"
RUN pip install hugr-qir

#RUN apt-get install -y     x11-apps 

# Final cleanup
RUN apt-get clean

# Set the default command to bash
CMD ["/bin/bash"]