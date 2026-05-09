FROM ubuntu:24.04

#  podman build --network=host -f ubu24-arm-quera-squin.dockerfile -t ubu24-arm-quera-squin:p1a --platform linux/arm64 
# --no-cache tells Podman not to use any cached layers
# on PM use 'podman-hpc' instead of 'podman' and all should work

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

# QuEra SQUIN lives in bloqade-circuit. Install the circuit package directly
# instead of the umbrella "bloqade" metapackage, which currently pulls desktop
# visualization dependencies that complicate ARM builds.
RUN pip install -U bloqade-circuit

# Fail the build early if SQUIN is not importable on ARM.
RUN python -c "from bloqade import squin; print('verified', squin.__name__)"


# Final cleanup
RUN apt-get clean

# Set the default command to bash
CMD ["/bin/bash"]
