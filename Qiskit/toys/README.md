# Qiskit Toys Directory

This directory contains various quantum computing examples, experiments, and utilities developed using Qiskit. The programs demonstrate different aspects of quantum programming including circuit optimization, noise modeling, tomography, and hardware execution.

## 📁 Directory Structure

### 🚀 **Circuit Construction & Analysis**
- **`bigEndianUnitary.py`** - Convert unitary matrices between big-endian and little-endian conventions with matrix display utilities
- **`check_CxRy_to_CzRyH.py`** - Verify equivalence of quantum circuit implementations: CX-RY-CX vs H-CZ-RY-CZ-H sequences  
- **`circ_2_big_gate.py`** - Create custom quantum gates (T-gates) and compose them into larger circuit structures
- **`compare_trasp_unitaries.py`** - Compare unitary matrices of original and transpiled quantum circuits across different backends
- **`examin_qpy_circ.py`** - Load and analyze quantum circuits from QPY files, evaluate transpilation for hardware backends
- **`write_read_parametric_qpy.py`** - Comprehensive workflow for creating, serializing, and deserializing parametric quantum circuits with metadata

### 🔬 **State Vector & Density Matrix Simulations**
- **`stateVect_ideal.py`** - Basic state vector simulation with custom unitary operations and state initialization
- **`stateVect_a.py`** - Advanced state vector simulation with iSWAP operators and custom circuit construction  
- **`stateVect_noisy.py`** - State vector simulation with realistic noise models from IBM quantum backends
- **`densMatrx_a.py`** - Density matrix simulation with random density matrices and Hadamard gate evolution
- **`densMatrx_noisy.py`** - Density matrix simulation with noise models extracted from real IBM quantum devices

### 🔊 **Noise Modeling & Error Mitigation**
- **`noisy1qCustomIdle.py`** - Custom 1-qubit noise model with depolarizing and Pauli errors for identity gates
- **`noisy2qCustomGate.py`** - Custom 2-qubit noise model with amplitude damping errors for iSWAP gates
- **`noisyBell_fakeTorino.py`** - Bell state preparation using FakeTorino backend with hardware noise characteristics
- **`noisyBell_myRho.py`** - Bell state generation with custom density matrix and noise modeling (needs cleanup)
- **`tensorReadErrCorr.py`** - Readout error correction using tensored calibration matrices for multi-qubit systems

### 📊 **Quantum Hardware Execution & Monitoring**
- **`toy_HW_sampler.py`** - Hardware execution on IBM quantum devices with parametric circuits and reset operations
- **`toy_aerSampler.py`** - AER simulator execution with feed-forward operations and hardware noise models
- **`toy_aerEstimator.py`** - Expectation value estimation using AER backend with Pauli observables
- **`toy_rndCircSampler.py`** - Random circuit generation and execution with density matrix simulation
- **`submitFractGateHW.py`** - Submit circuits with fractional gates (RZZ) to IBM quantum hardware
- **`evalFractGateHW.py`** - Load and analyze QPY circuits with fractional gate optimization for hardware execution
- **`run_parallel_GHZ.py`** - Parallel GHZ circuit execution across multiple qubit groups on quantum hardware
- **`ibmq_cost_by_day.py`** - Track and analyze IBM Quantum usage costs and quantum time consumption by backend
- **`retrieve_job.py`** - Retrieve and analyze completed quantum jobs with detailed execution metrics

### 🎯 **Quantum Tomography & Teleportation** 
- **`teleport_tomo.py`** - **Enhanced teleportation** with tomography for X, Y, Z bases, feed-forward corrections, and delay lines
- **`teleport_mZ.py`** - Simple teleportation protocol with Z-basis measurement and feed-forward
- **`teleport_mZ_noFF.py`** - Teleportation without feed-forward, using post-processing corrections
- **`teleport_tomo_noFF.py`** - Teleportation tomography template with classical post-processing for all measurement bases

### 🛠 **Utilities & Analysis Tools**
- **`feedForward_resilient.py`** - Feed-forward operations with majority voting for readout error compensation
- **`issue_calib.py`** - Backend calibration data extraction and gate fidelity analysis
- **`select_Tqubits_inQPU.py`** - Systematic selection of optimal qubit arrangements for T-shaped connectivity patterns
- **`toy_ermal.py`** - Error mitigation examples with state vector and density matrix methods (partial implementation)

### 📓 **Jupyter Notebooks**
- **`13_trotter_qrte.ipynb`** / **`13_trotterQRTE.ipynb`** - Trotter decomposition for quantum real-time evolution of Hamiltonian systems
- **`feedForw_A.ipynb`** - Interactive feed-forward quantum computing demonstrations
- **`run_HW_fakeSim.ipynb`** - Hardware simulation and noise model experiments
- **`rndClifford_to_qiskit.ipynb`** - Random Clifford group operations and circuit translations
- **`toy_trotter_jumpOper.ipynb`** - Trotter evolution with jump operators for open quantum systems
- **`trotter_Ising_line_statevect.ipynb`** - Ising model simulation on linear chains using state vector evolution
- **`trotter_Ising_shots_timeDependent.ipynb`** - Time-dependent Ising model with shot-based measurements

### 📦 **Data Files & Archives**
- **`inc_advection_solver_set400.qpy`** - Serialized quantum circuit for PDE solving applications
- **`ghz_circuit.png`** - GHZ circuit visualization
- **`ghz_counts_histogram.pdf`** - Measurement histogram analysis

## 🎯 **Key Features Demonstrated**

### **Quantum Algorithms**
- **Teleportation Protocol**: Complete implementation with tomography across X/Y/Z bases
- **GHZ State Preparation**: Multi-qubit entanglement generation and verification  
- **Trotter Evolution**: Time evolution simulation for quantum many-body systems

### **Hardware Integration** 
- **IBM Quantum Execution**: Real hardware job submission and result analysis
- **Noise Characterization**: Realistic error models from actual quantum devices
- **Resource Optimization**: Qubit selection and circuit placement strategies

### **Error Analysis & Mitigation**
- **Readout Error Correction**: Tensored calibration matrix approach
- **Feed-forward Operations**: Real-time conditional quantum operations
- **Statistical Analysis**: nSigma threshold evaluation for measurement validation

### **Advanced Qiskit Features**
- **QPY Serialization**: Circuit persistence with metadata and parametric gates
- **Custom Noise Models**: User-defined error channels and decoherence
- **Transpilation Analysis**: Circuit optimization across different backend targets

## 🚀 **Getting Started**

Most programs can be run independently with:
```bash
python3 <program_name>.py --help  # For command-line options
```

### **Recommended Starting Points:**
1. **`stateVect_ideal.py`** - Basic quantum simulation
2. **`teleport_tomo.py`** - Advanced quantum protocol with comprehensive features  
3. **`toy_aerSampler.py`** - Intermediate simulation with realistic noise
4. **`run_parallel_GHZ.py`** - Hardware execution example

## 📋 **Dependencies**
- **Qiskit** >= 1.2
- **qiskit-aer** >= 0.15
- **qiskit-ibm-runtime** (for hardware access)
- **numpy**, **matplotlib**, **networkx** (for analysis and visualization)

## 👨‍💻 **Author**
Jan Balewski - janstar1122@gmail.com

---
*This collection represents ongoing research and development in quantum computing, with programs ranging from basic educational examples to advanced research implementations.*
