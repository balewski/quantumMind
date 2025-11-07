#!/usr/bin/env python3
"""
Generates and displays stim circuits for repetition codes.
"""

import stim
import pymatching
import numpy as np
import argparse
import os
from toolbox.PlotterBackbone import PlotterBackbone


def get_parser():
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verb", type=int, choices=[0, 1, 2],
                        default=1, help="increase output verbosity")
    parser.add_argument("--circType", type=str, default="repetition_code:memory",
                        help="type of stim circuit to generate")
    parser.add_argument("--distance", type=int, default=9, help="distance of the code")
    parser.add_argument("--rounds", type=int, default=None, help="number of rounds")
    parser.add_argument("--before_round_data_depolarization", type=float, default=0.04,
                        help="depolarization before round data")
    parser.add_argument("--before_measure_flip_probability", type=float, default=0.001,
                        help="measurement flip probability")
    parser.add_argument('-d',"--dumpCirc", action='store_true', help="dump full circuit")
    parser.add_argument("-p","--showPlots", nargs='+', default=['a'],
                        help="list of plots to show, e.g., 'a b c'")
    parser.add_argument("--sampMeas", action='store_true', help="sample measurements and display")
    parser.add_argument("--sampDet", action='store_true', help="sample detectors and display")
    
    # Args for PlotterBackbone
    parser.add_argument("--outPath", default='out', help="output path for plots and saved data")
    parser.add_argument('-Y',"--noXterm", action='store_false', help="enable0 X-term for plotting")
   
    args = parser.parse_args()
    args.showPlots = "".join(args.showPlots)
    args.prjName='demoCodeDist%d'%args.distance
    if args.rounds==None : args.rounds=3*args.distance
    for arg in vars(args):
        print(f"myArg: {arg} = {getattr(args, arg)}")
        
    return args


# ==================================
#         PLOTTER
# ==================================
class MyPlotter(PlotterBackbone):
    def __init__(self, args):
        super().__init__(args)

    def circuit_diagram(self, circuit, figId=1):
        """Saves the circuit diagram as an SVG file."""
        diagram_obj = circuit.diagram('timeline-svg')
        svg_data = str(diagram_obj)
        
        # Use figId2name to be consistent with PlotterBackbone
        base_name = self.figId2name(figId)
        fname = os.path.join(self.outPath, base_name + '.svg')
        
        with open(fname, 'w') as fd:
            fd.write(svg_data)
        print(f"Saved circuit diagram to {fname}")

#...!...!..................
    def vary_xError(self,X,Y,roundL,tit=None,figId=2):
        figId=self.smart_append(figId)
        nrow,ncol=1,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(7,5))
        ax = self.plt.subplot(nrow,ncol,1)
        print('ver  shapes:',X.shape,Y.shape)
        for ir,k  in enumerate(roundL):
            print(ir,k)
            ax.plot(X, Y[ir], label=f"QEC rounds={k}")
        diagL=[0.05,0.5]  # for repet-code
        #diagL=[0.0005,0.05]  # for color-code
        ax.plot(diagL,diagL,'k--',label='y=x')
        ax.set( xscale='log', yscale='log', xlabel="physical CX error", ylabel="logical error rate per shot",title=tit)
        ax.legend(); ax.grid()
        
# ==================================
#       SAMPLING FUNCTIONS
# ==================================
def count_logical_errors(circuit: stim.Circuit, num_shots: int) -> int:
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    # Run the decoder.
    predictions = matcher.decode_batch(detection_events)

    # Count the mistakes.
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    return num_errors

def scan_X_err( xErrL,roundL,args,shots):
    Y=[]
    print('scan X_err:',xErrL)
    for k in roundL:
        ys = []
        for x in xErrL:
            circuit = stim.Circuit.generated(
                args.circType,
                distance=args.distance,
                rounds=k,
                before_round_data_depolarization=x,
                before_measure_flip_probability=args.before_measure_flip_probability
            )
            num_errors_sampled = count_logical_errors(circuit, shots)
            ys.append(num_errors_sampled / shots)
        Y.append(ys)
        print(k,ys)
    return np.array(xErrL),np.array(Y)

def sample_measurements(circuit, args):
    """Sample measurements from the circuit and display them."""
    num_qubits = circuit.num_qubits
    even_count = num_qubits // 2
    odd_count = even_count+1
    sampler = circuit.compile_sampler()
    one_sample = sampler.sample(shots=1)[0]
    
    print(f"Sampling measurements: num_qubits={num_qubits}, rounds={args.rounds}, n_odd={odd_count}  n_meas={len(one_sample)}")

    ntick=args.rounds+1
    for j in range(ntick):
        i0=j*even_count
        sl=one_sample[i0:i0+even_count]
        print(f't{j:02d}: ' + " ".join("1" if e else "_" for e in sl))
    sl=one_sample[-odd_count:]
    print(f'odd:' + " ".join("1" if e else "_" for e in sl))


def sample_detectors(circuit, args):
    """Sample detector results from the circuit and display them."""
    num_qubits = circuit.num_qubits
    odd_count = (num_qubits + 1) // 2
    even_count = num_qubits // 2

    total_dets =  (1+args.rounds) * even_count + odd_count
    print(f"Sampling detectors: num_qubits={num_qubits}, rounds={args.rounds}, det_total={total_dets}")

    detector_sampler = circuit.compile_detector_sampler()
    one_sample = detector_sampler.sample(shots=1)[0]
    for k in range(0, len(one_sample), even_count):
        timeslice = one_sample[k:k+even_count]
        j=k//even_count
        #print("".join("!" if e else "_" for e in timeslice))
        print('t%2d: '%j + " ".join("!" if e else "_" for e in timeslice))
    

# ==================================
# ==================================
#           MAIN
# ==================================
# ==================================

def main():
    """Main execution function."""
    args = get_parser()
    
    print(f"Generating stim circuit of type '{args.circType}'...")
    
    circuit = stim.Circuit.generated(
        args.circType,
        distance=args.distance,
        rounds=args.rounds,
        before_round_data_depolarization=args.before_round_data_depolarization,
        before_measure_flip_probability=args.before_measure_flip_probability
    )


    if args.dumpCirc:
        print("\nCircuit representation:")
        print(repr(circuit))

    if args.sampMeas:
        sample_measurements(circuit, args)

    if args.sampDet:
        print("\nSampling detectors:")
        sample_detectors(circuit, args)
         
    plotter = MyPlotter(args)
    if 'a' in args.showPlots:
        plotter.circuit_diagram(circuit, figId=1)

    roundL=[17,11,5]
    num_shots =100_000
    if 'b' in args.showPlots:
        plotter.jobName='logicalCode'
        xErrL=[   0.05, 0.1, 0.2, 0.5]  # for repet-code
        #xErrL=[5e-4,1e-3, 0.002,  0.005, 0.01, 0.02, 0.05]; num_shots =1000_000  # for color-code
        X,Y=scan_X_err( xErrL,roundL,args,shots=num_shots)
        plotter.vary_xError(X,Y,roundL,tit=args.circType, figId=2)
    plotter.display_all()
    

if __name__ == "__main__":
    main()
