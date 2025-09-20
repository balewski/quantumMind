#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import perceval as pcvl
from pprint import pprint

ascella = pcvl.RemoteProcessor("qpu:ascella")

print("\nThe optimal performance is: HOM = 100%, Transmittance = 100% and g2 = 0%. The Clock tells you how many photons are generated per second. (80 MHz means 80 million photons per second) \n")

perf_ascella = ascella.performance
print(f"The Performance of Ascella is: {perf_ascella}")
specs = ascella.specs
#pdisplay(specs["specific_circuit"])
print(ascella.name,"Platform constraints:")
pprint(specs["constraints"])
print("\nPlatform supported parameters:")
pprint(specs["parameters"])


belenos = pcvl.RemoteProcessor("qpu:belenos")
perf_belenos = belenos.performance
specs = belenos.specs
print(f"\nThe Performance of Belenos is: {perf_belenos}")
print(belenos.name,"Platform constraints:")
pprint(specs["constraints"])

