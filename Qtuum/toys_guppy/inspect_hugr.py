#!/usr/bin/env python3
"""
Inspect & visualize a HUGR circuit fetched from Quantinuum Nexus.

Usage:
    python3 inspect_hugr.py <job_id>

Produces (in out/ directory):
    hugr_<tag>_mod<i>.dot   – Graphviz DOT source
    hugr_<tag>_mod<i>.png   – Rendered PNG image
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os
import sys

import qnexus as qnx

OUT_DIR = "out"


def inspect_hugr(job_id):
    tag = job_id.split("-")[0]  # e.g. "0c1c9000"
    os.makedirs(OUT_DIR, exist_ok=True)

    ref = qnx.jobs.get(id=job_id)
    results = qnx.jobs.results(ref)
    hugr_pkg = results[0].get_input().download_hugr()  # hugr.package.Package

    print(f"Package has {len(hugr_pkg.modules)} modules")
    for i, mod in enumerate(hugr_pkg.modules):
        print(f"\n--- Module {i}  ({mod.num_nodes()} nodes) ---")

        base = os.path.join(OUT_DIR, f"hugr_{tag}_mod{i}")
        digraph = mod.render_dot()  # graphviz.Digraph

        # Save DOT source
        dot_file = base + ".dot"
        with open(dot_file, "w") as f:
            f.write(digraph.source)
        print(f"  DOT saved: {dot_file}")

        # Render PNG via graphviz Python package (quiet=True suppresses dot warnings)
        digraph.render(base, format="png", cleanup=False, quiet=True)
        print(f"  PNG saved: {base}.png")

    print(f"\nDone. Output in {OUT_DIR}/")
    print(f"  open {OUT_DIR}/hugr_{tag}_mod0.png")
    print(f"  or paste .dot into https://dreampuf.github.io/GraphvizOnline/")


if __name__ == "__main__":
    project = qnx.projects.get_or_create(name='feb-guppy')
    qnx.context.set_active_project(project)
    inspect_hugr(sys.argv[1])
