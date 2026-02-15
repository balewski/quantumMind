import qnexus as qnx
import tket.circuit
import sys

def inspect_hugr(job_id):
    ref = qnx.jobs.get(id=job_id)
    results = qnx.jobs.results(ref)
    hugr_pkg_raw = results[0].get_input().download_hugr()
    pack = tket.circuit.Package.from_bytes(hugr_pkg_raw.to_bytes())
    
    print(f"Package has {len(pack.modules)} modules")
    for i, mod in enumerate(pack.modules):
        print(f"\n--- Module {i} ---")
        # Try Path 1: Tk2Circuit from_bytes (The modern envelope way)
        try:
            tk2 = tket.circuit.Tk2Circuit.from_bytes(mod.to_bytes())
            print(f"  Path 1 (from_bytes): Success! {tk2.to_tket1().n_gates} gates")
        except Exception as e:
            print(f"  Path 1 (from_bytes): Fail ({e})")

        # Try Path 1b: Tk2Circuit from_str (The text envelope way)
        try:
            tk2 = tket.circuit.Tk2Circuit.from_str(mod.to_str())
            print(f"  Path 1b (from_str): Success! {tk2.to_tket1().n_gates} gates")
        except Exception as e:
            print(f"  Path 1b (from_str): Fail ({e})")

        # Try Path 2: Fallback to Graphviz visualization
        try:
            digraph = mod.render_dot()
            dot_file = f"hugr_mod_{i}.dot"
            digraph.save(dot_file)
            print(f"  Path 2 (Graphviz): Saved {dot_file}")
            # Try to render to PNG using the digraph object directly
            png_file = digraph.render(f"hugr_mod_{i}", format="png", cleanup=True)
            print(f"  Path 2 (Graphviz): Rendered {png_file}")
        except Exception as e:
            print(f"  Path 2 (Graphviz): Fail ({e})")


if __name__ == "__main__":
    project = qnx.projects.get_or_create(name='feb-guppy')
    qnx.context.set_active_project(project)
    inspect_hugr(sys.argv[1])
