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
        
        # Path 1: Mermaid (using the tket.circuit helper)
        try:
            m_file = f"hugr_mod_{i}.mmd"
            mermaid_str = tket.circuit.render_circuit_mermaid(mod)
            with open(m_file, "w") as f:
                f.write(mermaid_str)
            print(f"  Path 1 (Mermaid): Saved {m_file}")
        except BaseException as e:
             # Catching BaseException because PanicException can skip standard Exception
            print(f"  Path 1 (Mermaid): Skipped due to internal panic/error: {e}")

        # Path 2: Recursive search for gates in the tree
        def find_gates(node, depth=0):
            try:
                # We try both from_model and from_bytes for the specific node
                tk2 = tket.circuit.Tk2Circuit.from_model(mod, node)
                t1 = tk2.to_tket1()
                if t1.n_gates > 0:
                    print(f"{'  '*depth}--> [Node {node}] Found {t1.n_gates} gates!")
                    out = f"sub_circuit_{i}_{node}.html"
                    from pytket.circuit.display import render_circuit_as_html
                    render_circuit_as_html(t1, out)
                    print(f"{'  '*depth}    Saved plot to {out}")
                elif t1.n_qubits > 0:
                     print(f"{'  '*depth}    [Node {node}] Empty circuit with {t1.n_qubits} qubits")
            except BaseException:
                pass
            
            # Recurse to children
            try:
                children = list(mod.children(node))
                for child in children:
                    find_gates(child, depth + 1)
            except Exception:
                pass

        print("  Searching for gates in HUGR tree...")
        try:
            find_gates(mod.module_root)
        except Exception as e:
            print(f"  Tree search failed: {e}")


if __name__ == "__main__":
    project = qnx.projects.get_or_create(name='feb-guppy')
    qnx.context.set_active_project(project)
    inspect_hugr(sys.argv[1])
