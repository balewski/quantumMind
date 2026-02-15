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
            # Defensive import/check to avoid panics during top-level calls
            import tket.circuit
            mermaid_str = tket.circuit.render_circuit_mermaid(mod)
            with open(m_file, "w") as f:
                f.write(mermaid_str)
            print(f"  Path 1 (Mermaid): Saved {m_file}")
        except Exception as e:
             # This often panics with "AttributeError: 'Hugr' object has no attribute 'to_dict'"
            print(f"  Path 1 (Mermaid): Skipped (Internal bridge error: {e})")

        # Path 2: Recursive search for gates in the tree
        def find_gates(node, depth=0):
            # ... same as before but inside another try to be safe ...
            try:
                # We try both from_model and from_bytes for the specific node
                tk2 = tket.circuit.Tk2Circuit.from_model(mod, node)
                t1 = tk2.to_tket1()
                if t1.n_gates > 0:
                    print(f"{'  '*depth}--> Found {t1.n_gates} gates in Node {node}")
                    out = f"sub_circuit_{i}_{node}.html"
                    from pytket.circuit.display import render_circuit_as_html
                    render_circuit_as_html(t1, out)
                    print(f"{'  '*depth}    Saved plot to {out}")
            except Exception:
                pass
            
            # Recurse to children
            try:
                for child in mod.children(node):
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
