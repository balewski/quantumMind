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
        print(f"  Entrypoint: {mod.entrypoint}, Op: {mod.entrypoint_op(mod.entrypoint)}")
        
        # Path 1: Mermaid (Best for browser/Live Editor)
        try:
            mermaid_str = mod.render_mermaid()
            m_file = f"hugr_mod_{i}.mmd"
            with open(m_file, "w") as f:
                f.write(mermaid_str)
            print(f"  Path 3 (Mermaid): Saved {m_file} (Paste into mermaid.live)")
        except Exception as e:
            print(f"  Path 3 (Mermaid): Fail ({e})")

        # Path 4: Brute force search for Gates in children
        for child in mod.children(mod.module_root):
            try:
                tk2 = tket.circuit.Tk2Circuit.from_model(mod, child)
                t1 = tk2.to_tket1()
                if t1.n_gates > 0:
                    print(f"  --> Found gates in child {child}! Gates: {t1.n_gates}")
                    # Try to save this specific part
                    from pytket.circuit.display import render_circuit_as_html
                    out = f"sub_circuit_{i}_{child}.html"
                    render_circuit_as_html(t1, out)
                    print(f"      Saved partial plot to {out}")
            except Exception:
                continue


if __name__ == "__main__":
    project = qnx.projects.get_or_create(name='feb-guppy')
    qnx.context.set_active_project(project)
    inspect_hugr(sys.argv[1])
