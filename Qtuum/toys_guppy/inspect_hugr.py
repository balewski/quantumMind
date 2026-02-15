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
        root = mod.module_root
        children = list(mod.children(root))
        print(f"Root: {root}, Children: {len(children)}")
        for child in children:
            try:
                # Try to see if this child is a function definition
                tk2_c = tket.circuit.Tk2Circuit.from_model(mod, child)
                t1_c = tk2_c.to_tket1()
                print(f"  Child {child}: {t1_c.n_gates} gates")
            except Exception as e:
                print(f"  Child {child}: could not convert ({e})")

if __name__ == "__main__":
    inspect_hugr(sys.argv[1])
