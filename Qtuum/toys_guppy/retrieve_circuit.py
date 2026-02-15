#!/usr/bin/env python3
"""
Retrieve a compiled circuit from Quantinuum Nexus by job ID,
download it as a pytket Circuit, and render it as an HTML file.

Usage:
    python retrieve_circuit.py <execute-job-id> [--project <project-name>]

The script:
  1. Reconstructs an ExecuteJobRef from the supplied UUID.
  2. Prints job status and metadata.
  3. For each result in the job, downloads the compiled circuit
     (pytket Circuit) and saves an interactive HTML rendering.

Authentication uses the same env-var scheme as the other toys:
    export MY_QTUUM_NAME=...
    export MY_QTUUM_PASS=...
(only needed if the cached Nexus token has expired).
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import argparse
import json
import os
import sys
from pathlib import Path
from pprint import pprint

import qnexus as qnx


# ---- auth ----

def ensure_auth():
    """Re-authenticate via env vars if the cached token is stale."""
    try:
        qnx.users.get_self()          # quick connectivity check
    except Exception:
        name = os.environ.get("MY_QTUUM_NAME")
        pw   = os.environ.get("MY_QTUUM_PASS")
        if not name or not pw:
            sys.exit("ERROR: Nexus token expired and MY_QTUUM_NAME / "
                     "MY_QTUUM_PASS are not set.")
        qnx.auth._request_tokens(name, pw)
        print("Re-authenticated as", name)


# ---- job lookup ----

def get_job_ref(job_id: str, project_name: str | None = None):
    """Look up an execute job by its UUID.

    If *project_name* is given we scope the search to that project,
    otherwise we search across all projects the user can see.
    """
    filters = {"name_like": ""}  # match anything
    if project_name:
        project = qnx.projects.get_or_create(name=project_name)
        qnx.context.set_active_project(project)

    # Try to find the job by listing recent execute jobs
    # and matching by ID.  The qnexus SDK does not expose a
    # direct get-by-id for jobs, so we reconstruct the ref.
    from qnexus.models.references import ExecuteJobRef

    # Minimal reconstruction – only the id is required for
    # status / results queries.
    ref = ExecuteJobRef(**{
        "id": job_id,
        "annotations": {
            "name": "",
            "description": "",
            "properties": {},
            "created": "2000-01-01T00:00:00+00:00",
            "modified": "2000-01-01T00:00:00+00:00",
        },
        "job_type": "execute",
        "last_status": "SUBMITTED",
        "last_message": "",
        "project": {
            "id": "00000000-0000-0000-0000-000000000000",
            "annotations": {
                "name": project_name or "unknown",
                "description": "",
                "properties": {},
                "created": "2000-01-01T00:00:00+00:00",
                "modified": "2000-01-01T00:00:00+00:00",
            },
            "contents_modified": "2000-01-01T00:00:00+00:00",
            "archived": False,
            "type": "ProjectRef",
        },
        "type": "ExecuteJobRef",
    })
    return ref


# ---- circuit rendering ----

def render_circuit_html(circuit, out_path: Path):
    """Save an interactive HTML rendering of a pytket Circuit."""
    from pytket.circuit.display import render_circuit_as_html
    html = render_circuit_as_html(circuit)
    out_path.write_text(html)
    print(f"  → saved {out_path}  ({out_path.stat().st_size} bytes)")


def render_circuit_text(circuit):
    """Print a compact text summary of the circuit."""
    cmds = circuit.get_commands()
    print(f"  gates: {len(cmds)},  qubits: {circuit.n_qubits},  "
          f"bits: {circuit.n_bits}")
    for cmd in cmds[:30]:
        print(f"    {cmd}")
    if len(cmds) > 30:
        print(f"    ... ({len(cmds) - 30} more)")


# ---- main ----

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve circuit from Nexus execute job and render it.")
    parser.add_argument("job_id",
                        help="UUID of the Nexus execute job")
    parser.add_argument("--project", default='feb-guppy',
                        help="Nexus project name (optional)")
    parser.add_argument("--outdir", default=".",
                        help="Directory for output HTML files (default: cwd)")
    args = parser.parse_args()

    ensure_auth()

    # ---- Step 1: reconstruct the job reference ----
    ref_exec = get_job_ref(args.job_id, args.project)

    # Note: status is an ExecutionStatus object, status.status is the Enum
    full_status = qnx.jobs.status(ref_exec)
    status_enum = full_status.status if hasattr(full_status, 'status') else full_status
    print(f"\n=== Job {args.job_id} ===")
    print(f"  status: {full_status}")

    if "COMPLETED" not in str(status_enum):
        print("  ⚠  Job is not complete yet — circuit may not be available.")

    # ---- Step 2: retrieve results & compiled circuits ----
    try:
        results = qnx.jobs.results(ref_exec)
        print(f"  results: {len(results)} program(s)\n")
    except Exception as exc:
        print(f"  (could not retrieve results: {exc})")
        results = []

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, res_item in enumerate(results):
        print(f"--- Program {idx} ---")

        # Try to get the input program reference
        try:
            circ_ref = res_item.get_input()
            print(f"  input ref: {circ_ref}")
        except Exception as exc:
            print(f"  (could not get input ref: {exc})")
            circ_ref = None

        # Download the pytket Circuit if possible
        circuit = None
        if circ_ref is not None:
            try:
                # Case A: Native pytket Circuit
                if hasattr(circ_ref, "download_circuit"):
                    circuit = circ_ref.download_circuit()
                    print(f"  downloaded pytket Circuit")
                
                # Case B: HUGR from Guppy
                elif "HUGRRef" in str(type(circ_ref)):
                    try:
                        import tket.circuit
                        hugr_pkg_raw = circ_ref.download_hugr()
                        pack = tket.circuit.Package.from_bytes(hugr_pkg_raw.to_bytes())
                        
                        # Navigate modules to find the circuit function
                        for mod in pack.modules:
                            try:
                                # Use from_model to avoid the to_dict serialization panic
                                tk2_c = tket.circuit.Tk2Circuit.from_model(mod)
                                tmp_c = tk2_c.to_tket1()
                                if tmp_c.n_gates > 0:
                                    circuit = tmp_c
                                    print(f"  found circuit in HUGR module")
                                    break
                            except Exception:
                                continue
                        
                        if circuit is None:
                            print("  (HUGR package contains no Gates)")
                    except Exception as exc:
                        print(f"  (failed to convert HUGR: {exc})")
            except Exception as exc:
                print(f"  (download failed: {exc})")

        if circuit is not None:
            render_circuit_text(circuit)
            out_file = out_dir / f"circuit_{args.job_id[:8]}_{idx}.html"
            from pytket.circuit.display import render_circuit_as_html
            render_circuit_as_html(circuit, str(out_file))
            print(f"  → saved {out_file}  ({out_file.stat().st_size} bytes)")
        else:
            print("  ⚠  Could not obtain a pytket Circuit for rendering.")
            print("     (HUGR programs may not yet support back-conversion)")

        # Print result counts if available
        try:
            res_data = res_item.download_result()
            # Handle standard pytket results and QsysResults
            if hasattr(res_data, "get_counts"):
                counts = res_data.get_counts()
                print("\nResult Counts:")
                pprint(counts)
            elif hasattr(res_data, "results"):
                # Guppy QsysResult case
                print(f"\nResult: {len(res_data.results)} shots collected.")
        except Exception as exc:
            print(f"  (could not process result counts: {exc})")

        print()


if __name__ == "__main__":
    main()
