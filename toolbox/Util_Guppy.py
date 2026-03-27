def guppy_to_qiskit(qcGF,nq):  # qcG is guppy circuit function
    from selene_sim import build, Quest, CircuitExtractor
    from selene_sim.result_handling.parse_shot import postprocess_unparsed_stream
    from pytket.extensions.qiskit import tk_to_qiskit 

    runner = build(qcGF.compile(), "panic")
    circuit_extractor = CircuitExtractor()
    shots, error = postprocess_unparsed_stream(
        runner.run_shots(
            Quest(),
            n_qubits=nq,
            n_shots=1,
            parse_results=True,
            event_hook=circuit_extractor
        )
    )
    shot_number = 0
    qcTK = circuit_extractor.shots[shot_number].get_user_circuit()
    #print(type(qcTK))
    qcQi=tk_to_qiskit(qcTK)
    #print(qcQi)
    return qcQi
