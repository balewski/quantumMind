using Bloqade

# can show speedup with different backends
function lattice_run_time()

    # 4x4 lattice of atoms with adiabatic sweep to create Z2 ordered phase
    atom_distance = 6.7;
    atoms = generate_sites(SquareLattice(), 4, 4, scale = atom_distance);

    # define waveforms
    Ω_max = 2π * 2.5;
    Δ_val = 2π * 10.0;
    total_time = 3.0;
    time_ramp  = total_time * 0.083;
    clocks = [0, time_ramp, total_time - time_ramp, total_time];

    Ω = piecewise_linear(clocks = clocks, values = [0.0, Ω_max, Ω_max, 0.0]); 
    Δ = piecewise_linear(clocks = clocks, values = [-Δ_val, -Δ_val, Δ_val, Δ_val]);
    ϕ = constant(;duration = total_time, value = π);

    h = rydberg_h(atoms; Ω=Ω, Δ=Δ, ϕ=ϕ);

    reg = zero_state(16);

    total_time = 3.0
    println("Creating SchrodingerProblem...")
    prob = SchrodingerProblem(reg, total_time, h;algo=DP8());
    println("Running Emulation...")
    @time emulate!(prob);

    return nothing

end

