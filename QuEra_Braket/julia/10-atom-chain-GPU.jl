using Bloqade
using CUDA, Adapt

function chain_run_time_CPU()

    # define an adiabatic sweep to create a Z2 ordered phase on a chain of ten atoms

    total_time = 3.0;
    Ω_max = 2π * 4;
    Ω = piecewise_linear(clocks = [0.0, 0.1, 2.1, 2.2, total_time], values = [0.0, Ω_max, Ω_max, 0, 0]);

    U1 = -2π * 10;
    U2 = 2π * 10;
    Δ = piecewise_linear(clocks = [0.0, 0.6, 2.1, total_time], values = [U1, U1, U2, U2]);

    nsites = 14
    atoms = generate_sites(ChainLattice(), nsites, scale = 5.72)

    h = rydberg_h(atoms; Δ, Ω)
    reg = zero_state(nsites);
    println("Creating SchrodingerProblem")
    problem = SchrodingerProblem(reg, total_time, h);

    println("Measuring CPU time")
    @time emulate!(problem)
    
    # need to bring the problem back into main memory
    return nothing
end

function chain_run_time_GPU()

    # define an adiabatic sweep to create a Z2 ordered phase on a chain of fourteen atoms

    total_time = 3.0;
    Ω_max = 2π * 4;
    Ω = piecewise_linear(clocks = [0.0, 0.1, 2.1, 2.2, total_time], values = [0.0, Ω_max, Ω_max, 0, 0]);

    U1 = -2π * 10;
    U2 = 2π * 10;
    Δ = piecewise_linear(clocks = [0.0, 0.6, 2.1, total_time], values = [U1, U1, U2, U2]);

    nsites = 14
    atoms = generate_sites(ChainLattice(), nsites, scale = 5.72)

    h = rydberg_h(atoms; Δ, Ω)
    reg = zero_state(nsites);
    println("Creating SchrodingerProblem")
    problem = SchrodingerProblem(reg, total_time, h);

    # Take advantage of multiple dispatch here, objects that are CuArrays 
    # automatically have GPU routines applied to them versus CPU implementation
    println("Measuring GPU load/execution/unload time")
    @time begin
        gpu_problem = adapt(CuArray, problem)
        emulate!(gpu_problem)
        offloaded_problem = adapt(Array, gpu_problem)
    end
    
    # need to bring the problem back into main memory
    return nothing

end