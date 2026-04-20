using Bloqade
using Printf
#= modiffied example for use in batch mode
* num atoms is variable
* returns 3 MPV bitstrimngs as 1D : Vector{DitStr{2, 16, Int64}} 
=#


# https://github.com/QuantumBFS/Yao.jl/blob/b971dd440b86f89f3e517a36472c39bcb6638060/lib/YaoArrayRegister/src/register.jl#L791

function sort_by_prob(reg::ArrayReg{D}) where D
    imax = sortperm(probs(reg); rev=true)
    return DitStr{D,nqudits(reg)}.(imax .- 1)
end


function lattice_run_time(nAtom,dist,Rb,doSubspace)
    @printf("num atoms %d, dist=%.2f um,  doSubspace=%d\n", nAtom,dist,doSubspace)
    
    nAtomBase=isqrt(nAtom)
    @assert nAtomBase*nAtomBase==nAtom
    # NxN lattice of atoms with adiabatic sweep to create Z2 ordered phase
    atom_distance = dist;
    atoms = generate_sites(SquareLattice(), nAtomBase, nAtomBase, scale = atom_distance);

    # optional acceleration of computation
    if doSubspace
       @printf("do subspace,  blockade R=%.2f um\n",Rb)
       subspace = blockade_subspace(atoms, Rb)
       println("subspace size:",typeof(subspace))
       reg = zero_state(subspace);
    else
       reg = zero_state(nAtom);
    end    

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

    total_time = 3.0
    println("Creating SchrodingerProblem for nAtom=$nAtom")
    prob = SchrodingerProblem(reg, total_time, h;algo=DP8());
    println("Running Emulation...")
    @time emulate!(prob);

    #1println("prob.reg",prob.reg)
    #1println("prob.reg.state",prob.reg.state)
    
    
    #println("\nbb ",nqudits(prob.reg))
    #println("\ncc ",nqudits(prob.reg.nbatch))	# type ArrayReg has no field nbatch
    
    probsL=probs(prob.reg)
    #1println("\naa ",probsL,typeof(probsL))		
    #..... code below is not needed -keep it for eductational purposes
    
    numMpv=min(nAtom,3)  # for 1 atom thare are only 2 bit-strings possible
    #1mpvBitstr = sort_by_prob(prob.reg)  # my version 
    mpvBitstr = most_probable(prob.reg, numMpv)

    pbA=zeros(2^nAtom)
    
    for bitstring in mpvBitstr
    	iaddr=Int(bitstring)+1
	pval=probsL[ iaddr]
	pbA[iaddr]=pval
	#println("zz",typeof(bitstring),bitstring,"int:",iaddr)
    end
    #1println("pbA:",pbA)
    
    #println( "mpv-bitstr size:", size(mpvBitstr)[1])

    return probsL,mpvBitstr

end

