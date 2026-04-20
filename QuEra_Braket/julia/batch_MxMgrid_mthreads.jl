#!/usr/bin/env julia
#= In the terminal navigate to the folder  w/ all .jl files
1 thread execute: ./batch_4x4grid_mthreads.jl 
OR
julia --threads  8 ./batch_4x4grid_mthreads.jl --nAtom 8
julia -t 4 ./batch_MxMgrid_mthreads.jl --nAtom 4

 One time on Mac
 sudo ln -sf /Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia /usr/local/bin/julia

 On PM
 module load PrgEnv-cray
 module load julia/1.7.2-cray

=#

using Pkg,Printf

@printf "Julia ver %s\n" VERSION
#Pkg.add("ArgParse")  #chicken-and-egg problem
T0 = time();
pkgL=[ "ArgParse", "SafeTestsets","HDF5"]

#---------------------------------
using ArgParse
function get_parser()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--basePath"; arg_type = String; help = "head of all IO directories"
           default = "data/" # Jan PM
        "--nAtom","-a"; arg_type = Int64;  default = 4;  help = "number of atoms in the grid, only square numbers are allowed"
        "--Rb"; arg_type = Float64;  default = 8.4;  help = "blockade radius (um)"
        "--compile"; action = :store_true; help = "compile all packages (optional)"
        "--subspace","-s"; action = :store_true; help = "use Bloqade subspace  (optional)"
	"--dist"; arg_type = Float64;  default = 6.7;  help = "name of RB sequence"
	"--expName"; arg_type = String;  default = "emu123";  help = "name of experiment"
    end

    args=parse_args(s)
    println("Parsed args:")
    for (arg,val) in args ;   println("  $arg  =>  $val");  end
    if !  isdir(args["basePath"])
      @printf "%s NOT exist, abort\n" args["basePath"]
      exit(99)
    end
    @assert args["nAtom"] in [4,9,16]
    return args
end


#---------------------------------
#---------------------------------
#   M A I N
#---------------------------------
#---------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
   # executed only if this file is called as exec

   args=get_parser()
   println(" args typs is:", typeof(args))
   #---------------------------------  takes many minutes
   if args["compile"]
   @printf "add %d packages , may take minutes... \n"  length(pkgL)
   for pkN in pkgL
     Pkg.add(pkN)
      @printf "added package %s, elaT=%.2f min  \n"  pkN (time()-T0)/60.
      end
   else
      @printf "skip compilation of %d packages\n"  length(pkgL)
      Pkg.status()
   end
   # Pkg.activate(".")

   println("loading Pkg's...")

   using  BloqadeExpr #HDF5
   #1BloqadeExpr.set_backend("ThreadedSparseCSR")
   println("BloqadeExpr.backend:",BloqadeExpr.backend )

   using Base.Threads
   println("avaliable threads: ",nthreads())
   myCode="MxM-mthread.jl"

   println("loading my tasks $myCode...")
   include(myCode)
   include("Util_Hdf5io.jl")

   lattice_run_time(4,args["dist"],args["Rb"],args["subspace"])  # forced compilation
   @printf("\nM: MAIN start elaT=%.1f sec  \n",time()-T0)

   println("executing my tasks...")
   probArr,mpvBitstr=lattice_run_time(args["nAtom"],args["dist"],args["Rb"],args["subspace"])

   # Write results to external file
   outF = args["basePath"]  *args["expName"] * ".mxmGrid.h5"
   println("outF:",outF)
   #1fibo =Int32.([1, 1, 2, 3, 5, 8, 13])
   #1println("fibo type:",typeof(fibo))
   bigD=Dict()
   #1bigD["fido"]=fibo
   bigD["probs"]=probArr
   #1println("DD:",bigD)
   write4_data_hdf5(bigD,outF)	


   #..... code below is not needed -keep it for eductational purposes
   nSol=size(mpvBitstr)[1]
   @printf("M:got %d MPV bitstr\n", nSol)
   for i in 1:nSol
     rec=mpvBitstr[i]
      #@printf("\nsol=%d bits", i)
       println("sol=$i ",rec)
      end
    @printf "S:done, total elapsed time %.1f min \n" (time()-T0)/60.

end # MAIN