#!/usr/bin/env julia
# = = = = =  HD5 advanced storage = = =
#=
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
=#

using Printf, HDF5

#...!...!..................
function write4_data_hdf5(dataD,outF)
    #1println("DDin:",dataD)
    
    #1return nothing
    @printf("open to write:%s\n",outF)
    file3 = h5open(outF, "w")
    for (key, val) in dataD
	println("store key=$key size=",size(val))
	if ndims(val)==1
	   valT=val
	else
	  @assert ndims(val)==2
	  valT=convert(Matrix,transpose(val))
	end
	#println("tt",typeof(val), typeof(valT),typeof(val2))
	write(file3, key,valT)
    end
    close(file3)
    println(" Save phases to $outF")
end

#---------------------------------
#---------------------------------
#   M A I N
#---------------------------------
#---------------------------------
if abspath(PROGRAM_FILE) == @__FILE__ 
   # executed only if this file is called as exec
   outF="out/a1.h5"

   fibo =Int32.([1, 1, 2, 3, 5, 8, 13])
   println("fibo type:",typeof(fibo))
   C = Float32.(zeros(2,3))
   [ C[1,i] = i for i in 1:3]
   C[2,:] = rand(3);

   println("C type:",typeof(C))
   bigD=Dict()
   bigD["C"]=C
   bigD["fido"]=fibo

   println("DD:",bigD)
   write4_data_hdf5(bigD,outF)	

   @printf("Done\n")
   

end  #MAIN