import numpy as np
import h5py, time, os
# handles: array,  scalar--> array[1], strings as-is

#...!...!..................
def write_data_hdf5(dataD,outF,verb=1):
    dtvs = h5py.special_dtype(vlen=str)
    h5f = h5py.File(outF, 'w')
    if verb>0:
            print('saving data as hdf5:',outF)
            start = time.time()
    for item in dataD:
        rec=dataD[item]
        if type(rec)==str: # special case
            dset = h5f.create_dataset(item, (len(rec),), dtype=dtvs)
            dset[0]=rec
            if verb>0:print('h5-write :',item, 'as string',dset.shape,dset.dtype)
            continue
        if type(rec)!=np.ndarray:
            rec=np.array([rec])
        if verb>1: print('x=',item,type(rec))
        h5f.create_dataset(item, data=rec)
        if verb>0:print('h5-write :',item, rec.shape,rec.dtype)
    h5f.close()
    xx=os.path.getsize(outF)/1048576
    print('closed  hdf5:',outF,' size=%.2f MB, elaT=%.1f sec'%(xx,(time.time() - start)))
    
#...!...!..................
def read_data_hdf5(inpF,verb=1):
    if verb>0:
            print('read data from hdf5:',inpF)
            start = time.time()
    h5f = h5py.File(inpF, 'r')
    objD={}
    for x in h5f.keys():
        if verb>1: print('item=',x,type(h5f[x]),h5f[x].shape,h5f[x].dtype)
        if h5f[x].dtype==object:
            obj=h5f[x][0]
            #print('bbb',type(obj),obj.dtype)
            if verb>0: print('read str:',x,len(obj),type(obj))
        else:
            obj=h5f[x][:]
            if verb>0: print('read obj:',x,obj.shape,obj.dtype)
        objD[x]=obj
    if verb>0:
            print(' done h5, num rec:%d  elaT=%.1f sec'%(len(objD),(time.time() - start)))

    h5f.close()

    return objD
