#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

''' = = = = =  Numpy NPZ advanced storage = = =
It can hold:
* python dictionaries which must pass: json.dumps(dict)
* single float or int variables w/o np-array packing. It is recovered as 1-value array
* arbitrary numpy array (a large size payloads)
   - for an array of arbitrary strings must declare  dtype='object' at write and use .decode("utf-8")  to unpack 
'''

import numpy as np
import time, os
import json
import logging
from pprint import pprint

logging.basicConfig(level=logging.INFO)
# Configure logger
log = logging.getLogger(__name__)

#...!...!..................
def write_data_npz(dataD, outF, metaD=None, verb=1):
    assert dataD is not None
    assert len(outF) > 0
    
    # Create a copy to avoid modifying original data
    saveD = dataD.copy()
    
    if 'meta.JSON' in saveD:
        raise ValueError("Key 'meta.JSON' is reserved for internal use.")

    if metaD is not None:
        metaJ = json.dumps(metaD, default=str)
        saveD['meta.JSON'] = np.array([metaJ], dtype='object')
    
    if verb > 1:
        log.info(f'saving data as npz: {outF}')
    start = time.time()
    
    # Process data to ensure all items are numpy arrays
    for item in list(saveD.keys()):
        rec = saveD[item]
        if verb > 1:
            log.info(f'x={item} {type(rec)}')
        
        if isinstance(rec, str): # special case - convert string to object array
            saveD[item] = np.array([rec], dtype='object')
            log.debug(f'npz-write :{item} as string array, shape={saveD[item].shape} dtype={saveD[item].dtype}')
            continue
        
        if not isinstance(rec, np.ndarray): # packs a single value into np-array
            saveD[item] = np.array([rec])
            log.debug(f'npz-write :{item} packed as {saveD[item].shape} {saveD[item].dtype}')
        else:
            log.debug(f'npz-write :{item} shape={rec.shape} dtype={rec.dtype}')

    # Save to NPZ format
    np.savez_compressed(outF, **saveD)
    
    if verb > 0:
        xx = os.path.getsize(outF) / 1048576
        log.info(f'closed  npz: {outF}  size={xx:.2f} MB, elaT={(time.time() - start):.1f} sec')
    
    
#...!...!..................
def read_data_npz(inpF, verb=1):
    if verb > 0:
        log.info(f'read data from npz: {inpF}')
        start = time.time()
    
    objD = {}
    
    # Use context manager to ensure file is closed
    with np.load(inpF, allow_pickle=True) as npzData:
        for x in npzData.files:
            obj = npzData[x]
            if verb > 1:
                log.debug(f'\nitem={x} {type(obj)} {obj.shape} {obj.dtype}')
            
            if obj.dtype == object:
                if verb > 1:
                    log.info(f'read obj: {x} {len(obj)} {type(obj)}')
            else:
                if verb > 1:
                    log.info(f'read obj: {x} {obj.shape} {obj.dtype}')
            objD[x] = obj
    
    inpMD = None
    if 'meta.JSON' in objD:
        try:
            inpMD = json.loads(objD.pop('meta.JSON')[0])
            if verb > 1:
                log.debug(f'  recovered meta-data with {len(inpMD)} keys')
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            log.warning(f'Failed to recover meta-data: {e}')
            inpMD = None
            
    if verb > 1:
        log.info(f' done npz, num rec:{len(objD)}  elaT={(time.time() - start):.1f} sec')

    return objD, inpMD


#=================================
#=================================
#   U N I T   T E S T
#=================================
#=================================

if __name__=="__main__":
    # Setup basic configuration for logging during test
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print('testing npzIO ver 1')
    outF = 'abcTest.npz'
    verb = 1
    
    var1 = float(15) # single variable
    one = np.zeros(shape=5, dtype=np.int16); one[3] = 3
    two = np.zeros(shape=(2,3)); two[1,2] = 4

    three = np.empty((2), dtype='object')
    three[0] = 'record aaaa'
    three[1] = 'much longer record bbb'
    # WARN:  all decalred elements of three[] must be initialized before writeing NPZ
    
    # this works too:
    # three=np.array(['record aaaa','much longer record bbb'], dtype='object')
    
    text = 'This is text1'  

    metaD = {"age": 17, "dom": "white", "dates": [11, 22, 33]}
   
    outD = {'one': one, 'two': two, 'var1': var1, 'atext': text, 'three': three}

    write_data_npz(outD, outF, metaD=metaD, verb=verb)

    print('\nM: *****  verify by reading it back from', outF)
    big, meta2 = read_data_npz(outF, verb=verb)
    
    print(' recovered meta-data'); pprint(meta2)
    print('dump read-in data')
    for x in big:
        print('\nkey=', x); pprint(big[x])
  
    #get one string from string-array
    rec2 = big['three'][1] 
    print('rec2:', type(rec2), rec2)
    print(f'\n check raw content with: python -c "import numpy as np; data=np.load(\'{outF}\', allow_pickle=True); print(list(data.files)); data.close()"\n')
