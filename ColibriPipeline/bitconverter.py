"""
Filename:      
Author(s):  Peter Quigley
Contact:    pquigley@uwo.ca
Created:    Thu Oct  6 20:29:46 2022
Updated:    Thu Oct  6 20:29:46 2022
    
Usage:
$Description$
"""

import numba as nb
import numpy as np

# Function to read 12-bit data with Numba to speed things up
@nb.njit(nb.uint16[::1](nb.uint8[::1]),fastmath=True,parallel=True)
def conv_12to16(data_chunk):
    """
    Function to read 12-bit data with Numba to speed things up
    
        Parameters:
            data_chunk (arr): A contigous 1D array of uint8 data
                              eg.data_chunk = np.frombuffer(data_chunk, dtype=np.uint8)
   
        Returns:
            out (arr): Output data in 8-bit format
   """
   
    #ensure that the data_chunk has the right length
    assert np.mod(data_chunk.shape[0],3)==0

    out=np.empty(data_chunk.shape[0]//3*2,dtype=np.uint16)
 #   image1 = np.empty((2048,2048),dtype=np.uint16)
 #   image2 = np.empty((2048,2048),dtype=np.uint16)

    for i in nb.prange(data_chunk.shape[0]//3):
        fst_uint8=np.uint16(data_chunk[i*3])
        mid_uint8=np.uint16(data_chunk[i*3+1])
        lst_uint8=np.uint16(data_chunk[i*3+2])

        out[i*2] =   (fst_uint8 << 4) + (mid_uint8 >> 4)
        out[i*2+1] = ((mid_uint8 % 16) << 8) + lst_uint8

    return out