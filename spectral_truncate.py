import numpy as np
from matplotlib import pyplot as plt
from dedalus import public as de
from dedalus.core.operators import GeneralFunction
from dedalus.extras import flow_tools
import os
import time
import h5py

def truncate_data(input_size, output_size, filename):
    """
    reads in a dedalus created states file, crops the spectral coefficients
    from input_size to output_size, and then saves the data in trunc_data.h5
    """

    xlen_LR = output_size
    ylen_LR = output_size

    xsize = input_size
    ysize = input_size

    xlen = 2*np.pi
    ylen = 2*np.pi

    h5f_read = h5py.File(filename,'r')

    time_len = h5f_read['tasks/zeta'][:,0,0].shape[0]

    x_basis = de.Fourier('x', xsize, interval=(0,xlen), dealias=3/2)
    y_basis = de.Fourier('y', ysize, interval=(0,ylen), dealias=3/2)
    domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

    zeta_field = domain.new_field()
    zeta_coeffs = []

    #Now loop through the times that are saved.
    for ii in range(time_len):
        zeta_field['g'] = np.array(h5f_read['tasks/zeta'][ii,:,:])
       
        zeta_chunk1 = zeta_field['c'][0:int(xlen_LR/2),0:int(ylen_LR/2)-1]
        zeta_chunk2 = zeta_field['c'][0:int(xlen_LR/2),int(ysize/2)+1:int(ysize/2)+1+int(ylen_LR/2)]
        zeta_vals = np.concatenate((zeta_chunk1,zeta_chunk2),axis=1)

        zeta_coeffs.append(zeta_vals)

    h5f_new = h5py.File('trunc_data.h5','w')
    h5f_new.create_dataset('trunc_data', data = zeta_coeffs)
    h5f_new.close()
