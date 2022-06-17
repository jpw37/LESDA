import numpy as np
from matplotlib import pyplot as plt
from dedalus import public as de
from dedalus.core.operators import GeneralFunction
from dedalus.extras import flow_tools
import os
import time
import h5py

def truncate2dedalus(input_size, output_size, filename):
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

    time_len = h5f_read['trunc_data'][:,0,0].shape[0]

    x_basis = de.Fourier('x', xlen_LR, interval=(0,xlen), dealias=3/2)
    y_basis = de.Fourier('y', ylen_LR, interval=(0,ylen), dealias=3/2)
    domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

    zeta_field = domain.new_field()
    zeta_grid = []

    #Now loop through the times that are saved.
    for ii in range(time_len):
        zeta_field['c'] = np.array(h5f_read['trunc_data'][ii,:,:])
       
        zeta_grid.append(zeta_field['g'])#JPW: somehow this is saving everything as complex128 instead of real64

    h5f_new = h5py.File('grid_data.h5','w')
    h5f_new.create_dataset('grid_data', data = zeta_grid)
    h5f_new.close()
