from spectral_truncate import truncate_data
from truncate2dedalus import truncate2dedalus

truncate_data(1024, 256, 'states_s1.h5')
truncate2dedalus(1024, 256, 'trunc_data.h5')
