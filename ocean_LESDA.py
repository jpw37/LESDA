import numpy as np
from matplotlib import pyplot as plt
from dedalus import public as de
from dedalus.core.operators import GeneralFunction
from dedalus.extras import flow_tools
from base_simulator import BaseSimulator
import os
import time
import h5py
import logging
from mpi4py import MPI



RANK = MPI.COMM_WORLD.rank        # Which process this is running on 
SIZE = MPI.COMM_WORLD.size        # How many processes are running

class Ocean_LESDA(BaseSimulator):

    def __init__(self):

        log = True
        # Log file formatting                                                                                                                          
        LOG_FORMAT = "(%(asctime)s, {:0>2}/{:0>2}) %(levelname)s: %(message)s".format(
                                                                RANK+1, SIZE)
        LOG_FILE = "process{:0>2}.log".format(RANK)

        # Scales
        nu4 = 1e-6   # initial guess for bi-Laplacian coefficient
        self.nu4 = nu4
        time_scale = 1.2e6 # seconds
        length_scale = 504 * 1e4/np.pi # meters
        self.time_scale = time_scale
        self.length_scale = length_scale

        # Load the truncated Fourier coefficients
        self.load_data('grid_data.h5')

        # Parameters
        xsize, ysize = 256, 256
        xlen, ylen = 2*np.pi, 2*np.pi
        Cd = 1.25e-8 * length_scale # C_D/h, coefficient of quadratic bottom drag
        nu = 88 * time_scale/(length_scale**2) # viscosity
        mu = 100

        # Create bases and domain
        x_basis = de.Fourier('x', xsize, interval=(0,xlen), dealias=3/2)
        y_basis = de.Fourier('y', ysize, interval=(0,ylen), dealias=3/2)
        self.domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

        # Initialize variables and problem
        self.problem = de.IVP(self.domain, variables=['psi', 'zeta'])

        # Setup parameters
        self.problem.parameters['xlen'] = xlen
        self.problem.parameters['ylen'] = ylen
        self.problem.parameters['xsize'] = xsize
        self.problem.parameters['ysize'] = ysize
        self.problem.parameters['Cd'] = Cd
        self.problem.parameters['nu'] = nu
        self.problem.parameters['mu'] = mu
        self.problem.parameters['nu4'] = nu4
        self.problem.parameters['F'] = GeneralFunction(self.problem.domain, 'g', self.const_val, args=[])
        self.problem.parameters['nudge'] = GeneralFunction(self.problem.domain, 'g', self.nudging, args=[])
        self.problem.parameters['delta_nu4'] = GeneralFunction(self.problem.domain, 'g', self.const_val, args=[])

        # Setup auxiliary equations
        self.problem.substitutions['v'] = "-dy(psi)"
        self.problem.substitutions['w'] = "dx(psi)"
        self.problem.substitutions['u'] = 'sqrt(v*v + w*w)'
        self.problem.add_equation("zeta - dx(dx(psi)) - dy(dy(psi)) = 0", condition="(nx != 0) or (ny != 0)")

        # Setup evolution equations
        self.problem.add_equation('dt(zeta) + nu4*(dx(dx(dx(dx(zeta)))) + dy(dy(dy(dy(zeta)))) + 2*dx(dx(dy(dy(zeta))))) )- nu*(dx(dx(zeta)) + dy(dy(zeta))) = - v*dx(zeta) - w*dy(zeta) - Cd*(dx(w*u) - dy(v*u)) + F - mu*(zeta-nudge) - delta_nu4*(dx(dx(dx(dx(zeta)))) + dy(dy(dy(dy(zeta)))) + 2*dx(dx(dy(dy(zeta))))) ')

        # Add additional conditions
        self.problem.add_equation("psi = 0", condition="(nx == 0) and (ny == 0)")

        # Build solver
        self.solver = self.problem.build_solver('RK443')

        # Set stopping criteria
        self.solver.stop_sim_time = 10
        self.solver.stop_wall_time = np.inf
        self.solver.stop_iteration = np.inf

        # Set initial conditions
        x, y = self.domain.grid(0), self.domain.grid(1)
        zeta = self.solver.state['zeta']
        zeta.set_scales(1)
        zeta['g'] = np.exp(-x*y)*np.sin(x+y)

#        self.logger.info("BaseSimulator constructed")

        # Setup saving
        save = 0.01 # frequency of saving (in simulation time units
        
        records_dir = 'ocean' + time.strftime("__%m_%d_%Y__%H_%M")
        self.records_dir = records_dir
        if os.path.isdir(records_dir):  # Notify user if directory exists.
            print("Connecting to existing directory {}".format(records_dir))
        else:                           # Default: make new directory.
            try:
                os.mkdir(records_dir)
                print("Created new directory {}".format(records_dir))
        
                os.mkdir(os.path.join(records_dir, "states"))
            except FileExistsError:     # Other process might create it first.
                pass

        logger = logging.getLogger(records_dir + str(RANK))
        logger.setLevel(logging.DEBUG)
        if log and len(logger.handlers) == 0:
            logfile = logging.FileHandler(os.path.join(records_dir, LOG_FILE))
            logfile.setFormatter(logging.Formatter(LOG_FORMAT))
            logfile.setLevel(logging.DEBUG)
            logger.addHandler(logfile)
        self.logger = logger


        snaps = self.solver.evaluator.add_file_handler(os.path.join(records_dir, "states"), sim_dt=save, max_writes=5000, mode="append")
        snaps.add_task('zeta')
        snaps.add_task('psi')

    def load_data(self, file_name):
        """
        Loads the state from a previous run in order to nudge the current
        low resolution simulation to that higher resolved state
        
        Parameters:
            file_name (string): the name of the file to load the data from.
        """
        h5file = h5py.File(file_name,'r')
        self.true_zeta_grid = np.array(h5file['grid_data'])
        h5file.close()
#        with h5py.File(file_name, 'r') as states:
#            self.true_zeta_coeffs = states['trunc_data']

    #JPW: need to fix this for returning without set_scales
    def nudging(self):
        ind = self.solver.iteration
        zeta_val = self.problem.domain.new_field()
        chunk = np.shape(self.true_zeta_grid)[2] // SIZE
        subset = self.true_zeta_grid[ind,:,RANK*chunk:(RANK+1)*chunk]
        subset = np.float64(subset)
        zeta_val['g'] = subset
        
        zeta_val.set_scales(3/2)
        return zeta_val['g']

    def const_val(self, value, return_field=False):
        """
        Assuming that the problem is already defined, create a new field on the
        problem's domain with constant value given by the argument value.

        Parameters:
            value (numeric): the value which will be given to the new field.
        """
        coefficient_field = self.problem.domain.new_field()
        coefficient_field['g'] = value
        if return_field:
            return coefficient_field
        else:
            coefficient_field.set_scales(3/2)
            return coefficient_field['g']


    def new_params(self):
        """new_params take the current state and estimates the new parameter
        """

        # Set scales

        ind = self.solver.iteration
        self.solver.state['zeta'].set_scales(3/2)
        zeta_ = self.domain.new_field()
        zeta_['g'] = self.true_zeta_grid[ind,:,:]

        # Projections of errors
        proj_zeta_err = self.problem.domain.new_field()
        proj_zeta_err.set_scales(3/2)
        proj_zeta_err['g'] = self.solver.state['zeta']-zeta_

        # denominator of simple update
        del4_ip_temp = self.problem.domain.new_field()
        del4_ip_temp.set_scales(3/2)
        del4_ip_temp[‘g’] = self.solver.state[‘zeta_’].differentiate(x=4) + self.solver.state[‘zeta_’].differentiate(y=4) + 2*self.solver.state[‘zeta_’].differentiate(x=2)*self.solver.state[‘zeta_’].differentiate(y=2)  

        adjust_nu4 = self.mu*de.operators.integrate(proj_zeta_err**2, ‘x’, ‘y’)[‘g’][0,0]/de.operators.integrate(del4_ip_temp*proj_zeta_err, ‘x’, ‘y’)[‘g’][0,0]

        return float(self.problem.parameters['delta_nu4'].args[0]-adjust_nu4))

    def run_simulation(self):

        # Set forcing
        self.problem.parameters['F'].original_args = [1.]
        self.problem.parameters['F'].args = [1.]

        # Set nudging parameters
        self.problem.parameters['nudge'].original_args = []
        self.problem.parameters['nudge'].args = []

        # Set delta_nu4 initially
        self.problem.parameters['delta_nu4'].original_args = [0.]
        self.problem.parameters['delta_nu4'].args = [0.]


        # Setup CFL condition
        cfl = flow_tools.CFL(self.solver, initial_dt=1e-3, cadence=5, safety=.2,
                                 max_change=1.4, min_change=0.2,
                                 max_dt=0.01, min_dt=1e-11)
        cfl.add_velocities(('v',  'w' ))

        while self.solver.ok:

            # Get solver time
            t = self.solver.sim_time

            # Get phases
            phiy = np.pi*np.sin((1.2e-6 * np.pi/3) * self.time_scale*t)
            phix = np.pi*np.sin((1.2e-6) * self.time_scale*t)

            # Set
            x, y = self.domain.grid(0), self.domain.grid(1)
            A = 1e-1
            forcing = A*(np.cos(4*y + phiy) - np.cos(4*x + phix))
            self.problem.parameters['F'].original_args = [forcing]
            self.problem.parameters['F'].args = [forcing]

            # Step
            #dt = cfl.compute_dt()
            if self.solver.iteration == 0:

                # Use initial guesses
                delta_nu4_new = 0.

                elif self.solver.iteration % 100 == 0:

                    # Get update
                    delta_nu4_est = self.new_params()

                    print("New nu4: ", delta_nu4_est + self.nu4)
            

            # Set delta_nu4 initially                                                                                                              
            self.problem.parameters['delta_nu4'].original_args = [delta_nu4_est]
            self.problem.parameters['delta_nu4'].args = [delta_nu4_est]

            self.solver.step(dt)
            if (self.solver.iteration-1) % 100 == 0:
                print(dt)
                print(self.solver.sim_time)

            # Save
            #if self.solver.iteration % 1 == 0:
            #    zeta.set_scales(1)
            #    zeta_list.append(np.copy(zeta['g']))
            #    t_list.append(self.solver.sim_time)
            #print('Completed iteration {}'.format(self.solver.iteration))


        self.merge_results('states')
