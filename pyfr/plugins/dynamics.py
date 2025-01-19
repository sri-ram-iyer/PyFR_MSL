from collections import defaultdict
import numpy as np
import csv
from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.base import BaseSolnPlugin, init_csv

class DynamicsPlugin(BaseSolnPlugin):
    name = 'dynamics'
    systems = ['ac-euler', 'ac-navier-stokes', 'euler', 'navier-stokes']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        comm, rank, root = get_comm_rank_root()

        # Output frequency
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        # Constant variables
        self._constants = self.cfg.items_as('constants', float)

        # Dynamics variables
        self.ffcsv = self.cfg.get(cfgsect, 'fluidforcefile')
        self.scheme = self.cfg.get(cfgsect, 'scheme')
        self.dt = self.cfg.getfloat('solver-time-integrator', 'dt')
        self.solversystem = self.cfg.get('solver', 'system')
        self.inertia = self.cfg.getfloat(cfgsect, 'I')

        # Initialise for Dynamics Schemes
        self.prev_omega_dot = 0.0  # Store omega_dot from the previous timestep
        self.prev_omega = 0.0  # Store omega from from the previous timestep

        # The root rank needs to open the output file
        if rank == root:
            # CSV header
            header = ['time', 'alpha', 'omega', 'omega_dot']
            # Open
            self.outf = init_csv(self.cfg, cfgsect, ','.join(header))

        self.alpha = self._constants.get('alpha')
        self.omega = self._constants.get('omg')

    def __call__(self, intg):
        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Dynamics calculation
        if intg.nacptsteps % self.nsteps == 0:
            # Read the moments of the fluid force file
            if rank == root:
                with open(self.ffcsv, mode='r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    rows = list(csv_reader)
                    last_row = rows[-1]
                    mpz = float(last_row[3])
                    mvz = float(last_row[6])
                mz = mpz + mvz
                self.omega_dot = mz / self.inertia
            
        # Only the root rank writes to the CSV
        if rank == root:
            print(intg.tcurr, self.alpha, self.omega, self.omega_dot, file=self.outf)
            self.outf.flush()

            # Execute the dynamics scheme
            if self.scheme == 'euler':
                # Euler Method
                self.omega_next = self.omega + self.omega_dot * self.dt
                self.alpha_next = self.alpha + self.omega * self.dt
                self.omega = self.omega_next
                self.alpha = self.alpha_next

            elif self.scheme == 'heun':
                # Trapezoidal Method
                self.omega_next = self.omega + (self.dt / 2) * (self.omega_dot + self.prev_omega_dot)
                self.alpha_next = self.alpha + (self.dt / 2) * (self.omega + self.prev_omega)
                self.prev_omega_dot = self.omega_dot
                self.prev_omega = self.omega
                self.omega = self.omega_next
                self.alpha = self.alpha_next

            elif self.scheme == 'adams-bashforth':
                # Adams-Bashforth 2nd-order
                self.omega_next = self.omega + (3 * self.omega_dot - self.prev_omega_dot) * self.dt / 2
                self.alpha_next = self.alpha + (3 * self.omega - self.prev_omega) * self.dt / 2
                self.prev_omega_dot = self.omega_dot
                self.prev_omega = self.omega
                self.omega = self.omega_next
                self.alpha = self.alpha_next