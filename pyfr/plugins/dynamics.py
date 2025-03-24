from collections import defaultdict

import csv
import numpy as np
import re
import math

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.base import BaseSolnPlugin, SurfaceMixin, init_csv


class DynamicsPlugin(SurfaceMixin, BaseSolnPlugin):
    name = 'dynamics'
    systems = ['ac-euler', 'ac-navier-stokes', 'euler', 'navier-stokes']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        comm, rank, root = get_comm_rank_root()

        # Check if we need to compute viscous force
        self._viscous = 'navier-stokes' in intg.system.name

        # Check if the system is incompressible
        self._ac = intg.system.name.startswith('ac')

        # Viscous correction
        self._viscorr = self.cfg.get('solver', 'viscosity-correction', 'none')

        # Constant variables
        self._constants = self.cfg.items_as('constants', float)

        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Boundary to integrate over
        bc = f'bcon_{suffix}_p{intg.rallocs.prank}'

        # Dynamics Variables
        self.fluidforce_steps = self.cfg.getint(cfgsect, 'fluidforce_steps')
        self.output_steps = self.cfg.getint(cfgsect, 'output_steps')
        self.scheme = self.cfg.get(cfgsect, 'scheme')
        self.dof = self.cfg.getint(cfgsect, 'DoF')
        self.dt = self.cfg.getfloat('solver-time-integrator', 'dt')
        self.solversystem = self.cfg.get('solver', 'system')
        self.I_yy = self.cfg.getfloat(cfgsect, 'Iyy')
        self.mass = self.cfg.getfloat(cfgsect, 'm')
        self.rotating_U = self.cfg.getfloat(cfgsect, 'rotating_U')
        self.rotating_V = self.cfg.getfloat(cfgsect, 'rotating_V')
        self.rotating_freestream = np.sqrt(self.rotating_U ** 2 + self.rotating_V ** 2)
        self.theta_deg = np.degrees(np.arctan2(self.rotating_V, self.rotating_U))
        self.omega_rad = 0.0
        self.global_U = self.rotating_freestream
        self.global_V = 0.0
        self.prev_global_U = self.rotating_freestream
        self.prev_global_V = 0.0
        self.global_x_pos = 0.0
        self.global_y_pos = 0.0
        self.global_Velocity = self.rotating_freestream
        self.omega_deg = self.omega_rad * (180 / np.pi)
        self.prev_omega_dot = 0.0
        self.prev_omega = 0.0
        self.global_U_dot = 0.0
        self.global_V_dot = 0.0
        self.prev_U_dot = 0.0
        self.prev_V_dot = 0.0
        self.omega_dot_rad = 0.0
        self.omega_dot_deg = 0.0

        convars = intg.system.elementscls.convarmap[self.ndims]
        self.src_exprs = ["0.0"] * 4
        self.ele_map_items = intg.system.ele_map.items()
        ploc_in_src = False
        soln_in_src = False

        convars = intg.system.elementscls.convarmap[self.ndims]

        for etype, eles in intg.system.ele_map.items():
            eles.add_src_macro('pyfr.plugins.kernels.source', 'source',
                               {'src_exprs': self.src_exprs}, ploc=ploc_in_src,
                               soln=soln_in_src)

        # Moments
        mcomp = 3 if self.ndims == 3 else 1
        self._mcomp = mcomp if self.cfg.hasopt(cfgsect, 'morigin') else 0
        if self._mcomp:
            morigin = np.array(self.cfg.getliteral(cfgsect, 'morigin'))
            if len(morigin) != self.ndims:
                raise ValueError(f'morigin must have {self.ndims} components')

        # Get the mesh and elements
        mesh, self.elemap = intg.system.mesh, intg.system.ele_map

        # See which ranks have the boundary
        bcranks = comm.gather(bc in mesh, root=root)

        # The root rank needs to open the output file
        if rank == root:
            if not any(bcranks):
                raise RuntimeError(f'Boundary {suffix} does not exist')

            # CSV header
            header = ['t', 'px', 'py', 'pz'][:self.ndims + 1]
            if self._mcomp:
                header += ['mpx', 'mpy', 'mpz'][3 - mcomp:]
            if self._viscous:
                header += ['vx', 'vy', 'vz'][:self.ndims]
                if self._mcomp:
                    header += ['mvx', 'mvy', 'mvz'][3 - mcomp:]
            header += ['global_X', 'global_Y', 'global_U', 'global_V', 'global_U_dot', 'global_V_dot', 'alpha', 'gamma', 'theta', 'omega', 'omega_dot']

            # Open
            self.outf = init_csv(self.cfg, cfgsect, ','.join(header))

        # Interpolation matrices and quadrature weights
        self._m0 = m0 = {}
        self._qwts = qwts = defaultdict(list)

        if self._viscous:
            self._m4 = m4 = {}
            rcpjact = {}

        # If we have the boundary then process the interface
        if bc in mesh:
            # Element indices, associated face normals and relative flux
            # points position with respect to the moments origin
            eidxs = defaultdict(list)
            norms = defaultdict(list)
            rfpts = defaultdict(list)

            for etype, eidx, fidx, flags in mesh[bc].tolist():
                eles = self.elemap[etype]
                itype, proj, norm = eles.basis.faces[fidx]

                ppts, pwts = self._surf_quad(itype, proj, flags='s')
                nppts = len(ppts)

                # Get phyical normals
                pnorm = eles.pnorm_at(ppts, [norm]*nppts)[:, eidx]

                eidxs[etype, fidx].append(eidx)
                norms[etype, fidx].append(pnorm)

                if (etype, fidx) not in m0:
                    m0[etype, fidx] = eles.basis.ubasis.nodal_basis_at(ppts)
                    qwts[etype, fidx] = pwts

                if self._viscous and etype not in m4:
                    m4[etype] = eles.basis.m4

                    # Get the smats at the solution points
                    smat = eles.smat_at_np('upts').transpose(2, 0, 1, 3)

                    # Get |J|^-1 at the solution points
                    rcpdjac = eles.rcpdjac_at_np('upts')

                    # Product to give J^-T at the solution points
                    rcpjact[etype] = smat*rcpdjac

                # Get the flux points position of the given face and element
                # indices relative to the moment origin
                if self._mcomp:
                    ploc = eles.ploc_at_np(ppts)[..., eidx]
                    rfpt = ploc - morigin
                    rfpts[etype, fidx].append(rfpt)

            self._eidxs = {k: np.array(v) for k, v in eidxs.items()}
            self._norms = {k: np.array(v) for k, v in norms.items()}
            self._rfpts = {k: np.array(v) for k, v in rfpts.items()}

            if self._viscous:
                self._rcpjact = {k: rcpjact[k[0]][..., v]
                                 for k, v in self._eidxs.items()}

    def __call__(self, intg):
        # MPI info
        comm, rank, root = get_comm_rank_root()
        
        if intg.nacptsteps % self.fluidforce_steps == 0:

            # Solution matrices indexed by element type
            solns = dict(zip(intg.system.ele_types, intg.soln))
            ndims, nvars, mcomp = self.ndims, self.nvars, self._mcomp

            # Force and moment vectors
            fm = np.zeros((2 if self._viscous else 1, ndims + mcomp))

            for etype, fidx in self._m0:
                # Get the interpolation operator
                m0 = self._m0[etype, fidx]
                nfpts, nupts = m0.shape

                # Extract the relevant elements from the solution
                uupts = solns[etype][..., self._eidxs[etype, fidx]]

                # Interpolate to the face
                ufpts = m0 @ uupts.reshape(nupts, -1)
                ufpts = ufpts.reshape(nfpts, nvars, -1)
                ufpts = ufpts.swapaxes(0, 1)

                # Compute the pressure
                pidx = 0 if self._ac else -1

                ele = self.elemap[etype]
                plocupts = ele.ploc_at_np('upts')[..., self._eidxs[etype, fidx]]
                plocfpts = m0 @ plocupts.reshape(nupts, -1)
                plocfpts = plocfpts.reshape(nfpts, ndims, -1)
                plocfpts = plocfpts.swapaxes(1, 2)
                p = self.elementscls.con_to_pri(ufpts, self.cfg, plocfpts)[pidx]

                # Get the quadrature weights and normal vectors
                qwts = self._qwts[etype, fidx]
                norms = self._norms[etype, fidx]

                # Do the quadrature
                fm[0, :ndims] += np.einsum('i...,ij,jik', qwts, p, norms)

                if self._viscous:
                    # Get operator and J^-T matrix
                    m4 = self._m4[etype]
                    rcpjact = self._rcpjact[etype, fidx]

                    # Transformed gradient at solution points
                    tduupts = m4 @ uupts.reshape(nupts, -1)
                    tduupts = tduupts.reshape(ndims, nupts, nvars, -1)

                    # Physical gradient at solution points
                    duupts = np.einsum('ijkl,jkml->ikml', rcpjact, tduupts)
                    duupts = duupts.reshape(ndims, nupts, -1)

                    # Interpolate gradient to flux points
                    dufpts = np.array([m0 @ du for du in duupts])
                    dufpts = dufpts.reshape(ndims, nfpts, nvars, -1)
                    dufpts = dufpts.swapaxes(1, 2)

                    # Viscous stress
                    if self._ac:
                        vis = self.ac_stress_tensor(dufpts)
                    else:
                        vis = self.stress_tensor(ufpts, dufpts)

                    # Do the quadrature
                    fm[1, :ndims] += np.einsum('i...,klij,jil', qwts, vis, norms)

                if self._mcomp:
                    # Get the flux points positions relative to the moment origin
                    rfpts = self._rfpts[etype, fidx]

                    # Do the cross product with the normal vectors
                    rcn = np.atleast_3d(np.cross(rfpts, norms))

                    # Pressure force moments
                    fm[0, ndims:] += np.einsum('i...,ij,jik->k', qwts, p, rcn)

                    if self._viscous:
                        # Normal viscous force at each flux point
                        viscf = np.einsum('ijkl,lkj->lki', vis, norms)

                        # Normal viscous force moments at each flux point
                        rcf = np.atleast_3d(np.cross(rfpts, viscf))

                        # Do the quadrature
                        fm[1, ndims:] += np.einsum('i,jik->k', qwts, rcf)
            
            # Calculate omega_dot
            fm_omg_dot = np.zeros_like(fm)
            comm.Allreduce(fm, fm_omg_dot, op=mpi.SUM)
            self.px = fm_omg_dot[0, 0]
            self.py = fm_omg_dot[0, 1]
            self.mpz = fm_omg_dot[0, 2]
            if self._viscous:
                self.vx = fm_omg_dot[1, 0]
                self.vy = fm_omg_dot[1, 1]
                self.mvz = fm_omg_dot[1, 2]
            else:
                self.vx = 0.0
                self.vy = 0.0
                self.mvz = 0.0
            self.mz = self.mpz + self.mvz
            self.rotating_x_force = self.px + self.vx
            self.rotating_y_force = self.py + self.vy
            self.global_x_force = self.rotating_x_force*np.cos(np.radians(self.theta_deg)) + self.rotating_y_force*np.sin(np.radians(self.theta_deg))
            self.global_y_force = -self.rotating_x_force*np.sin(np.radians(self.theta_deg)) + self.rotating_y_force*np.cos(np.radians(self.theta_deg))
            if self.dof >= 1:
                self.omega_dot_rad = -self.mz / self.I_yy
                self.omega_dot_deg = self.omega_dot_rad * (180 / np.pi)
            if self.dof >= 2:
                self.global_V_dot = self.global_y_force / self.mass
            if self.dof == 3:
                self.global_U_dot = -self.global_x_force / self.mass

        if self.scheme == 'euler':
            # Euler Method
            if self.dof >= 1:
                self.omega_deg = self.omega_deg + self.omega_dot_deg * self.dt
                self.theta_deg = self.theta_deg + self.omega_deg * self.dt
            if self.dof >= 2:
                self.global_V = self.global_V + self.global_V_dot * self.dt
                self.global_y_pos = self.global_y_pos + self.global_V * self.dt
            if self.dof == 3:
                self.global_U = self.global_U + self.global_U_dot * self.dt
            self.global_x_pos = self.global_x_pos + self.global_U * self.dt

        elif self.scheme == 'heun':
            # Trapezoidal Method
            if self.dof >= 1:
                self.omega_deg = self.omega_deg + (self.dt / 2) * (self.omega_dot_deg + self.prev_omega_dot)
                self.theta_deg = self.theta_deg + (self.dt / 2) * (self.omega_deg + self.prev_omega)
                self.prev_omega_dot = self.omega_dot_deg
                self.prev_omega = self.omega_deg
            if self.dof >= 2:
                self.global_V = self.global_V + (self.dt / 2) * (self.global_V_dot + self.prev_V_dot)
                self.global_y_pos = self.global_y_pos + (self.dt / 2) * (self.global_V + self.prev_global_V)
                self.prev_V_dot = self.global_V_dot
                self.prev_global_V = self.global_V
            if self.dof == 3:
                self.global_U = self.global_U + (self.dt / 2) * (self.global_U_dot + self.prev_U_dot)
            self.global_x_pos = self.global_x_pos + (self.dt / 2) * (self.global_U + self.prev_global_U)
            if self.dof == 3:
                self.prev_U_dot = self.global_U_dot
                self.prev_global_U = self.global_U

        elif self.scheme == 'adams-bashforth':
            # Adams-Bashforth 2nd-order
            if self.dof >= 1:
                self.omega = self.omega_deg + (3 * self.omega_dot_deg - self.prev_omega_dot) * self.dt / 2
                self.theta = self.theta_deg + (3 * self.omega_deg - self.prev_omega) * self.dt / 2
                self.prev_omega_dot = self.omega_dot_deg
                self.prev_omega = self.omega_deg
            if self.dof >= 2:
                self.global_V = self.global_V + (3 * self.global_V_dot - self.prev_V_dot) * self.dt / 2
                self.global_y_pos = self.global_y_pos + (3 * self.global_V - self.prev_global_V) * self.dt / 2
                self.prev_V_dot = self.global_V_dot
                self.prev_global_V = self.global_V
            if self.dof == 3:
                self.global_U = self.global_U + (3 * self.global_U_dot - self.prev_U_dot) * self.dt / 2
            self.global_x_pos = self.global_x_pos + (3 * self.global_U - self.prev_global_U) * self.dt / 2
            if self.dof == 3:
                self.prev_U_dot = self.global_U_dot
                self.prev_global_U = self.global_U

        # Calculate velocities and omega_sqr
        self.global_Velocity = np.sqrt(self.global_U ** 2 + self.global_V ** 2)
        self.gamma_deg = np.degrees(np.arctan2(self.global_V, self.global_U))
        self.alpha_deg = self.theta_deg - self.gamma_deg
        self.u = self.global_Velocity * np.cos(np.radians(self.alpha_deg))
        self.v = self.global_Velocity * np.sin(np.radians(self.alpha_deg))
        self.du = self.u - self.rotating_U
        self.dv = self.v - self.rotating_V
        self.omega_rad = self.omega_deg * (np.pi / 180)
        self.omg_sqr_rad = self.omega_rad ** 2
        self.neg_omega_rad = -self.omega_rad
        self.neg_omega_dot_rad = -self.omega_dot_rad
        self.rotating_U_dot = self.global_U_dot*np.cos(np.radians(self.theta_deg)) + self.global_V_dot*np.sin(np.radians(self.theta_deg))
        self.rotating_V_dot = -self.global_U_dot*np.sin(np.radians(self.theta_deg)) + self.global_V_dot*np.cos(np.radians(self.theta_deg))

        # Reduce and output if we're the root rank
        if intg.nacptsteps % self.output_steps == 0:
            if rank == root:
                if self._viscous:
                    print(intg.tcurr, self.px, self.py, self.mpz,
                        self.vx, self.vy, self.mvz,
                        self.global_x_pos, self.global_y_pos,
                        self.global_U, self.global_V,
                        self.global_U_dot, self.global_V_dot,
                        self.alpha_deg, self.gamma_deg, self.theta_deg,
                        self.omega_deg, self.omega_dot_deg,
                        sep=',', file=self.outf)
                else:
                    print(intg.tcurr, self.px, self.py, self.mpz,
                        self.global_x_pos, self.global_y_pos,
                        self.global_U, self.global_V,
                        self.global_U_dot, self.global_V_dot,
                        self.alpha_deg, self.gamma_deg, self.theta_deg,
                        self.omega_deg, self.omega_dot_deg,
                        sep=',', file=self.outf)

                self.outf.flush()

        # Broadcast to solver
        if rank == root:
            intg.system.u = float(comm.bcast(self.du, root=root))
            intg.system.v = float(comm.bcast(self.dv, root=root))
            intg.system.omg_sqr = float(comm.bcast(self.omg_sqr_rad, root=root))
            intg.system.neg_omg = float(comm.bcast(self.neg_omega_rad, root=root))
            intg.system.omega_dot = float(comm.bcast(self.neg_omega_dot_rad, root=root))
            intg.system.global_U = float(comm.bcast(self.global_U, root=root))
            intg.system.global_V = float(comm.bcast(self.global_V, root=root))
            intg.system.rotating_U_dot = float(comm.bcast(self.rotating_U_dot, root=root))
            intg.system.rotating_V_dot = float(comm.bcast(self.rotating_V_dot, root=root))
        else:
            intg.system.u = float(comm.bcast(None, root=root))
            intg.system.v = float(comm.bcast(None, root=root))
            intg.system.omg_sqr = float(comm.bcast(None, root=root))
            intg.system.neg_omg = float(comm.bcast(None, root=root))
            intg.system.omega_dot = float(comm.bcast(None, root=root))
            intg.system.global_U = float(comm.bcast(None, root=root))
            intg.system.global_V = float(comm.bcast(None, root=root))
            intg.system.rotating_U_dot = float(comm.bcast(None, root=root))
            intg.system.rotating_V_dot = float(comm.bcast(None, root=root))

    def stress_tensor(self, u, du):
        c = self._constants

        # Density, energy
        rho, E = u[0], u[-1]

        # Gradient of density and momentum
        gradrho, gradrhou = du[:, 0], du[:, 1:-1]

        # Gradient of velocity
        gradu = (gradrhou - gradrho[:, None]*u[None, 1:-1]/rho) / rho

        # Bulk tensor
        bulk = np.eye(self.ndims)[:, :, None, None]*np.trace(gradu)

        # Viscosity
        mu = c['mu']

        if self._viscorr == 'sutherland':
            cpT = c['gamma']*(E/rho - 0.5*np.sum(u[1:-1]**2, axis=0)/rho**2)
            Trat = cpT/c['cpTref']
            mu *= (c['cpTref'] + c['cpTs'])*Trat**1.5 / (cpT + c['cpTs'])

        return -mu*(gradu + gradu.swapaxes(0, 1) - 2/3*bulk)

    def ac_stress_tensor(self, du):
        # Gradient of velocity and kinematic viscosity
        gradu, nu = du[:, 1:], self._constants['nu']

        return -nu*(gradu + gradu.swapaxes(0, 1))
