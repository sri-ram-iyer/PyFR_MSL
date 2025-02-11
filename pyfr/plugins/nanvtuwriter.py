import numpy as np
from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BaseSolnPlugin, PostactionMixin, RegionMixin
from pyfr.writers.native import NativeWriter
from mpi4py import MPI

class NaNVTUWriterPlugin(PostactionMixin, RegionMixin, BaseSolnPlugin):
    name = 'nanvtuwriter'
    systems = ['*']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Frequency at which this plugin is invoked
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')
        # Threshold count of NaNs needed to trigger VTU output
        self.nan_threshold = self.cfg.getint(cfgsect, 'nan_threshold')

        # Set up VTU writer parameters
        basedir = self.cfg.getpath(cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(cfgsect, 'basename')
        self._writer = NativeWriter(intg, basedir, basename, 'soln')
        self.tout_last = intg.tcurr

        # Get the field names and data type from the integrator
        self.fields = intg.system.elementscls.convarmap[self.ndims]
        self.fpdtype = intg.backend.fpdtype

        print("NaN VTU Sucessfully Initialized")

    def __call__(self, intg):
        # Only perform this check every nsteps
        if intg.nacptsteps % self.nsteps != 0:
            return

        # Count the total number of NaNs on this rank
        total_nan_local = sum(np.isnan(s).sum() for s in intg.soln)

        # Perform an MPI reduction to calculate the global sum of NaNs
        comm, rank, root = get_comm_rank_root()
        total_nan_global = comm.reduce(total_nan_local, op=MPI.SUM, root=root)

        # Root rank determines if the global NaN count exceeds the threshold
        if rank == root:
            should_trigger = total_nan_global >= self.nan_threshold
        else:
            should_trigger = None

        # Broadcast the decision to all ranks
        should_trigger = comm.bcast(should_trigger, root=root)

        if not should_trigger:
            # No ranks exceed the threshold; do nothing
            return

        print(f"Rank {rank}: NaN VTU Triggered (Total NaNs: {total_nan_global})")

        # All ranks execute the VTU writing logic
        # Create metadata
        stats = Inifile()
        stats.set('data', 'fields', ','.join(self.fields))
        stats.set('data', 'prefix', 'soln')
        intg.collect_stats(stats)

        # If we are the root rank then prepare the metadata
        if rank == root:
            metadata = dict(intg.cfgmeta,
                            stats=stats.tostr(),
                            mesh_uuid=intg.mesh_uuid)
        else:
            metadata = None

        # Fetch data from other plugins and add it to metadata with ad-hoc keys
        for csh in intg.plugins:
            try:
                prefix = intg.get_plugin_data_prefix(csh.name, csh.suffix)
                pdata = csh.serialise(intg)
            except AttributeError:
                pdata = {}

            if rank == root:
                metadata |= {f'{prefix}/{k}': v for k, v in pdata.items()}

        # Subset the solution for each element region
        data = dict(self._ele_region_data)
        for idx, etype, rgn in self._ele_regions:
            data[etype] = intg.soln[idx][..., rgn].astype(self.fpdtype)

        # Write out the VTU file
        print(f"Rank {rank}: Writing VTU File")
        solnfname = self._writer.write(data, intg.tcurr, metadata)
        print(f"Rank {rank}: VTU File Written ({solnfname})")

        # Optionally, run any post-actions (such as notifying external processes)
        self._invoke_postaction(intg=intg, mesh=intg.system.mesh.fname,
                                soln=solnfname, t=intg.tcurr)

        # Update the last output time
        self.tout_last = intg.tcurr

        comm.Barrier()

        if rank == root:
            raise RuntimeError(f'NaNs detected and VTU written at t = {intg.tcurr}')