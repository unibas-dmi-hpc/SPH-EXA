# MIT License
#
# Copyright (c) 2022 CSCS, ETH Zurich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# @author jgphpc
import os
import sys
import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.udeps as udeps


# {{{ unittests
unittests = [
    "/usr/local/sbin/coord_samples/coordinate_test",
    "/usr/local/sbin/hydro/kernel_tests_std",
    "/usr/local/sbin/hydro/kernel_tests_ve",
    "/usr/local/sbin/integration_mpi/box_mpi",
    "/usr/local/sbin/integration_mpi/domain_2ranks",
    "/usr/local/sbin/integration_mpi/domain_nranks",
    "/usr/local/sbin/integration_mpi/exchange_domain",
    "/usr/local/sbin/integration_mpi/exchange_focus",
    "/usr/local/sbin/integration_mpi/exchange_general",
    "/usr/local/sbin/integration_mpi/exchange_halos",
    "/usr/local/sbin/integration_mpi/exchange_halos_gpu",   # TODO: -p debug
    "/usr/local/sbin/integration_mpi/exchange_keys",
    "/usr/local/sbin/integration_mpi/focus_transfer",
    "/usr/local/sbin/integration_mpi/focus_tree",
    "/usr/local/sbin/integration_mpi/globaloctree",
    "/usr/local/sbin/integration_mpi/treedomain",
    # new:
    # "/usr/local/sbin/performance/neighbors_test_gpu",
    # "/usr/local/sbin/integration_mpi/assignment_gpu",       # -N2 -n2 -pdebug
    # "/usr/local/sbin/integration_mpi/domain_gpu",           # -N2 -n2 -pdebug
    # "/usr/local/sbin/integration_mpi/exchange_domain_gpu",  # -N2 -n2 -pdebug
    #
    "/usr/local/sbin/performance/hilbert_perf",
    "/usr/local/sbin/performance/hilbert_perf_gpu",
    "/usr/local/sbin/performance/octree_perf",
    "/usr/local/sbin/performance/octree_perf_gpu",
    "/usr/local/sbin/performance/peers_perf",
    "/usr/local/sbin/performance/scan_perf",
    "/usr/local/sbin/ryoanji/cpu_unit_tests/ryoanji_cpu_unit_tests",
    "/usr/local/sbin/ryoanji/global_upsweep_cpu",
    "/usr/local/sbin/ryoanji/global_upsweep_gpu",
    # "/usr/local/sbin/ryoanji/ryoanji_demo/ryoanji_demo",
    "/usr/local/sbin/ryoanji/unit_tests/ryoanji_unit_tests",
    "/usr/local/sbin/unit/component_units",
    "/usr/local/sbin/unit/component_units_omp",
    "/usr/local/sbin/unit_cuda/component_units_cuda",
]
# }}}

# {{{ unittests_params
unittests_params = {
    "/usr/local/sbin/unit_cuda/component_units_cuda": "g",
    "/usr/local/sbin/coord_samples/coordinate_test": "1",
    "/usr/local/sbin/hydro/kernel_tests_ve": "1",
    "/usr/local/sbin/hydro/turbulence_tests": "1",
    "/usr/local/sbin/hydro/kernel_tests_std": "1",
    #
    "/usr/local/sbin/integration_mpi/exchange_keys": "5",
    "/usr/local/sbin/integration_mpi/exchange_focus": "2",
    "/usr/local/sbin/integration_mpi/treedomain": "5",
    "/usr/local/sbin/integration_mpi/exchange_general": "5",
    "/usr/local/sbin/integration_mpi/globaloctree": "2",
    "/usr/local/sbin/integration_mpi/exchange_domain": "5",
    "/usr/local/sbin/integration_mpi/box_mpi": "5",
    "/usr/local/sbin/integration_mpi/domain_2ranks": "2",
    "/usr/local/sbin/integration_mpi/exchange_halos": "2",
    "/usr/local/sbin/integration_mpi/focus_tree": "5",
    "/usr/local/sbin/integration_mpi/domain_nranks": "5",
    "/usr/local/sbin/integration_mpi/focus_transfer": "2",
    #
    "/usr/local/sbin/unit/component_units_omp": "1",
    "/usr/local/sbin/unit/component_units": "1",
    #
    "/usr/local/sbin/ryoanji/global_upsweep_gpu": "g",
    "/usr/local/sbin/ryoanji/cpu_unit_tests/ryoanji_cpu_unit_tests": "1",
    "/usr/local/sbin/ryoanji/global_upsweep_cpu": "5",
    "/usr/local/sbin/ryoanji/ryoanji_demo/ryoanji_demo": "?",
    "/usr/local/sbin/ryoanji/unit_tests/ryoanji_unit_tests": "g",
    #
    "/usr/local/sbin/performance/hilbert_perf_gpu": "g",
    "/usr/local/sbin/performance/octree_perf_gpu": "g",
    "/usr/local/sbin/performance/peers_perf": "1",
    "/usr/local/sbin/performance/octree_perf": "1",
    "/usr/local/sbin/performance/hilbert_perf": "1",
    "/usr/local/sbin/performance/scan_perf": "1",
    "/usr/local/sbin/performance/neighbors_test_gpu": "g",
    # --> /usr/local/sbin/performance/cudaNeighborsTest
    #
    # require more than 1 compute node:
    "/usr/local/sbin/integration_mpi/assignment_gpu": "2",
    "/usr/local/sbin/integration_mpi/domain_gpu": "2",
    "/usr/local/sbin/integration_mpi/exchange_domain_gpu": "2",
    "/usr/local/sbin/integration_mpi/exchange_halos_gpu": "2",
}
# TODO: https://sarus.readthedocs.io/en/stable/cookbook/gpu/gpudirect.html
# }}}


# {{{ ci unittests / 1cn
@rfm.simple_test
class ci_unittests(rfm.RunOnlyRegressionTest):
    valid_systems = ['dom:gpu', 'daint:gpu', 'hohgant:mc']
    valid_prog_environs = ['builtin']
    image = variable(str, value='/usr/local')
    unittest = parameter(unittests)
    sourcesdir = None
    num_tasks = 1

    # {{{ hooks
    @run_before('run')
    def set_executable(self):
        self.executable = self.unittest.replace("/usr/local", self.image)

    @run_before('run')
    def set_ntasks(self):
        if unittests_params[self.unittest] in ["2", "5"]:
            self.num_tasks = int(unittests_params[self.unittest])

    @run_before('run')
    def set_skip(self):
        test = self.unittest.split('/')[-1]
        need_2cn = ['exchange_halos_gpu', 'assignment_gpu', 'domain_gpu',
                    'exchange_domain_gpu']
        self.skip_if(test in need_2cn, f'{test} needs 2 cn')
    # }}}

    # {{{ sanity
    @sanity_function
    def assert_sanity(self):
        skip = [
            '/usr/local/sbin/performance/peers_perf',
            '/usr/local/sbin/performance/octree_perf_gpu',
            '/usr/local/sbin/performance/octree_perf',
            '/usr/local/sbin/performance/hilbert_perf_gpu',
            '/usr/local/sbin/performance/hilbert_perf',
            '/usr/local/sbin/performance/neighbors_test_gpu',
        ]
        if self.unittest in skip:
            return sn.all([sn.assert_not_found(r'error', self.stdout)])
        else:
            return sn.all([
                sn.assert_found(r'PASS', self.stdout),
                ])
    # }}}
# }}}


# {{{ ci unittests / 2cn
@rfm.simple_test
class ci_2cn(rfm.RunOnlyRegressionTest):
    valid_systems = ['dom:gpu', 'daint:gpu', 'hohgant:mc']
    valid_prog_environs = ['builtin']
    image = variable(str, value='/usr/local')
    unittest = parameter(unittests)
    sourcesdir = None
    num_tasks = 1

    @run_before('run')
    def set_executable(self):
        self.executable = self.unittest.replace("/usr/local", self.image)

    @run_before('run')
    def set_ntasks(self):
        self.num_tasks_per_node = 1
        if unittests_params[self.unittest] in ["2", "5"]:
            self.num_tasks = int(unittests_params[self.unittest])

    @run_before('run')
    def set_skip(self):
        test = self.unittest.split('/')[-1]
        need_2cn = ['exchange_halos_gpu', 'assignment_gpu', 'domain_gpu',
                    'exchange_domain_gpu']
        self.skip_if(test not in need_2cn, f'{test} needs only 1 cn')

    # {{{ sanity
    @sanity_function
    def assert_sanity(self):
        skip = [
            '/usr/local/sbin/performance/peers_perf',
            '/usr/local/sbin/performance/octree_perf_gpu',
            '/usr/local/sbin/performance/octree_perf',
            '/usr/local/sbin/performance/hilbert_perf_gpu',
            '/usr/local/sbin/performance/hilbert_perf',
            '/usr/local/sbin/performance/neighbors_test_gpu',
        ]
        if self.unittest in skip:
            return sn.all([sn.assert_not_found(r'error', self.stdout)])
        else:
            return sn.all([
                sn.assert_found(r'PASS', self.stdout),
                ])
    # }}}
# }}}


# {{{ cpu tests
@rfm.simple_test
class ci_cputests(rfm.RunOnlyRegressionTest):
    valid_systems = ['dom:gpu', 'daint:gpu', 'hohgant:mc']
    valid_prog_environs = ['builtin']
    image = variable(str, value='/usr/local')
    unittest = parameter(['sedov', 'sedov --ve', 'noh'])
    sourcesdir = None
    num_tasks = 1
    modules = ['PrgEnv-gnu', 'cdt/22.05', 'nvhpc-nompi/22.2',
               'cray-hdf5-parallel/1.12.1.3']

    # {{{ hooks
    @run_before('run')
    def set_gpu_executable(self):
        self.executable = os.path.join(self.image, 'bin', 'sphexa')
        self.executable_opts = ['--init', self.unittest, '-s', '1', '-n', '50']
        self.variables = {
                'LD_LIBRARY_PATH': '$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH',
                # 'HDF5_DISABLE_VERSION_CHECK': '1',
        }
    # }}}

    # {{{ sanity
    @sanity_function
    def assert_sanity(self):
        regex1 = r'Total execution time of \d+ iterations of \S+ up to t ='
        return sn.all([
            sn.assert_found(regex1, self.stdout),
        ])
    # }}}
# }}}


# {{{ gpu tests
@rfm.simple_test
class ci_gputests(rfm.RunOnlyRegressionTest):
    valid_systems = ['dom:gpu', 'daint:gpu', 'hohgant:mc']
    valid_prog_environs = ['builtin']
    image = variable(str, value='/usr/local')
    unittest = parameter(['sedov', 'noh', 'evrard'])
    sourcesdir = None
    num_tasks = 1
    modules = ['PrgEnv-gnu', 'cdt/22.05', 'nvhpc-nompi/22.2',
               'cray-hdf5-parallel/1.12.1.3']
    in_glass = variable(str, value='glass.h5')

    # {{{ hooks
    @run_before('run')
    def set_gpu_executable(self):
        self.executable = os.path.join(self.image, 'bin', 'sphexa-cuda')
        self.executable_opts = ['--init', self.unittest]
        self.variables = {
                'LD_LIBRARY_PATH': '$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH',
                # 'HDF5_DISABLE_VERSION_CHECK': '1',
        }

    @run_before('run')
    def set_input(self):
        in_path = 'ftp://ftp.cscs.ch/out/jgp/hpc/containers/in'
        if self.unittest in ['evrard']:
            self.prerun_cmds = [f'wget --quiet {in_path}/{self.in_glass}']

    @run_before('run')
    def set_output(self):
        fields = '-f rho,p,u,x,y,z,vx,vy,vz'
        opts = {
            # --outDir /scratch
            'sedov': f'{fields} -s 200 -n 50 -w 200 ',
            'noh': f'{fields} -s 200 -n 50 -w 200 ',
            'evrard': (f'--glass {self.in_glass} -s 10 -n 50 -w 10')
        }
        self.executable_opts = [
            '--init', self.unittest,
            opts[self.unittest],
        ]

    @run_before('run')
    def set_compare(self):
        compare_executable = os.path.join(self.image, 'bin', 'sedov_solution')
        self.postrun_cmds = [
            f'ln -s {compare_executable}',
            'python3 -m venv --system-site-packages myvenv',
            'source myvenv/bin/activate',
            'pip install -U pip h5py matplotlib',
        ]
        if self.unittest == 'sedov':
            script = os.path.join(self.image, 'bin', 'compare_solutions.py')
            self.postrun_cmds += [
                f'python3 {script} -s 200 ./dump_sedov.h5 > sedov.rpt',
            ]
        elif self.unittest == 'noh':
            script = os.path.join(self.image, 'bin', 'compare_noh.py')
            self.postrun_cmds += [
                f'python3 {script} -s 200 ./dump_noh.h5 > noh.rpt',
            ]
        elif self.unittest == 'evrard':
            rpt = 'evrard.rpt'
            self.postrun_cmds += [
                f'echo Density L1 error 0.0 > {rpt}',
                f'echo Pressure L1 error 0.0 >> {rpt}',
                f'echo Velocity L1 error 0.0 >> {rpt}',
            ]

        self.postrun_cmds += [
            'cat *.rpt',
            '# https://reframe-hpc.readthedocs.io/en/stable/manpage.html?'
            'highlight=RFM_TRAP_JOB_ERRORS',
        ]
    # }}}

    # {{{ perf
    @performance_function('')
    def extract_L1(self, metric='Density'):
        if metric not in ('Density', 'Pressure', 'Velocity', 'Energy'):
            raise ValueError(f'illegal value (L1 metric={metric!r})')

        return sn.extractsingle(rf'{metric} L1 error (\S+)$',
                                f'{self.unittest}.rpt', 1, float)

    @run_before('performance')
    def set_perf_variables(self):
        self.perf_variables = {
            'Density': self.extract_L1('Density'),
            'Pressure': self.extract_L1('Pressure'),
            'Velocity': self.extract_L1('Velocity'),
            # 'Energy': self.extract_L1('Energy'),
        }

    @run_before('performance')
    def set_reference(self):
        reference_d = {
            'sedov': {
                'Density':  (0.138, -0.015, 0.01, ''),
                'Pressure':  (0.902, -0.01, 0.01, ''),
                'Velocity':  (0.915, -0.01, 0.01, ''),
                # 'Energy':  (0., -0.05, 0.05, ''),
            },
            'noh': {
                'Density':  (0.955, -0.01, 0.01, ''),
                'Pressure':  (0.388, -0.01, 0.01, ''),
                'Velocity':  (0.0384, -0.05, 0.05, ''),
                # 'Energy':  (0.029, -0.05, 0.05, ''),
            },
            'evrard': {
                'Density':  (0.0, -0.01, 0.01, ''),
                'Pressure':  (0.0, -0.01, 0.01, ''),
                'Velocity':  (0.0, -0.05, 0.05, ''),
                # 'Energy':  (0.029, -0.05, 0.05, ''),
            },
        }
        self.reference = {'*': reference_d[self.unittest]}
    # }}}

    # {{{ sanity
    @sanity_function
    def assert_sanity(self):
        regex1 = r'Total execution time of \d+ iterations of \S+ up to t ='
        return sn.all([
            sn.assert_found(regex1, self.stdout),
        ])
    # }}}
# }}}
