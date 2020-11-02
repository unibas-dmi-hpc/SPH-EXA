# jgphpc (CSCS)
import reframe as rfm
import reframe.utility.sanity as sn

commits = ['f982fde']
testnames = ['sedov']
gpu_cap = 'sm_60'
# tc_ver = '20.08'


# {{{ build base
@rfm.simple_test
class Base_Build_Test(rfm.CompileOnlyRegressionTest):
    # def __init__(self):
    def __init__(self):
        self.maintainers = ['JG']
        self.prebuild_cmds += [
                'git log --pretty=oneline -n1',
                'module rm xalt', 'module list -t']
        self.sourcesdir = 'src_gpu'
        self.build_system = 'Make'
        self.build_system.makefile = 'Makefile'
        self.build_system.cxx = 'CC'
        self.build_system.nvcc = 'nvcc'
        self.build_system.max_concurrency = 2
        self.prgenv_flags = {
            # The makefile adds -DUSE_MPI
            'PrgEnv-gnu': ['-I.', '-I./include', '-std=c++14', '-g', '-O3',
                           '-w', '-DUSE_MPI', '-DNDEBUG', '-fopenmp'],
            'PrgEnv-intel': ['-I.', '-I./include', '-std=c++14', '-g', '-O3',
                             '-DUSE_MPI', '-DNDEBUG', '-qopenmp'],
            'PrgEnv-cray': ['-I.', '-I./include', '-std=c++17', '-g', '-Ofast',
                            '-DUSE_MPI', '-DNDEBUG', '-fopenmp'],
            # -fopenmp[=libcraymp] (Cray runtime - default)
            # -fopenmp=libomp (Clang runtime)
            'PrgEnv-pgi': ['-I.', '-I./include', '-std=c++14', '-g', '-O3',
                           '-DUSE_MPI', '-DNDEBUG', '-mp'],
        }
        sed_ifdef = (r'"s-#include \"cuda/sph.cuh\"-#ifdef USE_CUDA\n'
                     r'#include \"cuda/sph.cuh\"\n#endif-"')
        # first this:
        self.prebuild_cmds = [
            f'sed -i {sed_ifdef} include/sph/findNeighbors.hpp',
            f'sed -i {sed_ifdef} include/sph/density.hpp',
            f'sed -i {sed_ifdef} include/sph/IAD.hpp',
            f'sed -i {sed_ifdef} include/sph/momentumAndEnergyIAD.hpp',
        ]
        # TODO: fullpath = f'{self.target_executable}.{self.testname}...
        self.sanity_patterns = sn.assert_not_found(r'warning', self.stdout)

    # {{{ hooks
    @rfm.run_before('compile')
    def setflags(self):
        # self.build_system.cxxflags = \
        #     self.prgenv_flags[self.current_environ.name]
        # self.modules += self.tool_modules[self.current_environ.name]
        flags = (' '.join(map(str,
                              self.prgenv_flags[self.current_environ.name])))
        self.build_system.options += [
            self.target_executable, f'MPICXX={self.build_system.cxx}',
            'SRCDIR=.', 'BUILDDIR=.', 'BINDIR=.', f'CXXFLAGS="{flags}"',
            'CUDA_PATH=$CUDATOOLKIT_HOME',
            f'TESTCASE={self.executable}',
        ]
# }}}
# }}}


# {{{ mpi+omp
@rfm.parameterized_test(*[[commit, testname]
                          for commit in commits
                          for testname in testnames])
class MPIOMP_Build_Test(Base_Build_Test):
    def __init__(self, commit, testname):
        super().__init__()
        self.commit = commit
        self.testname = testname
        self.descr = f'Build {testname} test ({commit}) with MPI+OpenMP'
        self.valid_systems = ['*']
        self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-cray', 'PrgEnv-pgi',
                                    'PrgEnv-intel']
        self.tags = {'sph', 'cpu'}
        self.prebuild_cmds += [
                f'git checkout {commit}',
                'git log --pretty=oneline -n1',
                'module rm xalt', 'module list -t']
        self.executable = testname
        self.target_executable = 'mpi+omp'
        fullpath = f'{self.target_executable}.{testname}.{commit}.$PE_ENV'
        self.postbuild_cmds = [
            f'cp {self.target_executable}.app $SCRATCH/{fullpath}',
        ]
# }}}


# {{{ Cuda
@rfm.parameterized_test(*[[commit, testname]
                          for commit in commits
                          for testname in testnames])
class CUDA_Build_Test(Base_Build_Test):
    def __init__(self, commit, testname):
        super().__init__()
        self.descr = f'Build {testname} test ({commit}) with CUDA'
        self.valid_systems = ['*']
        self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-cray', 'PrgEnv-pgi',
                                    'PrgEnv-intel']
        self.tags = {'sph', 'gpu'}
        self.prebuild_cmds += [
                f'git checkout {commit}',
                'git log --pretty=oneline -n1',
                'module rm xalt', 'module list -t']
        self.modules = ['craype-accel-nvidia60']
        self.executable = testname
        self.target_executable = 'mpi+omp+cuda'
        fullpath = f'{self.target_executable}.{testname}.{commit}.$PE_ENV'
        self.postbuild_cmds = [
            f'cp {self.target_executable}.app $SCRATCH/{fullpath}',
        ]
        # self.variables = {'CUDA_PATH': '$CUDATOOLKIT_HOME'}
        self.build_system.options = [
            f'NVCCFLAGS="-std=c++14 --expt-relaxed-constexpr -arch={gpu_cap}"',
            # --ptxas-options=-v -g -G"',
            f'NVCCLDFLAGS="-arch={gpu_cap}"',
        ]
# }}}


# {{{ OpenACC
@rfm.parameterized_test(*[[commit, testname]
                          for commit in commits
                          for testname in testnames])
class OPENACC_Build_Test(Base_Build_Test):
    def __init__(self, commit, testname):
        super().__init__()
        self.descr = f'Build {testname} test ({commit}) with OpenACC'
        self.valid_systems = ['*']
        self.valid_prog_environs = ['PrgEnv-pgi']
        self.tags = {'sph', 'gpu'}
        self.prebuild_cmds += [
                f'git checkout {commit}',
                'git log --pretty=oneline -n1',
                'module rm xalt', 'module list -t']
        self.modules = ['craype-accel-nvidia60']
        self.executable = testname
        self.target_executable = 'mpi+omp+acc'
        atomic_flag = ''
        fullpath = f'{self.target_executable}.{testname}.{commit}.$PE_ENV'
        self.postbuild_cmds = [
            f'cp {self.target_executable}.app $SCRATCH/{fullpath}',
        ]
        openacc_flag = '-acc -ta=tesla,cc60 -Minfo=accel '  # {atomic_flag}
        self.build_system.options = [f'LIB="{openacc_flag}"']
# }}}


# {{{ OpenMP Offload
@rfm.parameterized_test(*[[commit, testname]
                          for commit in commits
                          for testname in testnames])
class OPENMPgpu_Build_Test(Base_Build_Test):
    def __init__(self, commit, testname):
        super().__init__()
        self.descr = f'Build {testname} test ({commit}) with OpenMP Offloading'
        self.valid_systems = ['*']
        self.valid_prog_environs = ['PrgEnv-cray']
        self.tags = {'sph', 'gpu'}
        self.prebuild_cmds += [
                f'git checkout {commit}',
                'git log --pretty=oneline -n1',
                'module rm xalt', 'module list -t']
        self.modules = ['craype-accel-nvidia60']
        self.executable = testname
        self.target_executable = 'mpi+omp+target'
        fullpath = f'{self.target_executable}.{testname}.{commit}.$PE_ENV'
        self.postbuild_cmds = [
            f'cp {self.target_executable}.app $SCRATCH/{fullpath}',
        ]
        offload_flag = (r'-fopenmp-targets=nvptx64 -Xopenmp-target '
                        f'-march={gpu_cap}')
        self.build_system.options = [f'LIB="{offload_flag}"']
# }}}
