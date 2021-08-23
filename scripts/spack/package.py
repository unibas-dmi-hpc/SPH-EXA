# MIT License
#
# Copyright (c) 2021 CSCS, ETH Zurich
#               2021 University of Basel
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
# ----------------------------------------------------------------------------

from spack import *


class SphExaMiniApp(CMakePackage, CudaPackage):
    """
    The SPH-EXA mini-app is a scientific research code; its goal is to scale
    the Smoothed Particle Hydrodynamics (SPH) method to enable Tier-0 and
    Exascale simulations: https://hpc.dmi.unibas.ch/en/research/sph-exa/
    Example: spack install sph-exa-mini-app+cuda %gcc@9.3.0.21.05
    """
    homepage = "https://github.com/unibas-dmi-hpc/SPH-EXA_mini-app"
    git = "https://github.com/unibas-dmi-hpc/SPH-EXA_mini-app.git"
    maintainers = ['jgphpc', 'sebkelle1']
    version('develop', branch='develop', submodules=True, preferred=True)
    depends_on('cmake@3.20.2:', type='build')
    depends_on('mpi')
    depends_on('cuda@10:', when='+cuda')
    variant('mpi', default=True, description='Enable MPI support')
    variant('cuda', default=False, description='Enable CUDA support')
    # patch: temp. wkaround for "CollisionList not declared" build error
    patch(
        'collisions.patch',
        sha256=('ba14f265ce0e03ad4909503797e250e53f4a95a46c2da1b422787b5cb303'
                '4b90')
    )
    sanity_check_is_file = [join_path('bin', 'sedov')]

    def cmake_args(self):
        # Add arguments other than CMAKE_INSTALL_PREFIX and CMAKE_BUILD_TYPE
        args = ['-DCMAKE_VERBOSE_MAKEFILE=ON']
        args.append(f'-DCMAKE_CXX_COMPILER={self.compiler.cxx}')
        # -DUSE_CUDA not needed, keeping as reminder:
        # if '+cuda' in self.spec:
        #     args.append('-DUSE_CUDA')
        if '+cuda' in self.spec:
            sanity_check_is_file = [join_path('bin', 'sedov-cuda')]

        return args
