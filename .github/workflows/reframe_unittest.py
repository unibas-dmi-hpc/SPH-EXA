import os, sys
import reframe as rfm
import reframe.utility.sanity as sn

@rfm.simple_test
class SPHEXA_Unit_Test(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = ('RunOnlyRegressionTest')
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.sourcesdir = None
        self.executable = '$MYEXE'
        self.sanity_patterns = sn.assert_found('PASSED', self.stdout)
