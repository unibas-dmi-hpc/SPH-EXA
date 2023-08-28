import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class analytical_solution(rfm.RunOnlyRegressionTest):
    test = parameter(['sedov', 'noh'])  # noh
    rpt_path = variable(str, value='.')
    valid_systems = ['*']
    valid_prog_environs = ['*']

    @run_before('run')
    def check_file(self):
        self.executable = 'file'
        self.executable_opts = [f'dump_{self.test}.h5']

    @sanity_function
    def assert_hello(self):
        return sn.assert_found(r'Loaded \d+ particles',
                               f'{self.rpt_path}/{self.test}.rpt')

    @performance_function('')
    def extract_L1(self, metric='Density'):
        if metric not in ('Density', 'Pressure', 'Velocity', 'Energy'):
            raise ValueError(f'illegal value (L1 metric={metric!r})')

        return sn.extractsingle(rf'{metric} L1 error (\S+)$',
                                f'{self.rpt_path}/{self.test}.rpt', 1, float)

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
                'Density':  (0.229, -0.015, 0.01, ''),
                'Pressure':  (0.967, -0.01, 0.01, ''),
                'Velocity':  (0.977, -0.01, 0.01, ''),
                # 'Energy':  (0., -0.05, 0.05, ''),
            },
            'noh': {
                'Density':  (10.42, -0.01, 0.01, ''),
                'Pressure':  (2.88, -0.01, 0.01, ''),
                'Velocity':  (0.14, -0.05, 0.05, ''),
                # 'Energy':  (0.029, -0.05, 0.05, ''),
            },
        }
        self.reference = {'*': reference_d[self.test]}
