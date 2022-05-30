import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class analytical_solution(rfm.RunOnlyRegressionTest):
    test = parameter(['sedov', 'noh'])  # noh
    valid_systems = ['*']
    valid_prog_environs = ['*']

    @run_before('run')
    def check_file(self):
        self.executable = 'file'
        self.executable_opts = [f'dump_{self.test}.h5part']

    @sanity_function
    def assert_hello(self):
        return sn.assert_found(r'Loaded \d+ particles', f'{self.test}.rpt')

    @performance_function('')
    def extract_L1(self, metric='Density'):
        if metric not in ('Density', 'Pressure', 'Velocity', 'Energy'):
            raise ValueError(f'illegal value (L1 metric={metric!r})')

        return sn.extractsingle(rf'{metric} L1 error (\S+)$',
                                f'{self.test}.rpt', 1, float)

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
                'Density':  (0.138, -0.01, 0.01, ''),
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
        }
        self.reference = {'*': reference_d[self.test]}
