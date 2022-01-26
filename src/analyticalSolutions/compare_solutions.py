'''
    MIT License
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
'''

'''
    Command line utility for compare analytical solutions of some SPH-EXA test simulations.
    
    Usage examples:
        $ python ./compare_solutions.py --help'
        $ python ./compare_solutions.py --version'
        $ python ./compare_solutions.py sedov --help'
        $ python ./compare_solutions.py sedov --time 0.018458 --snapshot_file ./../../dump_sedov100.txt
        $ python ./compare_solutions.py sedov --constants_file ./../../constants.txt --iteration 200 --snapshot_file ./../../dump_sedov200.txt
'''

__program__ = "compare_solutions.py"
__author__  = "Jose A. Escartin (ja.escartin@gmail.com)"
__version__ = '0.1.0'

import os.path
import click

@click.group(invoke_without_command=True, context_settings=dict(help_option_names=["-h", "--help"]))
@click.option('--version', '-v', is_flag=True, help="Show version number of compare_solutions.py")
def cli(version):
    
    """The compare_solutions.py CLI can be used to compare the some SPH-EXA simulations tests with their analytical solutions."""

    if version:
        click.echo("\ncompare_solutions.py version: " + __version__ + "\n")

default_timestep  = 0.
default_constants = "./constants.txt"
default_iteration = 0
default_snapshot  = "./dump_sedov0.txt"

@cli.command()
@click.option('-t',  '--time',           required=False, default=default_timestep,  help='Simulation time. Default: ['                + default_timestep.__str__()  + '].', type=click.FLOAT )
@click.option('-cf', '--constants_file', required=False, default=default_constants, help='Simulation constants file. Default: ['      + default_constants           + '].', type=click.STRING)
@click.option('-i',  '--iteration',      required=False, default=default_iteration, help='Iteration in the constant file. Default: [' + default_iteration.__str__() + '].', type=click.INT   )
@click.option('-sf', '--snapshot_file',  required=True,  default=default_snapshot,  help='Simulation snapshot file. Default: ['       + default_snapshot            + '].', type=click.STRING)
def sedov(time, constants_file, iteration, snapshot_file):

    ''' 
        Compare SPH-EXA simulation with Analytical solution.
    '''
    
    '''
       * compute the analytical solution by running 
       * sedov_analytical and read the results
       * load the SPH simulation snapshot for the specified timestep
       * compare and plot SPH and analytical solutions
    '''
    
    print("")
    print("time           = " + time.__str__())
    print("constants_file = " + constants_file)
    print("iteration      = " + iteration.__str__())
    print("snapshot_file  = " + snapshot_file)
    print("")
        
    # Select the time
    if (time < 0.):
        print("No valid time = " + timestep.__str__())
        exit(-1)
    elif (time == 0.):
        if (iteration < 0): 
            print("No valid iteration = " + iteration.__str__())
            exit(-1)
        else:
            if (not os.path.isfile(constants_file)):
                print("Constants file [" + constants_file + "] doesn't exist.")
                exit(-1)
            else:
                print("Reading Constants file [" + constants_file + "]")
                file  = open(constants_file, 'r')
                lines = file.readlines()

                found = False     
                            
                for line in lines:
                    
                    tokens = line.split(' ')
                    iter_line = int(tokens[0])
                    time_line = float(tokens[1])
                    
                    if (iter_line == iteration):
                        time = time_line
                        found = True
                        break
                    
                if (not found):
                    print("Iteration=" + iteration.__str__() + " not found in the Constants file [" + constants_file + "].")
                    exit(-1)
                    
    print ("Solution will be calculated at time=" + time.__str__())




if __name__ == "__main__":
    cli()
