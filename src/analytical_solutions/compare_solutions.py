#!/usr/bin/env python3
# MIT License
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

"""
Command line utility for compare analytical solutions of some SPH-EXA test
simulations.

Usage examples:

    $ python src/analytical_solutions//compare_solutions.py --help
    $ python src/analytical_solutions//compare_solutions.py --version

    Check Sedov density snapshot:
    $ gnuplot
    gnuplot> plot "bin/dump_sedov0.txt" u (abs($3/$7)<2.?$1:1/0):2:8 w p pt 7 lc palette z
    gnuplot> plot "bin/dump_sedov200.txt" u (abs($3/$7)<2.?$1:1/0):2:8 w p pt 7 lc palette z
    
    $ python src/analytical_solutions/compare_solutions.py sedov --help
    $ python src/analytical_solutions/compare_solutions.py sedov --binary_file \
    bin/sedov_solution --only_solution --time 0.0673556 --out_dir bin/
    $ python src/analytical_solutions/compare_solutions.py sedov --binary_file \
    bin/sedov_solution --constants_file ./bin/constants_sedov.txt \
    --iteration 200 --nparts 125000 --snapshot_file ./bin/dump_sedov200.txt \
    --ascii --out_dir bin/ --error_rho --error_p --error_vel

    Check Noh density snapshot:
    $ gnuplot
    gnuplot> plot "bin/dump_noh0.txt" u (abs($3/$7)<2.?$1:1/0):2:8 w p pt 7 lc palette z
    gnuplot> plot "bin/dump_noh1000.txt" u (abs($3/$7)<2.?$1:1/0):2:8 w p pt 7 lc palette z
        
    $ python src/analytical_solutions/compare_solutions.py noh --help
    $ python src/analytical_solutions/compare_solutions.py noh --binary_file \
    bin/noh_solution --only_solution --time 0.281115 --out_dir bin/
    $ python src/analytical_solutions/compare_solutions.py noh --binary_file \
    bin/noh_solution --constants_file ./bin/constants_noh.txt \
    --iteration 1000 --nparts 1000000 --snapshot_file ./bin/dump_noh1000.txt \
    --ascii --out_dir bin/ --error_u --error_vel --error_cs
    
"""

__program__ = "compare_solutions.py"
__author__ = "Jose A. Escartin (ja.escartin@gmail.com)"
__version__ = "0.2.0"

import click
import os
from matplotlib import pyplot as plt


@click.group(
    invoke_without_command=True,
    context_settings=dict(help_option_names=["-h", "--help"]),
)
@click.option(
    "--version", "-v", is_flag=True,
    help="Show version number of compare_solutions.py"
)
def cli(version):
    """The compare_solutions.py CLI can be used to compare the some SPH-EXA
    simulations tests with their analytical solutions."""

    if version:
        click.echo("\ncompare_solutions.py version: " + __version__ + "\n")


def check_file(
    binary_file
):
    """
        Check binary_file
    """
    
    if not os.path.isfile(binary_file):
        print("\nBinary file [" + binary_file + "] doesn't exist.")
        exit(-1)
    else:
        print("\nBinary file [" + binary_file + "] in the path.\n")


def get_time(
    time,
    constants_file,
    iteration
):

    """    
        Select the time
    """
    
    if time < 0.0:
        print("No valid time=" + timestep.__str__())
        exit(-1)
    elif time == 0.0:
        if iteration < 0:
            print("No valid iteration=" + iteration.__str__() + ". It should be >0")
            exit(-1)
        else:
            if not os.path.isfile(constants_file):
                print("Constants file [" + constants_file + "] doesn't exist.")
                exit(-1)
            else:
                print("Reading Constants file [" + constants_file + "]")
                file = open(constants_file, "r")
                lines = file.readlines()

                found = False
                for line in lines:
                    tokens = line.split(" ")
                    iter_line = int(tokens[0])
                    time_line = float(tokens[1])

                    if iter_line == iteration:
                        time = time_line
                        found = True
                        break

                file.close()

                if not found:
                    print(
                        "Iteration="
                        + iteration.__str__()
                        + " not found in the Constants file ["
                        + constants_file
                        + "]."
                    )
                    exit(-1)

    print("Solution will be calculated at time=" + time.__str__() + "\n")
    
    return time


def check_snapshot(
    nparts,
    snapshot_file
):
    """
        Check simulated data
    """
    
    if nparts < 0:
        print("No valid number of particles=" + nparts.__str__())
        exit(-1)
    else:
        if not os.path.isfile(snapshot_file):
            print("Snapshot file [" + snapshot_file + "] doesn't exist.")
            exit(-1)
        else:
            print("Reading Snapshot file [" + snapshot_file + "]")
            file = open(snapshot_file, "r")
            lines = file.readlines()

            n = 0
            for line in lines:
                tokens = line.split(" ")
                n = n + 1

            file.close()

            if n != nparts:
                errmsg = (
                    "Parameter nParts=" + nparts.__str__() + " doesn't match with"  
                    + " the number of particle lines found in the Snapshot file " 
                    + snapshot_file + "."
                )
                print(errmsg)
                exit(-1)

    print("Particles checked from the file [nparts=" + nparts.__str__() + "]\n")
        

def sedov_load_file(
    file
):
    """
        Load file
    """
        
    if not os.path.isfile(file):
        print("The file [" + file + "] doesn't exist.")
        exit(-1)
    else:
        print("Reading file [" + file + "  ]")
        file = open(file, "r")

        # Read data lines without header
        lines = file.readlines()[1:]

        # Empty vectors
        r = []
        rho = []
        u = []
        p = []
        vel = []
        cs = []
        rhoShock = []
        uShock = []
        pShock = []
        velShock = []
        csShock = []
        rho0 = []

        for line in lines:
            tokens = line.split()

            r.append(float(tokens[0]))
            rho.append(float(tokens[1]))
            u.append(float(tokens[2]))
            p.append(float(tokens[3]))
            vel.append(float(tokens[4]))
            cs.append(float(tokens[5]))
            rhoShock.append(float(tokens[6]))
            uShock.append(float(tokens[7]))
            pShock.append(float(tokens[8]))
            velShock.append(float(tokens[9]))
            csShock.append(float(tokens[10]))
            rho0.append(float(tokens[11]))

        file.close()
        
        return r, rho,u,p,vel,cs, rhoShock,uShock,pShock,velShock,csShock, rho0


def noh_load_file(
    file
):
    """
        Load file
    """
        
    if not os.path.isfile(file):
        print("The file [" + file + "] doesn't exist.")
        exit(-1)
    else:
        print("Reading file [" + file + "  ]")
        file = open(file, "r")

        # Read data lines without header
        lines = file.readlines()[1:]

        # Empty vectors
        r = []
        rho = []
        u = []
        p = []
        vel = []
        cs = []

        for line in lines:
            tokens = line.split()

            r.append(float(tokens[0]))
            rho.append(float(tokens[1]))
            u.append(float(tokens[2]))
            p.append(float(tokens[3]))
            vel.append(float(tokens[4]))
            cs.append(float(tokens[5]))

        file.close()
        
        return r, rho,u,p,vel,cs


def plot_comparation(
    figureName, 
    sim_x,
    sol_x, 
    sim_y,
    sol_y, 
    x_label, 
    y_label, 
    title
):
    plt.plot(sim_x, sim_y, ".", label="Simulation")
    plt.plot(sol_x, sol_y, label="Solution")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.draw()
    plt.title(title)
    plt.legend(loc="upper right")
    plt.savefig(figureName, format="png")
    plt.figure().clear()
        

def check_errors_L1(
    error_rho,
    delta_rho,
    error_u,
    delta_u,
    error_p,
    delta_p,
    error_vel,
    delta_vel,
    error_cs,
    delta_cs
):
    
    """
        Check if it is need to calcule L1 errors
    """
    
    if error_rho or error_u or error_p or error_vel or error_cs:
        check_L1 = True
    else:
        check_L1 = False
    print("\nCheck L1 errors = " + check_L1.__str__())
    
    if check_L1:
        if error_rho:
            print(" * error_rho <= " + delta_rho.__str__() + " ?")
        if error_u:
            print(" * error_u   <= " + delta_u.__str__() + " ?")
        if error_p:
            print(" * error_p   <= " + delta_p.__str__() + " ?")
        if error_vel:
            print(" * error_vel <= " + delta_vel.__str__() + " ?")
        if error_cs:
            print(" * error_cs  <= " + delta_cs.__str__() + " ?")

    print("")
    
    return check_L1


def evaluate_errors_L1(
    check_L1,
    nparts,
    sol_rho, 
    sim_rho,
    error_rho,
    delta_rho,
    sol_u, 
    sim_u,
    error_u,
    delta_u,
    sol_p,
    sim_p,
    error_p,
    delta_p,
    sol_vel,
    sim_vel,
    error_vel,
    delta_vel,
    sol_cs,
    sim_cs,
    error_cs,
    delta_cs,
    time,
    out_dir,
    test_name
):

    """
    Errors L1 : Normalize distance between the theorical and simulated
    value.

    For scalar quatities: rho, pressure, internal energy, ... :
        L1=Sum[abs(theoretical - simulated)] / N

    For vector quantities: velocity, ... :
        L1 = sum ( sqrt (  (x_theo - x_sim)**2 +
                           (y_theo - y_sim)**2 +
                           (z_theo - z_sim)**2 ) ) / N
    """
    
    # Checking errors
    successfully = True
    if check_L1:

        print("\nChecking Errors L1 ...")

        # Calculate errors
        L1_rho = 0.0
        L1_u = 0.0
        L1_p = 0.0
        L1_vel = 0.0
        L1_cs = 0.0

        for i in range(nparts):

            L1_rho += abs(sol_rho[i] - sim_rho[i])
            L1_u += abs(sol_u[i] - sim_u[i])
            L1_p += abs(sol_p[i] - sim_p[i])
            L1_vel += abs(sol_vel[i] - sim_vel[i])
            L1_cs += abs(sol_cs[i] - sim_cs[i])

        L1_rho /= nparts
        L1_u /= nparts
        L1_p /= nparts
        L1_vel /= nparts
        L1_cs /= nparts

        # Write data in the file
        errFile = out_dir + test_name + "_errors_L1_" + time.__str__() + ".txt"
        file = open(errFile, "w")
        hstr = "#   01:L1_rho    02:L1_u    03:L1_p   04:L1_vel    05:L1_cs\n"
        file.write(hstr)
        file.write(L1_rho.__str__())
        file.write(" " + L1_u.__str__())
        file.write(" " + L1_p.__str__())
        file.write(" " + L1_vel.__str__())
        file.write(" " + L1_cs.__str__())
        file.write("\n")
        file.close()

        print("Error L1_rho = " + L1_rho.__str__())
        print("Error L1_u   = " + L1_u.__str__())
        print("Error L1_p   = " + L1_p.__str__())
        print("Error L1_vel = " + L1_vel.__str__())
        print("Error L1_cs  = " + L1_cs.__str__())
        print("")

        # Check errors L1
        
        if error_rho:
            if L1_rho < delta_rho:
                print(
                    "Checked Error L1_rho successfully: "
                    + L1_rho.__str__()
                    + " <= "
                    + delta_rho.__str__()
                )
            else:
                print(
                    "Checked Error L1_rho       failed: "
                    + L1_rho.__str__()
                    + " > "
                    + delta_rho.__str__()
                )
                successfully = False

        if error_u:
            if L1_u < delta_u:
                print(
                    "Checked Error L1_u   successfully: "
                    + L1_u.__str__()
                    + " <= "
                    + delta_u.__str__()
                )
            else:
                print(
                    "Checked Error L1_u         failed: "
                    + L1_u.__str__()
                    + " > "
                    + delta_u.__str__()
                )
                successfully = False

        if error_p:
            if L1_p < delta_p:
                print(
                    "Checked Error L1_p   successfully: "
                    + L1_p.__str__()
                    + " <= "
                    + delta_p.__str__()
                )
            else:
                print(
                    "Checked Error L1_p         failed: "
                    + L1_p.__str__()
                    + " > "
                    + delta_p.__str__()
                )
                successfully = False

        if error_vel:
            if L1_vel < delta_vel:
                print(
                    "Checked Error L1_vel successfully: "
                    + L1_vel.__str__()
                    + " <= "
                    + delta_vel.__str__()
                )
            else:
                print(
                    "Checked Error L1_vel       failed: "
                    + L1_vel.__str__()
                    + " > "
                    + delta_vel.__str__()
                )
                successfully = False

        if error_cs:
            if L1_cs < delta_cs:
                print(
                    "Checked Error L1_cs  successfully: "
                    + L1_cs.__str__()
                    + " <= "
                    + delta_cs.__str__()
                )
            else:
                print(
                    "Checked Error L1_cs        failed: "
                    + L1_cs.__str__()
                    + " > "
                    + delta_cs.__str__()
                )
                successfully = False

        print("\nWriting Errors L1 file [" + errFile + " ]")

    return successfully


default_binary = "./solution"

default_only_solution = False
default_time = 0.0

default_nparts = 10000
default_snapshot = "./dump_0.txt"
default_ascii = False

default_constants = "./constants.txt"
default_iteration = -1

default_no_plots = False

default_outDir = "./"

default_error_rho = False
default_delta_rho = 1.0
default_error_u = False
default_delta_u = 1.0
default_error_p = False
default_delta_p = 1.0
default_error_vel = False
default_delta_vel = 1.0
default_error_cs = False
default_delta_cs = 1.0


@cli.command()
@click.option(
    "-bf",
    "--binary_file",
    required=False,
    default=default_binary,
    help="Binary file to compare. Default: [" + default_binary + "].",
    type=click.STRING,
)
@click.option(
    "-s",
    "--only_solution",
    required=False,
    default=default_only_solution,
    help="Only analytical solution. Default: [" + default_only_solution.__str__() + "].",
    is_flag=True,
)
@click.option(
    "-t",
    "--time",
    required=False,
    default=default_time,
    help="Solution time. Default: [" + default_time.__str__() + "].",
    type=click.FLOAT,
)
@click.option(
    "-n",
    "--nparts",
    required=True,
    default=default_nparts,
    help="Number of particles. Default: [" + default_nparts.__str__() + "].",
    type=click.INT,
)
@click.option(
    "-sf",
    "--snapshot_file",
    required=False,
    default=default_snapshot,
    help="Simulation snapshot file. Default: [" + default_snapshot + "].",
    type=click.STRING,
)
@click.option(
    "-a",
    "--ascii",
    required=False,
    default=default_error_vel,
    help="Snapshot file in ASCII format. Default: [" + default_ascii.__str__() + "].",
    is_flag=True,
)
@click.option(
    "-cf",
    "--constants_file",
    required=False,
    default=default_constants,
    help="Simulation constants file. Default: [" + default_constants + "].",
    type=click.STRING,
)
@click.option(
    "-i",
    "--iteration",
    required=False,
    default=default_iteration,
    help="Iteration in the constant file. Default: ["
    + default_iteration.__str__()
    + "].",
    type=click.INT,
)
@click.option(
    "-np",
    "--no_plots",
    required=False,
    default=default_no_plots,
    help="No create plots. Default: [" + default_no_plots.__str__() + "].",
    is_flag=True,
)
@click.option(
    "-o",
    "--out_dir",
    required=False,
    default=default_outDir,
    help="Output directory. Default: [" + default_outDir + "].",
    type=click.Path(exists=True),
)
@click.option(
    "-er",
    "--error_rho",
    required=False,
    default=default_error_rho,
    help="Check error L1 in rho. Default: [" + default_error_rho.__str__() + "].",
    is_flag=True,
)
@click.option(
    "-dr",
    "--delta_rho",
    required=False,
    default=default_delta_rho,
    help="Delta error L1 in rho. Default: [" + default_delta_rho.__str__() + "].",
    type=click.FLOAT,
)
@click.option(
    "-eu",
    "--error_u",
    required=False,
    default=default_error_u,
    help="Check error L1 in u.   Default: [" + default_error_u.__str__() + "].",
    is_flag=True,
)
@click.option(
    "-du",
    "--delta_u",
    required=False,
    default=default_delta_u,
    help="Delta error L1 in u.   Default: [" + default_delta_u.__str__() + "].",
    type=click.FLOAT,
)
@click.option(
    "-ep",
    "--error_p",
    required=False,
    default=default_error_p,
    help="Check error L1 in p.   Default: [" + default_error_p.__str__() + "].",
    is_flag=True,
)
@click.option(
    "-dp",
    "--delta_p",
    required=False,
    default=default_delta_p,
    help="Delta error L1 in p.   Default: [" + default_delta_p.__str__() + "].",
    type=click.FLOAT,
)
@click.option(
    "-ev",
    "--error_vel",
    required=False,
    default=default_error_vel,
    help="Check error L1 in vel. Default: [" + default_error_vel.__str__() + "].",
    is_flag=True,
)
@click.option(
    "-dv",
    "--delta_vel",
    required=False,
    default=default_delta_vel,
    help="Delta error L1 in vel. Default: [" + default_delta_vel.__str__() + "].",
    type=click.FLOAT,
)
@click.option(
    "-ec",
    "--error_cs",
    required=False,
    default=default_error_cs,
    help="Check error L1 in cs.  Default: [" + default_error_cs.__str__() + "].",
    is_flag=True,
)
@click.option(
    "-dc",
    "--delta_cs",
    required=False,
    default=default_delta_cs,
    help="Delta error L1 in cs.  Default: [" + default_delta_cs.__str__() + "].",
    type=click.FLOAT,
)
def sedov(
    binary_file,
    only_solution,
    time,
    nparts,
    snapshot_file,
    ascii,
    constants_file,
    iteration,
    no_plots,
    out_dir,
    error_rho,
    delta_rho,
    error_u,
    delta_u,
    error_p,
    delta_p,
    error_vel,
    delta_vel,
    error_cs,
    delta_cs,
):

    """
    Compare SPH-EXA simulation with Sedov Analytical solution.
    """

    # Parameters
    print("")
    print("binary_file     = " + binary_file)
    print("only_solution   = " + only_solution.__str__())
    print("time            = " + time.__str__())
    print("nparts          = " + nparts.__str__())
    print("snapshot_file   = " + snapshot_file)
    print("constants_file  = " + constants_file)
    print("iteration       = " + iteration.__str__())
    print("no_plots        = " + no_plots.__str__())
    print("out_dir         = " + out_dir.__str__())

    # Check binary_file
    check_file(binary_file)

    if only_solution:
        check_L1 = False
    else:
        
        # Check if it is need to calcule L1 errors
        check_L1 = check_errors_L1(
            error_rho,delta_rho,
            error_u,delta_u,
            error_p,delta_p,
            error_vel,delta_vel,
            error_cs,delta_cs)

        # Select the time
        time = get_time(time, constants_file, iteration)
    
        # Check simulated data
        check_snapshot(nparts, snapshot_file)

    # Make command line and execute
    command = binary_file
    command += " --outDir " + out_dir
    command += " --time " + time.__str__()
    if only_solution:
        command += " --only_solution"
    else:
        command += " --nParts " + nparts.__str__()
        command += " --input " + snapshot_file
    if check_L1:
        command += " --complete"
    if ascii:
        command += " --ascii"
    print("Command:\n" + command)
    os.system(command)


    print("Checking outputs ...")

    # Load Solution file
    solFile = out_dir + "sedov_solution_" + time.__str__() + ".txt"
    sol_r, sol_rho,sol_u,sol_p,sol_vel,sol_cs, sol_rhoShock,sol_uShock,sol_pShock,sol_velShock,sol_csShock, sol_rho0 = sedov_load_file(solFile)

    if only_solution:
        print("\nSolution generated successfully!\n")
        exit(0)

    # Load Simulated file
    simFile = out_dir + "sedov_simulation_" + time.__str__() + ".txt"
    sim_r, sim_rho,sim_u,sim_p,sim_vel,sim_cs, sim_rhoShock,sim_uShock,sim_pShock,sim_velShock,sim_csShock, sim_rho0 = sedov_load_file(simFile)


    # Plot graphics
    if not no_plots:
        print("\nGenerating graphics ...")

        figureName = out_dir + "sedov_density_" + time.__str__() + ".png"
        plot_comparation(figureName, sim_r,sol_r, sim_rho,sol_rho, "r (cm)", "rho (g.cm^3)", "Density")
        print("'Radius vs Density'  done [" + figureName + "   ]")

        figureName = out_dir + "sedov_pressure_" + time.__str__() + ".png"
        plot_comparation(figureName, sim_r,sol_r, sim_p,sol_p, "r (cm)", "p (erg.cm^-3)", "Pressure")
        print("'Radius vs Pressure' done [" + figureName + "  ]")

        figureName = out_dir + "sedov_velocity_" + time.__str__() + ".png"
        plot_comparation(figureName, sim_r,sol_r, sim_vel,sol_vel, "r (cm)", "|vel| (vm.s^-1)", "Velocity")
        print("'Radius vs Velocity' done [" + figureName + "  ]")


    # Checking errors L1
    successfully = evaluate_errors_L1(
        check_L1,
        nparts,
        sol_rho,sim_rho,error_rho,delta_rho,
        sol_u,sim_u,error_u,delta_u,
        sol_p,sim_p,error_p,delta_p,
        sol_vel,sim_vel,error_vel,delta_vel,
        sol_cs,sim_cs,error_cs,delta_cs,
        time,
        out_dir, "sedov");

    # Check finish
    if successfully:
        print("\nComparison finished successfully!\n")
        exit(0)
    else:
        print("\nComparison failed!\n")
        exit(-1)



@cli.command()
@click.option(
    "-bf",
    "--binary_file",
    required=False,
    default=default_binary,
    help="Binary file to compare. Default: [" + default_binary + "].",
    type=click.STRING,
)
@click.option(
    "-s",
    "--only_solution",
    required=False,
    default=default_only_solution,
    help="Only analytical solution. Default: [" + default_only_solution.__str__() + "].",
    is_flag=True,
)
@click.option(
    "-t",
    "--time",
    required=False,
    default=default_time,
    help="Solution time. Default: [" + default_time.__str__() + "].",
    type=click.FLOAT,
)
@click.option(
    "-n",
    "--nparts",
    required=True,
    default=default_nparts,
    help="Number of particles. Default: [" + default_nparts.__str__() + "].",
    type=click.INT,
)
@click.option(
    "-sf",
    "--snapshot_file",
    required=False,
    default=default_snapshot,
    help="Simulation snapshot file. Default: [" + default_snapshot + "].",
    type=click.STRING,
)
@click.option(
    "-a",
    "--ascii",
    required=False,
    default=default_error_vel,
    help="Snapshot file in ASCII format. Default: [" + default_ascii.__str__() + "].",
    is_flag=True,
)
@click.option(
    "-cf",
    "--constants_file",
    required=False,
    default=default_constants,
    help="Simulation constants file. Default: [" + default_constants + "].",
    type=click.STRING,
)
@click.option(
    "-i",
    "--iteration",
    required=False,
    default=default_iteration,
    help="Iteration in the constant file. Default: ["
    + default_iteration.__str__()
    + "].",
    type=click.INT,
)
@click.option(
    "-np",
    "--no_plots",
    required=False,
    default=default_no_plots,
    help="No create plots. Default: [" + default_no_plots.__str__() + "].",
    is_flag=True,
)
@click.option(
    "-o",
    "--out_dir",
    required=False,
    default=default_outDir,
    help="Output directory. Default: [" + default_outDir + "].",
    type=click.Path(exists=True),
)
@click.option(
    "-er",
    "--error_rho",
    required=False,
    default=default_error_rho,
    help="Check error L1 in rho. Default: [" + default_error_rho.__str__() + "].",
    is_flag=True,
)
@click.option(
    "-dr",
    "--delta_rho",
    required=False,
    default=default_delta_rho,
    help="Delta error L1 in rho. Default: [" + default_delta_rho.__str__() + "].",
    type=click.FLOAT,
)
@click.option(
    "-eu",
    "--error_u",
    required=False,
    default=default_error_u,
    help="Check error L1 in u.   Default: [" + default_error_u.__str__() + "].",
    is_flag=True,
)
@click.option(
    "-du",
    "--delta_u",
    required=False,
    default=default_delta_u,
    help="Delta error L1 in u.   Default: [" + default_delta_u.__str__() + "].",
    type=click.FLOAT,
)
@click.option(
    "-ep",
    "--error_p",
    required=False,
    default=default_error_p,
    help="Check error L1 in p.   Default: [" + default_error_p.__str__() + "].",
    is_flag=True,
)
@click.option(
    "-dp",
    "--delta_p",
    required=False,
    default=default_delta_p,
    help="Delta error L1 in p.   Default: [" + default_delta_p.__str__() + "].",
    type=click.FLOAT,
)
@click.option(
    "-ev",
    "--error_vel",
    required=False,
    default=default_error_vel,
    help="Check error L1 in vel. Default: [" + default_error_vel.__str__() + "].",
    is_flag=True,
)
@click.option(
    "-dv",
    "--delta_vel",
    required=False,
    default=default_delta_vel,
    help="Delta error L1 in vel. Default: [" + default_delta_vel.__str__() + "].",
    type=click.FLOAT,
)
@click.option(
    "-ec",
    "--error_cs",
    required=False,
    default=default_error_cs,
    help="Check error L1 in cs.  Default: [" + default_error_cs.__str__() + "].",
    is_flag=True,
)
@click.option(
    "-dc",
    "--delta_cs",
    required=False,
    default=default_delta_cs,
    help="Delta error L1 in cs.  Default: [" + default_delta_cs.__str__() + "].",
    type=click.FLOAT,
)
def noh(
    binary_file,
    only_solution,
    time,
    nparts,
    snapshot_file,
    ascii,
    constants_file,
    iteration,
    no_plots,
    out_dir,
    error_rho,
    delta_rho,
    error_u,
    delta_u,
    error_p,
    delta_p,
    error_vel,
    delta_vel,
    error_cs,
    delta_cs,
):

    """
    Compare SPH-EXA simulation with Noh Analytical solution.
    """

    # Parameters
    print("")
    print("binary_file     = " + binary_file)
    print("only_solution   = " + only_solution.__str__())
    print("time            = " + time.__str__())
    print("nparts          = " + nparts.__str__())
    print("snapshot_file   = " + snapshot_file)
    print("constants_file  = " + constants_file)
    print("iteration       = " + iteration.__str__())
    print("no_plots        = " + no_plots.__str__())
    print("out_dir         = " + out_dir.__str__())

    # Check binary_file
    check_file(binary_file)

    if only_solution:
        check_L1 = False
    else:
        
        # Check if it is need to calcule L1 errors
        check_L1 = check_errors_L1(
            error_rho,delta_rho,
            error_u,delta_u,
            error_p,delta_p,
            error_vel,delta_vel,
            error_cs,delta_cs)
    
        # Select the time
        time = get_time(time, constants_file, iteration)
    
        # Check simulated data
        check_snapshot(nparts, snapshot_file)

    # Make command line and execute
    command = binary_file
    command += " --outDir " + out_dir
    command += " --time " + time.__str__()
    if only_solution:
        command += " --only_solution"
    else:
        command += " --nParts " + nparts.__str__()
        command += " --input " + snapshot_file
    if check_L1:
        command += " --complete "
    if ascii:
        command += " --ascii"
    print("Command:\n" + command)
    os.system(command)


    print("Checking outputs ...")

    # Load Solution file
    solFile = out_dir + "noh_solution_" + time.__str__() + ".txt"
    sol_r, sol_rho,sol_u,sol_p,sol_vel,sol_cs = noh_load_file(solFile)

    if only_solution:
        print("\nSolution generated successfully!\n")
        exit(0)

    # Load Simulated file
    simFile = out_dir + "noh_simulation_" + time.__str__() + ".txt"
    sim_r, sim_rho,sim_u,sim_p,sim_vel,sim_cs = noh_load_file(simFile)


    # Plot graphics
    if not no_plots:
        print("\nGenerating graphics ...")

        figureName = out_dir + "noh_density_" + time.__str__() + ".png"
        plot_comparation(figureName, sim_r,sol_r, sim_rho,sol_rho, "r (cm)", "rho (g.cm^3)", "Density")
        print("'Radius vs Density'  done [" + figureName + "   ]")

        figureName = out_dir + "noh_energy_" + time.__str__() + ".png"
        plot_comparation(figureName, sim_r,sol_r, sim_u,sol_u, "r (cm)", "u (erg.g^-1)", "Energy")
        print("'Radius vs Energy' done [" + figureName + "  ]")

        figureName = out_dir + "noh_pressure_" + time.__str__() + ".png"
        plot_comparation(figureName, sim_r,sol_r, sim_p,sol_p, "r (cm)", "p (erg.cm^-3)", "Pressure")
        print("'Radius vs Pressure' done [" + figureName + "  ]")

        figureName = out_dir + "noh_velocity_" + time.__str__() + ".png"
        plot_comparation(figureName, sim_r,sol_r, sim_vel,sol_vel, "r (cm)", "|vel| (cm.s^-1)", "Velocity")
        print("'Radius vs Velocity' done [" + figureName + "  ]")


    # Checking errors L1
    successfully = evaluate_errors_L1(
        check_L1,
        nparts,
        sol_rho,sim_rho,error_rho,delta_rho,
        sol_u,sim_u,error_u,delta_u,
        sol_p,sim_p,error_p,delta_p,
        sol_vel,sim_vel,error_vel,delta_vel,
        sol_cs,sim_cs,error_cs,delta_cs,
        time,
        out_dir, "noh");

    # Check finish
    if successfully:
        print("\nComparison finished successfully!\n")
        exit(0)
    else:
        print("\nComparison failed!\n")
        exit(-1)


if __name__ == "__main__":
    cli()
