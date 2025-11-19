#!/usr/bin/python3 -u

import time
import numpy as np

from abinited.hamiltonian   import system, fixed_node
from abinited.log_info      import print_title, print_logging_info
from abinited.solvers       import davidson, lanczos
from abinited.wave_function import twfn

def main(n_basis, n_electrons, n_dets, \
        filename_dets, filename_hamiltonian, filename_twfn, \
        Ecore, \
        lines_to_skip_dets=0, lines_to_skip_hamiltonian=0, lines_to_skip_twfn=0, \
        n_eigenvals=1, is_eigenvec=True, max_iters=100000, \
        debug=False):
   
    print_title("Initializing the Hilbert Space",'=')
    print_logging_info("Number of basis functions (spin orb.) = {}".format(n_basis))
    print_logging_info("Number of electrons = {}".format(n_electrons))
    print_logging_info("Number of determinants = {}".format(n_dets))

    myHMLT = system.HILBERTSPACE(n_basis, n_electrons, n_dets, Ecore)

    print_logging_info("Reading the determinants from file: {}".format(filename_dets))
    print_logging_info("Skipping the first {} lines.".format(lines_to_skip_dets), level=1)
    start_time_dets = time.time()
    myHMLT.get_determinants(filename_dets, lines_to_skip_dets)
    end_time_dets = time.time()
    if debug:
        print_logging_info("Debugging info: first 10 determinants:")
        for i in range(min(10, n_dets)):
            print(" Det[{}] | ilut = {} | det = {}".format(i+1, myHMLT.ilut_list[i], myHMLT.occnum_vec[i,:]))
        print(" ... ")
    print_logging_info("Number of dets read = {}".format(len(myHMLT.ilut_list)), level=1)
    print_logging_info("Time taken to read determinants: {:.2f} seconds.".format(end_time_dets - start_time_dets), level=1)
    print_logging_info("Determinants read successfully ...")

    print_title("Initializing the System Hamiltonian H[fn]",'=')
    print_logging_info("Reading the hamiltonian from file: {}".format(filename_hamiltonian))
    print_logging_info("Skipping the first {} lines.".format(lines_to_skip_hamiltonian), level=1)
    start_time_hamiltonian = time.time()
    myHMLT.get_hamiltonian(filename_hamiltonian, lines_to_skip_hamiltonian)
    end_time_hamiltonian = time.time()
    if debug:
        print_logging_info("Debugging info: first 10x10 block of the Hamiltonian matrix:")  
        print(myHMLT.hamiltonian_matrix[:10,:10])
    print_logging_info("Hamltonian of shape [{},{}]".format(myHMLT.hamiltonian_matrix.shape[0], myHMLT.hamiltonian_matrix.shape[1]), level=1)
    print_logging_info("Time taken to read Hamiltonian: {:.2f} seconds.".format(end_time_hamiltonian - start_time_hamiltonian), level=1)
    print_logging_info("Hamiltonian read successfully ...")

    print_title('Diagonalizing the H[fn] Hamiltonian by Lanczos Algorithm', '=')
    eigenvalues  = np.zeros(n_eigenvals, dtype=np.float64)
    eigenvectors = np.zeros((myHMLT.n_dets, n_eigenvals), dtype=np.float64) if is_eigenvec else None
    mySolver = lanczos.LanczosDiagonalizer
    print_logging_info("Sarting Lanczos diagonalization for {} lowest eigenvalues ...".format(n_eigenvals))
    start_time_lanczos = time.time()
    if is_eigenvec:
        eigenvalues, eigenvectors = mySolver.diagonalize(myHMLT.hamiltonian_matrix, n_eigenvals, is_eigenvec, iters=max_iters)
    else:
        eigenvalues = mySolver.diagonalize(myHMLT.hamiltonian_matrix, n_eigenvals, is_eigenvec, iters=max_iters)
    end_time_lanczos = time.time()
    print_logging_info("Lanczos diagonalization completed in {:.2f} seconds.".format(end_time_lanczos - start_time_lanczos))
    print_logging_info("E[o]       = {}".format(eigenvalues[0]))
    print_logging_info("E[o]+Ecore = {}".format(eigenvalues[0]+Ecore))
    if debug:
        print_logging_info("Debugging info: printing requested roots:")
        print(eigenvalues[:n_eigenvals])
        if is_eigenvec:
            print_logging_info("Debugging info: printing first 10 components of the first eigenvector:")
            print(eigenvectors[:10,0])

    print_title("Initializing the Trial Wave Function",'=')
    print_logging_info("Reading the hamiltonian from file: {}".format(filename_twfn))
    print_logging_info("Skipping the first {} lines.".format(lines_to_skip_twfn), level=1)
    myTWFN = twfn.TRIALWAVEFUNCTION(n_basis, n_electrons, n_dets)
    myTWFN.get_trialwavefunction(filename_twfn, lines_to_skip_twfn)
    if debug:
        print_logging_info("Debugging info: first 10 Trial Wave Function amplitudes:")
        print(myTWFN.twfn[:10])
    print_logging_info("Number of amplitudes read = {}".format(len(myTWFN.twfn)), level=1)
    print_logging_info("Trial Wave Function read successfully ...")
    print_logging_info("Calculating the energy of the Trial Wave Function ...")
    start_time_twfn_energy = time.time()
    myTWFN.calculate_trial_energy(myHMLT.hamiltonian_matrix)
    end_time_twfn_energy = time.time()
    print_logging_info("Time taken to calculate the Trial Wave Function energy: {:.2f} seconds.".format(end_time_twfn_energy - start_time_twfn_energy), level=1)
    print_logging_info("E[T] = {}".format(myTWFN.e_trial + Ecore))

    quit()

    print_title("Initializing the Fixed-Node Hamiltonian H[fn]",'=')
    myFIXEDNODE = fixed_node.FIXEDNODE(n_basis, n_electrons, n_dets, Ecore)
    myFIXEDNODE.twfn = myTWFN.twfn
    print_logging_info("Building the Fixed-Node Hamiltonian matrix ...")
    start_time_fixednode = time.time()
    myFIXEDNODE.build_fixednode_hamiltonian(myHMLT.hamiltonian_matrix)
    end_time_fixednode = time.time()
    if debug:
        print_logging_info("Debugging info: first 10x10 block of the Fixed-Node Hamiltonian matrix:")  
        print(myFIXEDNODE.fixednode_hamiltonian_matrix[:10,:10])
    print_logging_info("Time taken to build the Fixed-Node Hamiltonian: {:.2f} seconds.".format(end_time_fixednode - start_time_fixednode), level=1)
    print_logging_info("Fixed-Node Hamiltonian built successfully ...")
    print_logging_info("Calculating the expectation value of the Sign Flip Potential ...")
    start_time_sfp = time.time()
    myFIXEDNODE.get_av_sign_flip_potential(myHMLT.hamiltonian_matrix)
    end_time_sfp = time.time()
    print_logging_info("Time taken to calculate the Sign Flip Potential: {:.2f} seconds.".format(end_time_sfp - start_time_sfp), level=1)
    print_logging_info(" <Vsf> = {}".format(myFIXEDNODE.av_sign_flip_potential))

    print_title('Diagonalizing the H[fn] Hamiltonian by Lanczos Algorithm', '=')
    fixednode_eigenvalues  = np.zeros(n_eigenvals, dtype=np.float64)
    fixednode_eigenvectors = np.zeros((myFIXEDNODE.n_dets, n_eigenvals), dtype=np.float64) if is_eigenvec else None
    print_logging_info("Sarting Lanczos diagonalization for {} lowest eigenvalues ...".format(n_eigenvals))
    start_time_lanczos = time.time()
    if is_eigenvec:
        fixednode_eigenvalues, fixednode_eigenvectors = mySolver.diagonalize(myFIXEDNODE.fixednode_hamiltonian_matrix, n_eigenvals, is_eigenvec)
    else:
        fixednode_eigenvalues = mySolver.diagonalize(myFIXEDNODE.fixednode_hamiltonian_matrix, n_eigenvals, is_eigenvec)
    end_time_lanczos = time.time()
    print_logging_info("Lanczos diagonalization completed in {:.2f} seconds.".format(end_time_lanczos - start_time_lanczos))
    print_logging_info("E[fn]       = {}".format(fixednode_eigenvalues[0]))
    print_logging_info("E[fn]+Ecore = {}".format(fixednode_eigenvalues[0]+Ecore))
    if debug:
        print_logging_info("Debugging info: printing requested fixed-node roots:")
        print(fixednode_eigenvalues[:n_eigenvals])
        if is_eigenvec:
            print_logging_info("Debugging info: printing first 10 components of the first fixed-node eigenvector:")
            print(fixednode_eigenvectors[:10,0])

if __name__ == '__main__':

    n_basis = 14
    n_electrons = 10
    n_dets = 441

    Ecore = -1570.756534354645

    filename_dets = 'DETSPACE'
    lines_to_skip_dets = 3
    filename_hamiltonian = 'FIXEDNODE-HAMILTONIAN'
    lines_to_skip_hamiltonian = 4
    filename_twfn = 'FIXEDNODE-TWFN'
    lines_to_skip_twfn = 4
    #filename_fixednode_hamiltonian = 'FIXEDNODE-HAMILTONIAN'
    #lines_to_skip_fixednode_hamiltonian = 4

    n_roots = 1
    is_eigenvec = True
    max_iters = 500000

    debug = True

    main(n_basis, n_electrons, n_dets, \
            filename_dets, filename_hamiltonian, filename_twfn, \
            Ecore, \
            lines_to_skip_dets, lines_to_skip_hamiltonian, lines_to_skip_twfn, \
            n_roots, is_eigenvec, max_iters=max_iters, \
            debug=debug)
