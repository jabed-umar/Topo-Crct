# Purpose: This file contains the code for the SSH Model simulation.
# Author: Jabed Umar


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import math


# Define the Hamiltonian
def SSH(n, diag, alternate1, alternate2):
    """This function generates the SSH Hamiltonian of N unit cells.

    Args:
        n (Int): Number of cells.
        diag (float): Onsite energy term.
        alternate1 (float): Intracell hopping term.
        alternate2 (float): Intercell hopping term.
    Raises:
        ValueError: size of the Hamiltonian must be at least 2x2.
        ValueError: size of the Hamiltonian must be even for alternating diagonals.

    Returns:
       array: The generated Hamiltonian.
    """
    if n < 2:
        raise ValueError("The size of the matrix must be at least 2x2.")

    if n % 2 != 0:
        raise ValueError("The size of the matrix must be even for alternating diagonals.")
    
    matrix = np.zeros((n, n))

    for i in range(n):
        matrix[i, i] = diag  # Main diagonal

        if i < n - 1:
            if i % 2 == 0:
                matrix[i, i + 1] = alternate1  # Alternate upper diagonal
            else:
                matrix[i, i + 1] = alternate2  # Alternate upper diagonal

        if i > 0:
            if i % 2 == 0:
                matrix[i, i - 1] = alternate2  # Alternate lower diagonal
            else:
                matrix[i, i - 1] = alternate1  # Alternate lower diagonal

    return matrix



# Define the Hamiltonian of the Circuit
def SSH_sim(c1,c2,L,m,n):
    """This function generated the Laplacian of the circuit

    Args:
        c1 (float): first capacitor
        c2 (float): 2nd capacitor
        L (float): inductor
        m (float): angular frequency
        n (int): size of the matrix

    Raises:
        ValueError: Size of the matrix must be at least 2x2.
        ValueError: Size of the matrix must be even for alternating diagonals.

    Returns:
        array: the laplacian (Hamiltonian) of the circuit
    """
    if n < 2:
        raise ValueError("The size of the matrix must be at least 2x2.")

    if n % 2 != 0:
        raise ValueError("The size of the matrix must be even for alternating diagonals.")

    matrix = np.zeros((n, n))
    m1 = 2*(math.pi)*m
    t = c1/c2  # t is intra cell hopping 
    w = (1/(L*(c1+c2)))**0.5 # w is the resonance frequency of the SSH model
    for i in range(n):
        matrix[i, i] =  (1+t)*(1-(w/m1)**2) # Main diagonal

        if i < n - 1:
            if i % 2 == 0:
                matrix[i, i + 1] = -t  # Alternate upper diagonal
            else:
                matrix[i, i + 1] = -1  # Alternate upper diagonal

        if i > 0:
            if i % 2 == 0:
                matrix[i, i - 1] = -1 # Alternate lower diagonal
            else:
                matrix[i, i - 1] = -t  # Alternate lower diagonal

    return matrix


# Calculate the Impedance between any two nodes in the SSH circuit at resonance
def impedance(a,b,M):
    """This function calculates the impedance between any two nodes in the SSH circuit at resonance.

    Args:
        a (int): Give the first node.
        b (int): give the second node.
        M (array): SSH Hamiltonian.

    Raises:
        ValueError: length of the nodes must be less than the size of the eigenstates.

    Returns:
        float: The impedance between the two nodes.
    """
    eigenvalues, eigenvectors = np.linalg.eig(M)  # find the eigenvalues and eigenvectors of the SSH Hamiltonian
    E_min = np.min(np.abs(eigenvalues))  # find the minimum energy states
    i  = np.argmin(np.abs(eigenvalues))  # find the index of the minimum energy states
    ith_column = eigenvectors[:, i]  # find the eigenstates corresponding to the minimum energy states
    if a & b > len(ith_column):
        raise ValueError("The Nodes must be less than the size of the eigenstates.")
    if a == b:
        impedance = 0
    if a != b:
        impedance = (abs((ith_column[a]) ** 2 - (ith_column[b])) ** 2/ E_min)
    
    return impedance


# Calculate the resonance frequency of the SSH circuit
def resonance(m1,m2,L):
    """Calculates the resonance frequency of a SSH circuit
    Args:
        m1 (float): value of 1st capacitor
        m2 (float): value of 2nd capacitor
        L (float): value of inductor

    Returns:
        float: value of resonance frequency of the circuit
    """
    import numpy as np
    w = np.sqrt(1/(L*(m1+m2)))
    f = w/(2*math.pi)
    return f,w


# Plot the impedance vs frequency graph of the circuit
def plott(c1, c2, n1, n2,k,steps = 1,):
    """This function plots the impedance vs frequency graph of the circuit

    Args:
        c1 (float): The capacitance of the first capacitor
        c2 (float): The capacitance of the second capacitor
        n1 (Int): the first node
        n2 (int): the second node
        k (int): the number of unit cells
        steps (int, optional): _description_. Defaults to 1.
    """
    t = c1/c2
    t = round(t,2)
    x = []
    y = []

    i = 0.87*resonance(c1*10**(-6),c2*10**(-6),10**(-5))[0]
    while i <= 1.1*resonance(c1*10**(-6),c2*10**(-6),10**(-5))[0] :
        x.append(i)
        y.append(impedance(n1,n2,SSH_sim(c1*10**(-6),c2*10**(-6),10**(-5),i,k)))
        i += 1
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=f't = {t}, n={n1} - {n2},s={steps}')
    

    plt.yscale('log')
    # Set the y-axis range (limits)
    plt.ylim(10**0, 10**6)  
    plt.xlabel('Angular Frequency(Hz)')
    plt.ylabel('Impedance(ohm)')
    print(resonance(c1*10**(-6),c2*10**(-6),10**(-5))[0])
    plt.title('Impedance vs Angular Frequency')
    plt.legend()
    plt.show()