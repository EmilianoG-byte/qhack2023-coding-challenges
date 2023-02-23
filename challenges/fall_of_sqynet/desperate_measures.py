import json
import pennylane as qml
import pennylane.numpy as np

num_wires = 4
dev = qml.device("default.mixed", wires=num_wires)

@qml.qnode(dev)
def heisenberg_trotter(couplings, p, time, depth):
    """This QNode returns the final state of the spin chain after evolution for a time t, 
    under the Trotter approximation of the exponential of the Heisenberg Hamiltonian.
    
    Args:
        couplings (list(float)): 
            An array of length 4 that contains the coupling constants and the magnetic field 
            strength, in the order [J_x, J_y, J_z, h].
        p (float): The depolarization probability after each CNOT gate.
        depth (int): The Trotterization depth.
        time (float): Time during which the state evolves

    Returns:
        (numpy.tensor): The evolved quantum state.
    """


    # Put your code here #
    def XX(i):
        # XX
        qml.RY(0,wires=1)
        qml.CNOT([i,(i+1)%4][::-1])
        qml.DepolarizingChannel(p, wires=i)
        qml.RX(-2*couplings[0]*time/depth, wires=(i+1)%4)
        qml.CNOT([i,(i+1)%4][::-1])
        qml.DepolarizingChannel(p, wires=i)
    def YY(i):
        #YY
        qml.RZ(np.pi/2, wires=i)
        qml.RZ(np.pi/2, wires=(i+1)%4)
        qml.CNOT([i,(i+1)%4][::-1])
        qml.DepolarizingChannel(p, wires=i)
        qml.RX(-2*couplings[1]*time/depth, wires=(i+1)%4)
        qml.CNOT([i,(i+1)%4][::-1])
        qml.DepolarizingChannel(p, wires=i)
        qml.RZ(-np.pi/2, wires=i)
        qml.RZ(-np.pi/2, wires=(i+1)%4)
    def ZZ(i):
        #ZZ
        qml.CNOT([i,(i+1)%4])
        qml.DepolarizingChannel(p, wires=(i+1)%4)
        qml.RZ(-2*couplings[2]*time/depth, wires=(i+1)%4)
        qml.CNOT([i,(i+1)%4])
        qml.DepolarizingChannel(p, wires=(i+1)%4)
    def magnet(i):
        #magnetic field
        qml.RX(-2*couplings[3]*time/depth, wires=i)
    for j in range(depth):
        #first the XX, YY, ZZ part:
        XX(0)
        XX(1)
        XX(2)
        XX(3)
        YY(0)
        YY(1)
        YY(2)
        YY(3)
        ZZ(0)
        ZZ(1)
        ZZ(2)
        ZZ(3)
        magnet(0)
        magnet(1)
        magnet(2)
        magnet(3)

    return qml.state()



def calculate_fidelity(couplings, p, time, depth):
    """This function returns the fidelity between the final states of the noisy and
    noiseless Trotterizations of the Heisenberg models, using only CNOT and rotation gates

    Args:
        couplings (list(float)): 
            A list with the J_x, J_y, J_z and h parameters in the Heisenberg Hamiltonian, as
            defined in the problem statement.
        p (float): The depolarization probability of the depolarization gate that acts on the
                   target qubit of each CNOT gate.
        time (float): The period of time evolution simulated by the Trotterization.
        depth (int): The Trotterization depth.

    Returns:
        (float): Fidelity between final states of the noisy and noiseless Trotterizations
    """
    return qml.math.fidelity(heisenberg_trotter(couplings,0,time, depth),heisenberg_trotter(couplings,p,time,depth))


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:

    ins = json.loads(test_case_input)
    output =calculate_fidelity(*ins)

    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    """
    Compare solution with expected.

    Args:
            solution_output: The output from an evaluated solution. Will be
            the same type as returned.
            expected_output: The correct result for the test case.

    Raises: 
            ``AssertionError`` if the solution output is incorrect in any way.
            
    """
    def create_hamiltonian(params):

        couplings = [-params[-1]]
        ops = [qml.PauliX(3)]

        for i in range(3):

            couplings = [-params[-1]] + couplings
            ops = [qml.PauliX(i)] + ops        

        for i in range(4):

            couplings = [-params[-2]] + couplings
            ops = [qml.PauliZ(i)@qml.PauliZ((i+1)%4)] + ops

        for i in range(4):

            couplings = [-params[-3]] + couplings
            ops = [qml.PauliY(i)@qml.PauliY((i+1)%4)] + ops

        for i in range(4):

            couplings = [-params[0]] + couplings
            ops = [qml.PauliX(i)@qml.PauliX((i+1)%4)] + ops    

        return qml.Hamiltonian(couplings,ops)

    @qml.qnode(dev)
    def evolve(params, time, depth):

        qml.ApproxTimeEvolution(create_hamiltonian(params), time, depth)

        return qml.state()
    
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    
    tape = heisenberg_trotter.qtape
    names = [op.name for op in tape.operations]
    
    random_params = np.random.uniform(low = 0.8, high = 3.0, size = (4,) )
    
    assert qml.math.fidelity(heisenberg_trotter(random_params,0,1,2),evolve(random_params,1,2)) >= 1, "Your circuit does not Trotterize the Heisenberg Model"
    
    assert names.count('ApproxTimeEvolution') == 0, "Your circuit must not use the built-in PennyLane Trotterization"
     
    assert set(names) == {'DepolarizingChannel', 'RX', 'RY', 'RZ', 'CNOT'}, "Your circuit must only use RX, RY, RZ, CNOT, and depolarizing gates (don't use qml.Rot or Paulis)"
    
    assert solution_output >= expected_output-0.005, "Your fidelity is not high enough. You may be using more CNOT gates than needed"


test_cases = [['[[1,2,1,0.3],0.05,2.5,1]', '0.33723981123369573'], ['[[1,3,2,0.3],0.05,2.5,2]', '0.15411351752086694']]

for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")