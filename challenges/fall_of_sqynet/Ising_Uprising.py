import json
import pennylane as qml
import pennylane.numpy as np

def create_Hamiltonian(h):
    """
    Function in charge of generating the Hamiltonian of the statement.

    Args:
        h (float): magnetic field strength

    Returns:
        (qml.Hamiltonian): Hamiltonian of the statement associated to h
    """


    couplings = [-h]
    ops = [qml.PauliX(3)]

    for i in range(3):
        couplings = [-h] + couplings
        ops = [qml.PauliX(i)] + ops   
    
    for i in range(4):
        couplings = [-1] + couplings
        ops = [qml.PauliZ(i)@qml.PauliZ((i+1)%4)] + ops

    return qml.Hamiltonian(couplings,ops)



dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def model(params, H):
    """
    To implement VQE you need an ansatz for the candidate ground state!
    Define here the VQE ansatz in terms of some parameters (params) that
    create the candidate ground state. These parameters will
    be optimized later.

    Args:
        params (numpy.array): parameters to be used in the variational circuit
        H (qml.Hamiltonian): Hamiltonian used to calculate the expected value

    Returns:
        (float): Expected value with respect to the Hamiltonian H
    """


    for i in range(4):
        qml.RY(params[i], wires=i)
    
    for i in range(4):
        qml.CNOT([i,(i+1)%4])
    for i in range(4):
        qml.RZ(params[i+4], wires=i)
        
    
    for i in range(4):
        qml.RY(params[i+8], wires=i)
    
    for i in range(4):
        qml.CNOT([i,(i+1)%4])
    for i in range(4):
        qml.RZ(params[i+12], wires=i)

    return qml.expval(H)


def train(h):
    """
    In this function you must design a subroutine that returns the
    parameters that best approximate the ground state.

    Args:
        h (float): magnetic field strength

    Returns:
        (numpy.array): parameters that best approximate the ground state.
    """


    step_size = 0.02
    opt = qml.GradientDescentOptimizer(stepsize=0.2)
    #opt = qml.QNGOptimizer(stepsize=step_size, approx="block-diag")
    H =  create_Hamiltonian(h)
    theta = np.array([np.pi/2 for i in range(16)], requires_grad=True)
    # Put your code here #
    # store the values of the cost function
    energy = [model(theta, H)]

    # store the values of the circuit parameter
    params = [theta]

    max_iterations = 500
    conv_tol = 1e-06

    for n in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(model, theta, H=H)
        energy.append(model(theta, H))
        params.append(theta)

        conv = np.abs(energy[-1] - prev_energy)

        # if n % 2 == 0:
        print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

        if conv <= conv_tol:
            break
    return theta


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    params = train(ins)
    return str(model(params, create_Hamiltonian(ins)))


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-1
    ), "The expected value is not correct."


test_cases = [['1.0', '-5.226251859505506'], ['2.3', '-9.66382463698038']]

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