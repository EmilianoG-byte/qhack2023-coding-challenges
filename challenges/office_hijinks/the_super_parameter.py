import json
import pennylane as qml
import pennylane.numpy as np

dev = qml.device("default.qubit", wires=3)


@qml.qnode(dev)
def model(alpha):
    """In this qnode you will define your model in such a way that there is a single 
    parameter alpha which returns each of the basic states.

    Args:
        alpha (float): The only parameter of the model.

    Returns:
        (numpy.tensor): The probability vector of the resulting quantum state.
    """
    #get nearest state position:
    statenumber=int(np.floor(alpha))
    pos=0.5+statenumber
    distance=abs(alpha-pos)
    rotation_strength=2*np.pi*(0.5-distance)
    #get active wires:
    active_wires=list(np.base_repr(statenumber,2).zfill(3))
    if active_wires[0]=="1":
        qml.RY(rotation_strength,wires=0)
    if active_wires[1]=="1":
        qml.RY(rotation_strength,wires=1)
    if active_wires[2]=="1":
        qml.RY(rotation_strength,wires=2)
    return qml.probs(wires=range(3))
    
def generate_coefficients():
    """This function must return a list of 8 different values of the parameter that
    generate the states 000, 001, 010, ..., 111, respectively, with your ansatz.
    
    Returns:
        (list(int)): A list of eight real numbers.
    """
    return([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
    # Put your code here #


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    return None

def check(solution_output, expected_output: str) -> None:
    coefs = generate_coefficients()
    output = np.array([model(c) for c in coefs])
    epsilon = 0.001

    for i in range(len(coefs)):
        assert np.isclose(output[i][i], 1)

    def is_continuous(function, point):
        limit = calculate_limit(function, point)

        if limit is not None and sum(abs(limit - function(point))) < epsilon:
            return True
        else:
            return False

    def is_continuous_in_interval(function, interval):
        for point in interval:
            if not is_continuous(function, point):
                return False
        return True

    def calculate_limit(function, point):
        x_values = [point - epsilon, point, point + epsilon]
        y_values = [function(x) for x in x_values]
        average = sum(y_values) / len(y_values)

        return average

    assert is_continuous_in_interval(model, np.arange(0,10,0.001))

    for coef in coefs:
        assert coef >= 0 and coef <= 10


test_cases = [['No input', 'No output']]

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