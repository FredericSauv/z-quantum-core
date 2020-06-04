from .interfaces.cost_function import CostFunction
from .interfaces.backend import QuantumBackend
from .circuit import build_ansatz_circuit, Circuit
from typing import Callable, Optional, Dict
import numpy as np
import copy
from openfermion import SymbolicOperator, QubitOperator
from .utils import (convert_array_to_dict, convert_dict_to_array,
                    ValueEstimate)
from qeopenfermion._io import convert_isingop_to_dict, convert_dict_to_isingop 

class BasicCostFunction(CostFunction):
    """
    Basic implementation of the CostFunction interface.
    It allows to pass any function (and gradient) when initialized.

    Args:
        function (Callable): function we want to use as our cost function. Should take a numpy array as input and return a single number.
        gradient_function (Callable): function used to calculate gradients. Optional.
        gradient_type (str): parameter indicating which type of gradient should be used.
        save_evaluation_history (bool): flag indicating whether we want to store the history of all the evaluations.
        epsilon (float): epsilon used for calculating gradient using finite difference method.

    Params:
        function (Callable): see Args
        gradient_function (Callable): see Args
        evaluations_history (list): List of the tuples (parameters, value) representing all the evaluation in a chronological order.
        gradient_type (str): see Args
        save_evaluation_history (bool): see Args
        epsilon (float): see Args

    """

    def __init__(self, function:Callable, 
                        gradient_function:Optional[Callable]=None, 
                        gradient_type:str='custom',
                        save_evaluation_history:bool=True, 
                        epsilon:float=1e-5):
        self.evaluations_history = []
        self.save_evaluation_history = save_evaluation_history
        self.gradient_type = gradient_type
        self.function = function
        self.gradient_function = gradient_function
        self.epsilon = epsilon

    def _evaluate(self, parameters:np.ndarray) -> float:
        """
        Evaluates the value of the cost function for given parameters.

        Args:
            parameters: parameters for which the evaluation should occur

        Returns:
            value: cost function value for given parameters, either int or float.
        """
        value = self.function(parameters)
        return value

    def get_gradient(self, parameters:np.ndarray) -> np.ndarray:
        """
        Evaluates the gradient of the cost function for given parameters.
        What method is used for calculating gradients is indicated by `self.gradient_type` field.

        Args:
            parameters: parameters for which we calculate the gradient.

        Returns:
            np.ndarray: gradient vector 
        """
        if self.gradient_type == 'custom':
            if self.gradient_function is None:
                raise Exception("Gradient function has not been provided.")
            else:
                return self.gradient_function(parameters)
        elif self.gradient_type == 'finite_difference':
            if self.gradient_function is not None:
                raise Warning("Using finite difference method for calculating gradient even though self.gradient_function is defined.")
            return self.get_gradients_finite_difference(parameters)
        else:
            raise Exception("Gradient type: %s is not supported", self.gradient_type)



class EvaluateOperatorCostFunction(CostFunction):
    """
    Cost function used for evaluating given operator using given ansatz.

    Args:
        target_operator (openfermion.QubitOperator): operator to be evaluated
        ansatz (dict): dictionary representing the ansatz
        backend (zquantum.core.interfaces.backend.QuantumBackend): backend used for evaluation
        gradient_type (str): parameter indicating which type of gradient should be used.
        save_evaluation_history (bool): flag indicating whether we want to store the history of all the evaluations.
        epsilon (float): epsilon used for calculating gradient using finite difference method.

    Params:
        target_operator (openfermion.QubitOperator): see Args
        ansatz (dict): see Args
        backend (zquantum.core.interfaces.backend.QuantumBackend): see Args
        evaluations_history (list): List of the tuples (parameters, value) representing all the evaluation in a chronological order.
        save_evaluation_history (bool): see Args
        gradient_type (str): see Args
        epsilon (float): see Args

    """

    def __init__(self, target_operator:SymbolicOperator, 
                        ansatz:Dict, 
                        backend:QuantumBackend, 
                        gradient_type:str='finite_difference',
                        save_evaluation_history:bool=True,
                        epsilon:float=1e-5):
        self.target_operator = target_operator
        self.ansatz = ansatz
        self.backend = backend
        self.evaluations_history = []
        self.save_evaluation_history = save_evaluation_history
        self.gradient_type = gradient_type
        self.epsilon = epsilon

    def _evaluate(self, parameters:np.ndarray) -> float:
        """
        Evaluates the value of the cost function for given parameters.

        Args:
            parameters: parameters for which the evaluation should occur.

        Returns:
            value: cost function value for given parameters, either int or float.
        """
        circuit = build_ansatz_circuit(self.ansatz, parameters)
        expectation_values = self.backend.get_expectation_values(circuit, self.target_operator)
        final_value = np.sum(expectation_values.values)
        return final_value

    def get_gradient(self, parameters:np.ndarray) -> np.ndarray:
        """
        Evaluates the gradient of the cost function for given parameters.
        What method is used for calculating gradients is indicated by `self.gradient_type` field.

        Args:
            parameters: parameters for which we calculate the gradient.

        Returns:
            np.ndarray: gradient vector 
        """
        if self.gradient_type == "finite_difference":
            return self.get_gradients_finite_difference(parameters)
        else:
            raise Exception("Gradient type: %s is not supported", self.gradient_type)


class OperatorFrame:
    """A class representing a single frame within a composite objective function.

    Broadly speaking, a "frame" is information needed in addition to the ansatz circuit itself 
    for evaluating the objective function of a variational quantum algorithm. This includes
    circuits that may be executed before and/or after the ansatz, as well as the operator that
    is measured at the end. Without loss of generality we can assume that the operator is 
    an Ising operator, since we can always rotate any other measurement context to the Z
    basis via a circuit that executed after the ansatz. But in practice the operator can
    be any sum of Paulis strings.
    
    Args:
        preprog (zmachine.core.circuit.Circuit): the circuit that is applied before the ansatz.
        postprog (zmachine.core.circuit.Circuit): the circuit that is applied after the ansatz.
        op (openfermion.op.IsingOperator): the operator represingting this frame's contribution to the objective function.

    Attributes:
        preprog (zmachine.core.circuit.Circuit): the circuit that is applied before the ansatz.
        postprog (zmachine.core.circuit.Circuit): the circuit that is applied after the ansatz.
        op (openfermion.op.IsingOperator): the operator represingting this frame's contribution to the objective function.

    """

    def __init__(self, preprog, postprog, op):
        """
        Args:
            preprog: Circuit (zmachine.core.circuit)
                The "preprogram" which is the quantum circuit executed before the ansatz
                circuit.
            postprog: Circuit (zmachine.core.circuit)
                The "postprogram" which is the quantum circuit executed after the ansatz
                circuit.
            op: QubitOperator (zmachine.core.qubitoperator)
                The operator to be measured with respect to the output state
        """
        if preprog is not None: 
            self.preprog = preprog
        else:
            self.preprog = Circuit()

        if postprog is not None: # The pyquil Program to be applied after the ansatz
            self.postprog = postprog 
        else:
            self.postprog = Circuit()
        self.op = op # The IsingOperator whose expectation value will be measured

    def get_qubit_op(self):
        """Get the frame's operator, represented as a qubit operator.

        This method looks at the postprog circuit to determine if each
        Pauli operator should be X, Y, or Z. An exception is raised if
        the postprog contains anything besides single-qubit
        context-selection gates.

        Returns:
            openfermion.ops.QubitOperator: a qubit operator representing the
                frame's operator 
        """
        
        context = self.get_measurement_context()
        qubit_op = QubitOperator()
        for term in self.op.terms:
            pauli_string = []
            for factor in term:
                pauli_op = context.get(factor[0], 'Z')
                pauli_string.append((factor[0], pauli_op))
            qubit_op.terms[tuple(pauli_string)] = self.op.terms[term]
        
        return qubit_op

    def get_measurement_context(self):
        """Get the measurement context determined by postprog.

        An exception is raised if the postprog contains anything besides single-qubit
        context-selection gates.

        Returns:
            dict: a dictionary whose keys are qubit indices and values are
                'X' or 'Y', representing measurement context. Any qubit whose
                index is not a key is being measured in the Z context.
        """

        context = {}
        for gate in self.postprog.gates:
            if gate.name == 'Ry' and np.isclose(gate.params[0], -np.pi/2):
                context[gate.qubits[0].index] = 'X'
            elif gate.name == 'Rx' and np.isclose(gate.params[0], np.pi/2):
                context[gate.qubits[0].index] = 'Y'
            else:
                raise RuntimeError('Gate {} does not appear to be a context selection gate'.format(gate))
        return context


class OperaterFramesCostFunction(CostFunction):
    """
    Cost function used for evaluating given operator using given ansatz.

    Args:
        target_operator (openfermion.QubitOperator): operator to be evaluated
        ansatz (dict): dictionary representing the ansatz
        backend (zquantum.core.interfaces.backend.QuantumBackend): backend used for evaluation
        gradient_type (str): parameter indicating which type of gradient should be used.
        save_evaluation_history (bool): flag indicating whether we want to store the history of all the evaluations.
        epsilon (float): epsilon used for calculating gradient using finite difference method.

    Params:
        target_operator (openfermion.QubitOperator): see Args
        ansatz (dict): see Args
        backend (zquantum.core.interfaces.backend.QuantumBackend): see Args
        evaluations_history (list): List of the tuples (parameters, value) representing all the evaluation in a chronological order.
        save_evaluation_history (bool): see Args
        gradient_type (str): see Args
        epsilon (float): see Args
    """
    # TODO must have ansatz positional argument in init
    # TODO create a positional argument gor list of OperatorFrames
    def __init__(self, cost_function='linear'):
        self.frames = [] # List of OperatorFrame objects
        self.constant = 0
        self.cost_function = cost_function

    def _evaluate(self, expectation_values):
        """Evaluate the objective function given a set of expectation values.

        Args: 
            expectation_values (zquantum.core.measurement.ExpectationValues): the expectation values of Pauli operators (not including coefficients)
        Returns:
            float: the value of the objective function
        """
       
        # 1. Build circuit from parameters and ansatz 
        # 2. We are going to loop through frames 
        #   b. create frame_circuit = frame.preprog + circuit+ frame.postprog  
        #   c. call get_expectation_values using backend
        #   d. add the values to a list : expecation_vales
        #   e. Put list in ExpectationValues(expectation_values)   
        # 3. Do the code below

        # Check that the number of expectation values is correct
        n_terms = sum([len(frame.op.terms) for frame in self.frames])
        if n_terms != expectation_values.values.shape[0]:
            raise(RuntimeError('Objective function has {} terms, but {} expectation values were provided.'.format(n_terms, expectation_values.values.shape[0])))

        total = self.constant
        term_index = 0
        for frame in self.frames:
            for term in frame.op.terms:
                total += frame.op.terms[term]*expectation_values.values[term_index]
                term_index += 1

        if expectation_values.covariances is not None:

            total_variance = 0.0   
            for n, frame in enumerate(self.frames):
                for index1, term1 in enumerate(frame.op.terms):
                    for index2, term2 in enumerate(frame.op.terms):
                        total_variance += (frame.op.terms[term1] * frame.op.terms[term2] * 
                                            expectation_values.covariances[n][index1, index2])
            precision = np.sqrt(total_variance)

        else:

            precision = None

        return ValueEstimate(total, precision)

    def to_dict(self):
        """Convert to a dictionary"""

        data = {'schema' : 'io-zapOS-v1alpha1-objective_function',
                'frames' : []}
        
        # Add the frames to the dict
        for frame in self.frames:
            frame_dict = {'preprog' : frame.preprog.to_dict(),
                        'postprog' : frame.postprog.to_dict(),
                        'op': convert_isingop_to_dict(frame.op)}
            data['frames'].append(frame_dict)

        # Add the constant to the dict
        data['constant'] = convert_array_to_dict(np.array(self.constant))

        # Specify the cost function
        data['cost_function'] = self.cost_function

        return data

    @classmethod
    def from_dict(cls, dictionary):
        """Create a LinearOperatorFramesCostFunction from a dictionary."""

        objfun = cls()

        # Load the frames
        for frame_dict in dictionary['frames']:
            if frame_dict.get('preprog'):
                preprog = Circuit.from_dict(frame_dict['preprog'])
            else:
                preprog = None
            if frame_dict.get('postprog'):
                postprog = Circuit.from_dict(frame_dict['postprog'])
            else:
                postprog = None
            op = convert_dict_to_isingop(frame_dict['op'])
            objfun.frames.append(OperatorFrame(
                preprog,
                postprog,
                op
            ))

        # Load the constant (should be loaded as just a singular float)
        objfun.constant = convert_dict_to_array(dictionary['constant']).item()
        
        return objfun

    
