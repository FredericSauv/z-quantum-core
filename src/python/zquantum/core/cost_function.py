from .interfaces.cost_function import CostFunction
from .interfaces.backend import QuantumBackend
from .circuit import build_ansatz_circuit, Circuit
from typing import Callable, Optional, Dict, List
import numpy as np
import copy
import openfermion
import pyquil
import rapidjson as json
from zquantum.core.measurement import ExpectationValues, group_comeasureable_terms_greedy
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


class OperatorFramesCostFunction(CostFunction):
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
    def __init__(self, 
                    circuit= None,
                    cost_function='linear'):
        self.frames = [] # List of OperatorFrame objects
        self.constant = 0
        self.cost_function = cost_function
        self.ansatz = None
        self.backend = None
        self.circuit = None
        self.expectation_values = None

    def _evaluate(self, parameters:np.ndarray=np.zeros((0,)))-> float:
        """Evaluate the objective function given a set of expectation values.

        Args: 
            expectation_values (zquantum.core.measurement.ExpectationValues): the expectation values of Pauli operators (not including coefficients)
        Returns:
            float: the value of the objective function
        """
        if self.ansatz is not None and self.circuit is None :
            assert(len(parameters) > 0)
            self.circuit = build_ansatz_circuit(self.ansatz, parameters)
        expectation_values = np.zeros((0,))  # 1D array of length zero
        for frame in self.frames:
            frame_circuit = frame.preprog + self.circuit + frame.postprog
            frame_expvals = self.backend.get_expectation_values(frame_circuit, frame.op)
            expectation_values = np.append(expectation_values, frame_expvals)
        
        self.expectation_values = ExpectationValues(expectation_values)
        
        # Check that the number of expectation values is correct
        n_terms = sum([len(frame.op.terms) for frame in self.frames])
        if n_terms != self.expectation_values.values.shape[0]:
            raise(RuntimeError('Objective function has {} terms, but {} expectation values were provided.'.format(n_terms, expectation_values.values.shape[0])))

        total = self.constant
        term_index = 0
        for frame in self.frames:
            for term in frame.op.terms:
                total += frame.op.terms[term]*self.expectation_values.values[term_index]
                term_index += 1

        if self.expectation_values.covariances is not None:

            total_variance = 0.0   
            for n, frame in enumerate(self.frames):
                for index1, term1 in enumerate(frame.op.terms):
                    for index2, term2 in enumerate(frame.op.terms):
                        total_variance += (frame.op.terms[term1] * frame.op.terms[term2] * 
                                            self.expectation_values.covariances[n][index1, index2])
            precision = np.sqrt(total_variance)

        else:

            precision = None

        return ValueEstimate(total, precision)

    def to_dict(self):
        """Convert to a dictionary"""

        data = {'schema' : 'io-zapOS-v1alpha1-framescostfunction',
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

        framescostfunction = cls()

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
            framescostfunction.frames.append(OperatorFrame(
                preprog,
                postprog,
                op
            ))

        # Load the constant (should be loaded as just a singular float)
        framescostfunction.constant = convert_dict_to_array(dictionary['constant']).item()
        
        return framescostfunction

    
def evaluate_framescostfunction(framescostfunction, expectation_values):
    """Evaluate an objective function with respect to a set of expectation values.

    Args:p
        framescostfunction (zmachine.core.objective.ObjectiveFunction)
        expectation_values (zmachine.core.measurement.ExpectationValues)
    
    Returns:
        dict: the estimated value and variance of the objective function
    """
    return framescostfunction._evaluate(expectation_values)

def get_framescostfunction_from_qubit_operator(qubit_operator, grouping_strategy='individual',
                                               sort_terms = False):
    """Get an objective function representing the minimization of the expectation value
    of an operator.

    Args:
        qubit_operator (openfermion.ops.QubitOperator): the qubit operator whose expectation value is to be minimized
        grouping_strategy (str): the strategy for grouping Pauli strings. 
            Possible values include
                'individual': each term is a separate frame
                'greedy': grouping co-measurable terms using greedy approach
                'all-in-one': put all terms in one frame
        sort_terms (bool): whether to sort terms by the absolute value of the coefficient when grouping
    
    Returns:
        tuple: a two-element tuple containing 
        - **framescostfunction** (**zmachine.core.objective.ObjectiveFunction**): the objective 
            function with context-selection gates added to each frame
        - **reordered_qubit_operator** (**openfermion.ops.QubitOperator**): a qubit operator whose terms have 
            been reordered to correspond to the order terms in the objective function
    """

    framescostfunction = OperatorFramesCostFunction()

    # Check if there is a constant term
    if qubit_operator.terms.get(()):
        framescostfunction.constant = qubit_operator.terms[()]

    if grouping_strategy == 'individual':
        # Create a frame for each term in the qubit operator
        for term in qubit_operator.terms:
            if len(term) > 0:
                context_selection_circuit = Circuit()
                operator = openfermion.ops.IsingOperator(())
                for factor in term:
                    if factor[1] == 'X':
                        context_selection_circuit += Circuit(pyquil.gates.RY(-np.pi / 2, factor[0]))
                    elif factor[1] == 'Y':
                        context_selection_circuit += Circuit(pyquil.gates.RX(np.pi / 2, factor[0]))
                    operator *= openfermion.ops.IsingOperator((factor[0], 'Z'))
                operator *= qubit_operator.terms[term]
                framescostfunction.frames.append(OperatorFrame(Circuit(), context_selection_circuit, operator))    
        return (framescostfunction, qubit_operator)

    elif grouping_strategy == 'greedy' or grouping_strategy == 'greedy-with-duplicates':

        # Using a greedy grouping algorithm
        duplicate_terms = True if grouping_strategy == 'greedy-with-duplicates' else False
        groups = group_comeasureable_terms_greedy(qubit_operator, duplicate_terms=duplicate_terms,
                                                  sort_terms = sort_terms)
        return get_framescostfunction_from_list_of_qubit_operators(groups, framescostfunction, qubit_operator)
    
    elif grouping_strategy == 'all-in-one':
        # No grouping - all terms are in the same basis (as in e.g. QAOA)

        # Remove the constant term
        qubit_operator_without_constant = copy.deepcopy(qubit_operator)
        if qubit_operator.terms.get(()) is not None:
            qubit_operator_without_constant.terms.pop(())
            print('hello')
        print(qubit_operator_without_constant)
        # Check if all terms are actually in the same basis
        list_symbols = []
        for key in qubit_operator.terms.keys():
            for term in key:
                list_symbols.append(term[1])
        if len(set(list_symbols)) > 1:
            raise Exception("Input operator contains non-commuting sets of terms - unable to group under a single basis")
        framescostfunction.frames.append(OperatorFrame(Circuit(), Circuit(), qubit_operator_without_constant))

        return (framescostfunction, qubit_operator)

    else:
        raise RuntimeError('Grouping method {} not supported'.format(grouping_strategy))

def evaluate_framescostfunction_for_expectation_values_history(framescostfunction,
                                                               expectation_values_history):
    """Convert an expectation value history (i.e. a list of lists of estimators)
        to the corresponding ValueEstimate history.

    Args:
        expectation_values_history (list of lists of zmachine.core.measurement.ExpectationValues): 
        Contains the history of expectation values

    Returns:
        A list of zmachine.core.utils.ValueEstimate objects: Contains the estimate of the 
        objective function value (as well as its precision) at each inference round. 
    """
    value_estimate_history = []

    for expvals in expectation_values_history:
        value_estimate_object = framescostfunction._evaluate(expvals)
        value_estimate_history.append(value_estimate_object)
    
    return value_estimate_history

def get_framescostfunction_from_list_of_qubit_operators(groups, framescostfunction, qubit_operator):
    """
    Generates and objective function and a reordered qubit operator
    from a list of qubit operators.
        Args:
            groups (list of zmachine.core.qubitoperator.QubitOperator): list
                of qubit operators corresponding to comeasurable sets
            framescostfunction (zmachine.core.objective.ObjectiveFunction): objective function
                to be filled.
            qubit_operator (zmachine.core.qubitOperator): the original qubit
                operator for which the grouping is being done.
        
    Returns:
        tuple: a two-element tuple containing 
        - **framescostfunction** (**zmachine.core.objective.ObjectiveFunction**): the objective 
            function with context-selection gates added to each frame
        - **reordered_qubit_operator** (**openfermion.ops.QubitOperator**): a qubit operator whose terms have 
            been reordered to correspond to the order terms in the objective function
    """
    for group in groups:
        operator = openfermion.ops.IsingOperator()
        context_map = {} # Map between qubit index and measurement context ('X', 'Y', or 'Z')
            
        for term in group.terms:
            ising_term = openfermion.ops.IsingOperator(())
            if len(term) > 0:
                for factor in term:
                    context_map[factor[0]] = factor[1]
                    ising_term *= openfermion.ops.IsingOperator((factor[0], 'Z'))
                ising_term *= group.terms[term]
                operator += ising_term

        # Create context selection circuit
        context_selection_circuit = Circuit()
        for qubit in context_map:
            if context_map[qubit] == 'X':
                context_selection_circuit += Circuit(pyquil.gates.RY(-np.pi / 2, qubit))
            elif context_map[qubit] == 'Y':
                context_selection_circuit += Circuit(pyquil.gates.RX(np.pi / 2, qubit))

        framescostfunction.frames.append(OperatorFrame(Circuit(), context_selection_circuit, operator))
        
    # Construct the reordered qubit operator. Note that we are taking advantage of the fact that
    # the QubitOperator class preserves the order of terms when they are added
    reordered_qubit_operator = openfermion.ops.QubitOperator()
    # Add identity back in reordered operator
    if qubit_operator.terms.get(()):
        reordered_qubit_operator += openfermion.ops.QubitOperator((),qubit_operator.terms[()])
    for group in groups:
        for term in group.terms:
            reordered_qubit_operator += openfermion.ops.QubitOperator(term, group.terms[term])

    return (framescostfunction, reordered_qubit_operator)

def get_framescostfunction_from_qubit_qubit_operator(qubit_operator, grouping_strategy='individual',
                                               sort_terms = False):
    """Get an objective function representing the minimization of the expectation value
    of an operator.

    Args:
        qubit_operator (openfermion.ops.QubitOperator): the qubit operator whose expectation value is to be minimized
        grouping_strategy (str): the strategy for grouping Pauli strings. 
            Possible values include
                'individual': each term is a separate frame
                'greedy': grouping co-measurable terms using greedy approach
                'all-in-one': put all terms in one frame
        sort_terms (bool): whether to sort terms by the absolute value of the coefficient when grouping
    
    Returns:
        tuple: a two-element tuple containing 
        - **objective_function** (**zmachine.core.objective.ObjectiveFunction**): the objective 
            function with context-selection gates added to each frame
        - **reordered_qubit_operator** (**openfermion.ops.QubitOperator**): a qubit operator whose terms have 
            been reordered to correspond to the order terms in the objective function
    """

    objfun =OperatorFramesCostFunction()

    # Check if there is a constant term
    if qubit_operator.terms.get(()):
        objfun.constant = qubit_operator.terms[()]

    if grouping_strategy == 'individual':
        # Create a frame for each term in the qubit operator
        for term in qubit_operator.terms:
            if len(term) > 0:
                context_selection_circuit = Circuit()
                operator = openfermion.ops.IsingOperator(())
                for factor in term:
                    if factor[1] == 'X':
                        context_selection_circuit += Circuit(pyquil.gates.RY(-np.pi / 2, factor[0]))
                    elif factor[1] == 'Y':
                        context_selection_circuit += Circuit(pyquil.gates.RX(np.pi / 2, factor[0]))
                    operator *= openfermion.ops.IsingOperator((factor[0], 'Z'))
                operator *= qubit_operator.terms[term]
                objfun.frames.append(OperatorFrame(Circuit(), context_selection_circuit, operator))    
        return (objfun, qubit_operator)

    elif grouping_strategy == 'greedy' or grouping_strategy == 'greedy-with-duplicates':

        # Using a greedy grouping algorithm
        duplicate_terms = True if grouping_strategy == 'greedy-with-duplicates' else False
        groups = group_comeasureable_terms_greedy(qubit_operator, duplicate_terms=duplicate_terms,
                                                  sort_terms = sort_terms)
        return get_framescostfunction_from_list_of_qubit_operators(groups, objfun, qubit_operator)
    
    elif grouping_strategy == 'all-in-one':
        # No grouping - all terms are in the same basis (as in e.g. QAOA)

        # Remove the constant term
        qubit_operator_without_constant = copy.deepcopy(qubit_operator)
        if qubit_operator.terms.get(()) is not None:
            qubit_operator_without_constant.terms.pop(())
            print('hello')
        print(qubit_operator_without_constant)
        # Check if all terms are actually in the same basis
        list_symbols = []
        for key in qubit_operator.terms.keys():
            for term in key:
                list_symbols.append(term[1])
        if len(set(list_symbols)) > 1:
            raise Exception("Input operator contains non-commuting sets of terms - unable to group under a single basis")
        objfun.frames.append(OperatorFrame(Circuit(), Circuit(), qubit_operator_without_constant))

        return (objfun, qubit_operator)

    else:
        raise RuntimeError('Grouping method {} not supported'.format(grouping_strategy))

