import unittest
import numpy as np
from .cost_function import (BasicCostFunction, EvaluateOperatorCostFunction,
                OperatorFrame, OperatorFramesCostFunction, evaluate_framescostfunction,
                evaluate_framescostfunction_for_expectation_values_history, 
                get_framescostfunction_from_qubit_operator)
from .interfaces.mock_objects import MockQuantumSimulator
from .interfaces.cost_function_test import CostFunctionTests
from openfermion import QubitOperator
from .measurement import (is_comeasureable, group_comeasureable_terms_greedy, 
                        ExpectationValues)
from .core.circuit import Circuit
import openfermion
import pyquil

class TestBasicCostFunction(unittest.TestCase, CostFunctionTests):

    def setUp(self):
        # Setting up inherited tests
        function = np.sum
        cost_function = BasicCostFunction(function)
        self.cost_functions = [cost_function]
        self.params_sizes = [2]

    def test_evaluate(self):
        # Given
        function = np.sum
        params_1 = np.array([1,2,3])
        params_2 = np.array([1,2,3,4])
        target_value_1 = 6
        target_value_2 = 10
        cost_function = BasicCostFunction(function)

        # When
        value_1 = cost_function.evaluate(params_1)
        value_2 = cost_function.evaluate(params_2)
        history = cost_function.evaluations_history

        # Then
        self.assertEqual(value_1, target_value_1)
        self.assertEqual(value_2, target_value_2)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['value'], target_value_1)
        np.testing.assert_array_equal(history[0]['params'], params_1)
        self.assertEqual(history[1]['value'], target_value_2)
        np.testing.assert_array_equal(history[1]['params'], params_2)

    def test_get_gradient(self):
        # Given
        function = np.sum
        def gradient_function(params):
            return np.ones(params.size)
        params_1 = np.array([1,2,3])
        params_2 = np.array([1,2,3,4])
        target_gradient_value_1 = np.array([1,1,1])
        target_gradient_value_2 = np.array([1,1,1,1])
        cost_function = BasicCostFunction(function, gradient_function=gradient_function, gradient_type='custom')

        # When
        gradient_value_1 = cost_function.get_gradient(params_1)
        gradient_value_2 = cost_function.get_gradient(params_2)

        # Then
        np.testing.assert_array_equal(gradient_value_1, target_gradient_value_1)
        np.testing.assert_array_equal(gradient_value_2, target_gradient_value_2)

    def test_get_finite_difference_gradient(self):
        # Given
        function = np.sum
        params_1 = np.array([1,2,3])
        params_2 = np.array([1,2,3,4])
        target_gradient_value_1 = np.array([1,1,1])
        target_gradient_value_2 = np.array([1,1,1,1])
        cost_function = BasicCostFunction(function, gradient_type='finite_difference')

        # When
        gradient_value_1 = cost_function.get_gradient(params_1)
        gradient_value_2 = cost_function.get_gradient(params_2)

        # Then
        np.testing.assert_almost_equal(gradient_value_1, target_gradient_value_1)
        np.testing.assert_almost_equal(gradient_value_2, target_gradient_value_2)

class TestEvaluateOperatorCostFunction(unittest.TestCase, CostFunctionTests):

    def setUp(self):
        target_operator = QubitOperator('Z0')
        ansatz = {'ansatz_module': 'zquantum.core.interfaces.mock_objects', 'ansatz_func': 'mock_ansatz', 'ansatz_kwargs': {}, 'n_params': [1]}
        backend = MockQuantumSimulator()
        self.single_term_op_cost_function = EvaluateOperatorCostFunction(target_operator, ansatz, backend)

        # Setting up inherited tests
        self.cost_functions = [self.single_term_op_cost_function]
        self.params_sizes = [2]

    def test_evaluate(self):
        # Given
        params = np.array([1, 2])

        # When
        value_1 = self.single_term_op_cost_function.evaluate(params)
        value_2 = self.single_term_op_cost_function.evaluate(params)
        history = self.single_term_op_cost_function.evaluations_history

        # Then
        self.assertGreaterEqual(value_1, 0)
        self.assertLessEqual(value_1, 1)
        self.assertGreaterEqual(value_2, 0)
        self.assertLessEqual(value_2, 1)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['value'], value_1)
        np.testing.assert_array_equal(history[0]['params'], params)
        self.assertEqual(history[1]['value'], value_2)
        np.testing.assert_array_equal(history[1]['params'], params)

# class TestOperatorFramesCostFunction(unittest.TestCase):

#     def setUp(self):
#        pass
#     def test_evaluate_framescostfunction_for_expectation_values_history(self):
#         op = QubitOperator('2.0 [] + [Y0] + [Z1] + [X0 Y1]')
#         frames_cost_function, _  = get_framescostfunction_from_qubit_operator(op)
        

#         expvals_history = [ExpectationValues(np.array([1., 0.5, -1.0]), 
#                           covariances = [np.array([[0.1]]), np.array([[0.1]]), np.array([[0.1]])]),
#                            ExpectationValues(np.array([0.9, 0.4, -0.9]))]
        
#         value_estimate_history = evaluate_framescostfunction_for_expectation_values_history(
#             frames_cost_function, expvals_history)

#         self.assertEqual(value_estimate_history[0].value, 2.5)
#         self.assertEqual(value_estimate_history[0].precision, 0.5477225575051662, 7)
#         self.assertEqual(value_estimate_history[1].value, 2.4)
#         self.assertEqual(value_estimate_history[1].precision, None)

#     def test_get_framescostfunction_from_qubit_operator(self):

#         # No grouping
#         op1 = QubitOperator('2.0 [] + [X0] + [Z1] + [X0 Y1]')
#         objfun, reordered_op = get_framescostfunction_from_qubit_operator(op1)
#         self.assertAlmostEqual(objfun.constant, 2.)
#         self.assertEqual(op1, reordered_op)

#         # Greedy grouping
#         op2 = QubitOperator('[Z0 Z1] + [X0 X1] + [Z0] + [X0]')
#         objfun, reordered_op = get_framescostfunction_from_qubit_operator(op2, 'greedy')

#         target_op_frame1 = openfermion.ops.IsingOperator('[Z0] + [Z0 Z1]')
#         target_op_frame2 = openfermion.ops.IsingOperator('[Z0] + [Z0 Z1]')
#         target_postprog_frame1 = Circuit()
#         target_postprog_frame2 = Circuit(pyquil.gates.RY(-np.pi / 2, 0)) + Circuit(pyquil.gates.RY(-np.pi / 2, 1))

#         self.assertEqual(target_op_frame1, objfun.frames[0].op)
#         self.assertEqual(target_op_frame2, objfun.frames[1].op)
#         self.assertEqual(target_postprog_frame1, objfun.frames[0].postprog)
#         self.assertEqual(target_postprog_frame2, objfun.frames[1].postprog)
#         self.assertAlmostEqual(objfun.constant, 0.)
#         self.assertEqual(len(objfun.frames), 2)
#         self.assertEqual(op2, reordered_op)

#         # All-in-one without constant term
#         op3 = QubitOperator.from_coeffs_and_labels([1,1,1,0.1,0.1,0.1],[[3,3,0],[3,0,3],[0,3,3],[3,0,0],[0,3,0],[0,0,3]])
#         objfun, qubitop = get_framescostfunction_from_qubit_operator(op3, 'all-in-one')
#         self.assertEqual(op3, qubitop)

#         # All-in-one with constant term
#         op4 = op3 + QubitOperator((), 3)
#         objfun, qubitop = get_framescostfunction_from_qubit_operator(op4, 'all-in-one')
#         self.assertEqual(op3, objfun.frames[0].op)
#         self.assertEqual(len(objfun.frames), 1)
#         self.assertAlmostEqual(objfun.constant, 3)
#         self.assertEqual(op4, qubitop)

#         # Check that the terms have been reordered correctly
#         self.assertEqual(list(reordered_op.terms)[0],
#                          ((0, 'Z'), (1, 'Z')))
#         self.assertEqual(list(reordered_op.terms)[1],
#                          ((0, 'Z'),))
#         self.assertEqual(list(reordered_op.terms)[2],
#                          ((0, 'X'), (1, 'X')))
#         self.assertEqual(list(reordered_op.terms)[3],
#                          ((0, 'X'),))

