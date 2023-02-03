import torch
import numpy as np
from quantum_error_mitigation.model.NN_Base_Class.Neural_Network_Base_Class import Neural_Network_Base_Class


def test_make_probability_for_each_state_features_returns_one_at_the_correct_position_for_only_one_entry():
    nnbc_obj = Neural_Network_Base_Class()
    n_qubits = 3
    n_samples = 1
    output_size = "full"
    training_solutions = np.array([0, 0, 1, 1., 4]).reshape((1, 1, n_qubits+2))
    training_inputs = np.array([0, 1, 0, 1., 2]).reshape((1, 1, n_qubits+2))
    X, _ = nnbc_obj._make_probability_for_each_state_features(training_inputs, training_solutions, n_qubits, n_samples, output_size)
    X_true = np.array([0, 0, 1., 0, 0, 0, 0, 0])
    assert np.allclose(X, X_true)


def test_make_probability_for_each_state_features_returns_correct_X_value_for_multiple_entries():
    nnbc_obj = Neural_Network_Base_Class()
    n_qubits = 3
    n_samples = 1
    output_size = "full"
    training_solutions = np.array([[0, 0, 1, 0.8, 4], [1, 0, 1, 0.2, 5]]).reshape((1, 2, n_qubits+2))
    training_inputs = np.array([[0, 1, 0, 0.6, 2], [1, 1, 0, 0.4, 3]]).reshape((1, 2, n_qubits+2))
    X, _ = nnbc_obj._make_probability_for_each_state_features(training_inputs, training_solutions, n_qubits, n_samples, output_size)
    X_true = np.array([0, 0, 0.6, 0.4, 0, 0, 0, 0])
    assert np.allclose(X, X_true)


def test_make_probability_for_each_state_features_returns_correct_y_value_for_multiple_entries():
    nnbc_obj = Neural_Network_Base_Class()
    n_qubits = 3
    n_samples = 1
    output_size = "full"
    training_solutions = np.array([[0, 0, 1, 0.8, 4], [1, 0, 1, 0.2, 5]]).reshape((1, 2, n_qubits+2))
    training_inputs = np.array([[0, 1, 0, 0.6, 2], [1, 1, 0, 0.4, 3]]).reshape((1, 2, n_qubits+2))
    _, y = nnbc_obj._make_probability_for_each_state_features(training_inputs, training_solutions, n_qubits, n_samples, output_size)
    y_true = np.array([0, 0, 0, 0, 0.8, 0.2, 0, 0])
    assert np.allclose(y, y_true)


def test_make_k_most_frequent_state_features_returns_correct_code_for_only_one_entry():
    nnbc_obj = Neural_Network_Base_Class()
    n_qubits = 3
    n_samples = 1
    output_size = "subregisters"
    k = 1
    training_solutions = np.array([0, 0, 1, 1., 4]).reshape((1, 1, n_qubits+2))
    training_inputs = np.array([0, 1, 0, 1., 2]).reshape((1, 1, n_qubits+2))
    X, _ = nnbc_obj._make_k_most_frequent_state_features(training_inputs, training_solutions, n_qubits, n_samples, output_size, k)
    X_true = np.array([-1., 1., -1.])
    assert np.allclose(X, X_true)


def test_make_k_most_frequent_state_features_returns_correct_X_value_for_multiple_entries():
    nnbc_obj = Neural_Network_Base_Class()
    n_qubits = 3
    n_samples = 1
    output_size = "subregisters"
    k = 2
    training_solutions = np.array([[0, 0, 1, 0.8, 4], [1, 0, 1, 0.2, 5]]).reshape((1, 2, n_qubits+2))
    training_inputs = np.array([[0, 1, 0, 0.6, 2], [1, 1, 0, 0.4, 3]]).reshape((1, 2, n_qubits+2))
    X, _ = nnbc_obj._make_k_most_frequent_state_features(training_inputs, training_solutions, n_qubits, n_samples, output_size, k)
    X_true = np.array([-0.6, 0.6, -0.6, 0.4, 0.4, -0.4])
    assert np.allclose(X, X_true)

def test_make_k_most_frequent_state_features_returns_correct_y_value_for_multiple_entries():
    nnbc_obj = Neural_Network_Base_Class()
    n_qubits = 3
    n_samples = 1
    k = 2
    output_size = "subregisters"
    training_solutions = np.array([[0, 0, 1, 0.8, 4], [1, 0, 1, 0.2, 5]]).reshape((1, 2, n_qubits+2))
    training_inputs = np.array([[0, 1, 0, 0.6, 2], [1, 1, 0, 0.4, 3]]).reshape((1, 2, n_qubits+2))
    _, y = nnbc_obj._make_k_most_frequent_state_features(training_inputs, training_solutions, n_qubits, n_samples, output_size, k)
    y_true = np.array([-0.8, -0.8, 0.8, 0.2, -0.2, 0.2])
    assert np.allclose(y, y_true)

def test_get_sample_returns_correct_probability():
    nnbc_obj = Neural_Network_Base_Class()
    n_qubits = 3
    sample_idx = 0
    subindices = [0,1,2]
    training_inputs = np.array([[0, 1, 0, 0.6, 2], [1, 1, 0, 0.4, 3]]).reshape((1, 2, n_qubits+2))
    state_idx = 0
    prob_0, _ = nnbc_obj._get_sample(training_inputs, sample_idx, state_idx, subindices)
    state_idx = 1
    prob_1, _ = nnbc_obj._get_sample(training_inputs, sample_idx, state_idx, subindices)
    np.testing.assert_almost_equal(prob_0, 0.6)
    np.testing.assert_almost_equal(prob_1, 0.4)


def test_get_sample_returns_correct_integer_value_of_registers():
    nnbc_obj = Neural_Network_Base_Class()
    n_qubits = 3
    sample_idx = 0
    subindices = [0,1,2]
    training_inputs = np.array([[0, 1, 0, 0.6, 2], [1, 1, 0, 0.4, 3]]).reshape((1, 2, n_qubits+2))
    state_idx = 0
    _, int_value_0 = nnbc_obj._get_sample(training_inputs, sample_idx, state_idx, subindices)
    state_idx = 1
    _, int_value_1 = nnbc_obj._get_sample(training_inputs, sample_idx, state_idx, subindices)
    np.testing.assert_almost_equal(int_value_0, 2)
    np.testing.assert_almost_equal(int_value_1, 3)


def test_m_re_se_loss_returns_correct_loss_value():
    nnbc_obj = Neural_Network_Base_Class()
    n_qubits = 3
    y_hat = torch.tensor([0, 0.2, 0.9]).reshape(1, n_qubits)
    y_true = torch.tensor([0.1, 0.2, 0.7]).reshape(1, n_qubits)
    loss = nnbc_obj.m_rel_se_loss(y_hat, y_true)
    loss_true = (1 / n_qubits) *  ((0.1 / 0.1) ** 2 + (0.0 / 0.2) ** 2 + (0.2 / 0.7) ** 2)
    np.testing.assert_almost_equal(loss, loss_true)


def test_m_re_se_loss_for_two_samples():
    nnbc_obj = Neural_Network_Base_Class()
    n_qubits = 3
    y_hat = torch.tensor([[0, 0.2, 0.9], [0, 0.3, 1.1]]).reshape(2, n_qubits)
    y_true = torch.tensor([[0.1, 0.2, 0.7], [0.1, 0.5, 1]]).reshape(2, n_qubits)
    loss = nnbc_obj.m_rel_se_loss(y_hat, y_true)
    loss_sample_1_true = (1 / n_qubits) * ((0.1 / 0.1) ** 2 + (0.0 / 0.2) ** 2 + (0.2 / 0.7) ** 2)
    loss_sample_2_true = (1 / n_qubits) * ((0.1 / 0.1) ** 2 + (0.2 / 0.5) ** 2 + (0.1 / 1.0) ** 2)
    loss_true = (loss_sample_1_true + loss_sample_2_true) / 2
    np.testing.assert_almost_equal(loss, loss_true)


def test_convert_khp_features_to_probability_vector_with_two_entries():
    nnbc_obj = Neural_Network_Base_Class()
    nnbc_obj.n_qubits = 3
    nnbc_obj.k = 2
    features = torch.tensor([[0.1, 0.1, -0.1], [0.9, -0.9, -0.9]]).reshape(1, nnbc_obj.n_qubits * nnbc_obj.k)
    prob_vector = nnbc_obj.convert_khp_features_to_probability_vector(features)
    sol_vector = torch.tensor([0, 0.9, 0, 0.1, 0, 0, 0, 0])
    assert np.allclose(prob_vector, sol_vector)


def test_infidelity_loss_returns_correct_value():
    nnbc_obj = Neural_Network_Base_Class()
    n_qubits = 3
    y_hat = torch.tensor([0, 0.2, 0.8]).reshape(1, n_qubits)
    y_true = torch.tensor([0.1, 0.2, 0.7]).reshape(1, n_qubits)
    loss = nnbc_obj.infidelity_loss(y_hat, y_true)
    loss_true = 1 - ((np.sqrt(0*0.1) + np.sqrt(0.2*0.2) + np.sqrt(0.8*0.7)) ** 2)
    np.testing.assert_almost_equal(loss, loss_true)
