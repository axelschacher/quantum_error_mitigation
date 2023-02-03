import numpy as np
from quantum_error_mitigation.data.training_data.Data_Generator import Data_Generator

# For all tests:
backend="AerSimulatorQuito"


def test_create_calibation_circuits_returns_same_number_of_circuits_and_bitstrings():
    generator = Data_Generator(n_qubits=2, n_samples=2, n_different_measurements=3, method="rotation_angles", backend=backend)
    returned_circs, returned_bitstrings = generator.create_calibration_circuits()
    assert isinstance(returned_circs, list)
    assert isinstance(returned_circs, list)
    assert len(returned_circs) == len(returned_bitstrings)

def test_create_calibation_circuits_bitstrings_contain_only_0_or_1():
    generator = Data_Generator(n_qubits=2, n_samples=2, n_different_measurements=3, method="rotation_angles", backend=backend)
    _, returned_bitstrings = generator.create_calibration_circuits()
    for bitstring in returned_bitstrings:
        for bit in bitstring:
            assert bit=="0" or bit=="1" or bit==0 or bit==1

def test_generate_random_calibration_bits_length_of_bitstrings_equals_number_of_qubits():
    generator = Data_Generator(n_qubits=2, n_samples=2, n_different_measurements=3, method="rotation_angles", backend=backend)
    returned_bitstrings = generator._generate_random_calibration_bits(n_qubits=10, number_of_samples=4)
    for bitstring in returned_bitstrings:
        assert len(bitstring) == 10

def test_generate_random_calibration_bits_number_of_bitstrings_fits():
    generator = Data_Generator(n_qubits=2, n_samples=2, n_different_measurements=3, method="rotation_angles", backend=backend)
    returned_bitstrings = generator._generate_random_calibration_bits(n_qubits=10, number_of_samples=4)
    assert len(returned_bitstrings) == 4

def test_compute_ry_rotation_theoretical_probability_is_1_for_all_angles_0():
    generator = Data_Generator(n_qubits=2, n_samples=2, n_different_measurements=3, method="rotation_angles", backend=backend)
    state_to_measure = [0,0]
    rotation_angles = np.array([0,0])
    probability = generator._compute_ry_rotation_theoretical_probability(state_to_measure, rotation_angles)
    np.testing.assert_almost_equal(probability, 1.)
