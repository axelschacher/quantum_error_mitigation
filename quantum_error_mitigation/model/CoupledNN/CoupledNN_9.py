import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType
from quantum_error_mitigation.model.NN_Base_Class.Neural_Network_Base_Class import Neural_Network_Base_Class


class Coupled_Neural_Network(Neural_Network_Base_Class):
    def __init__(self):
        """
        Constructor of class Coupled_Neural_Network.
        """
        super().__init__()
        self.model_name = "CoupledNN"
        self.inputs_for_NN = "probability_for_each_state"
        # Hyperparameter initialization
        self.epochs = 100000
        self.learning_rate = 5e-3
        self.n_train = 3200
        self.batch_size = int(self.n_train/1)
        self.evaluation_frequency_epochs = self.epochs

        # Layer initialization
        self.num_input_nodes = 48
        self.num_output_nodes = 512
        self.linear1_1 = nn.Linear(32, 32)
        self.linear2_1 = nn.Linear(32, 32)
        self.linear3_1 = nn.Linear(32, 32)

        self.linear1_2 = nn.Linear(16, 16)
        self.linear2_2 = nn.Linear(16, 16)
        self.linear3_2 = nn.Linear(16, 16)

        self.linear1 = nn.Linear(48, 2**9)

        self.num_trainable_parameters = self._count_parameters()


    def forward(self, x: list[TensorType]) -> TensorType:
        """
        Forward pass of the model.
        """
        # we use 2 independent NNs for the forward pass
        x0 = self.linear1_1(x[0])
        x0 = torch.sigmoid(x0)
        x0 = self.linear2_1(x0)
        x0 = torch.sigmoid(x0)
        x0 = self.linear3_1(x0)
        x0 = F.relu(x0)

        x1 = self.linear1_2(x[1])
        x1 = torch.sigmoid(x1)
        x1 = self.linear2_2(x1)
        x1 = torch.sigmoid(x1)
        x1 = self.linear3_2(x1)
        x1 = F.relu(x1)

        x = torch.cat((x0, x1), dim=1)
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = F.normalize(x, p=1.0, dim=1)
        return x

    def generate_mitigation_service_request(self, counts, cm_gen_method):
        """
        Mitigation service request generator for the QuAntiL error mitigation service.
        Update the 'qubits'parameter to the actual used qubits during execution.
        """
        request = {
          "provider": "ibm",
          "qpu": "aer_qasm_simulator", # aer_qasm_simulator or ibmq_quito
          # "noise_model": "ibm_auckland",  # for others: ibmq_quito, ibm_auckland, ibm_nairobi
          # "only_measurement_errors": "True",
          "credentials": {
            "token": "Your Token"
          },
          "cm_gen_method": cm_gen_method,  # standard or tpnm
          "mitigation_method": "inversion",  # inversion or tpnm
          "counts": counts,
          "shots": self.shots,
          "qubits": [0,1,2,3,4,5,6,7,8],  # Array of used qubits, e.g., [0,1,2,3,7,8]
          "max_age": 1400
        }
        return request


if __name__ == '__main__':
    """
    Main method of this model.
    To train the model, run this main method from this files directory.
    If you want to train models for different numbers of qubits, it is easier to store the respective files in seperate directories on your disk to avoid errors caused by loading a not existing model.
    """
    model = Coupled_Neural_Network()
    # model = model.load_model_weights_from_file()
    training_inputs, training_solutions = model.load_training_data(path="4000_samples_rotation_9_qubits")
    datasets_train, datasets_val = model.prepare_training_data(training_inputs, training_solutions, inputs_for_NN = "probability_for_each_state", split_indices=[[0,1,2,3,4], [5,6,7,8]])  # inputs for NN eiter register_probability_repeat or probability_for_each_state
    model.train_model(datasets_train, datasets_val, evaluation_path="Plots/Loss-Training-Plots")
    # model.learning_rate_range_test(datasets_train, datasets_val)
