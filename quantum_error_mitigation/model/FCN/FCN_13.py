import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType
from quantum_error_mitigation.model.NN_Base_Class.Neural_Network_Base_Class import Neural_Network_Base_Class


class Fully_connected_Neural_Network(Neural_Network_Base_Class):
    def __init__(self):
        """
        Constructor of class Fully_connected_Neural_Network.
        """
        super().__init__()
        self.model_name = "FCN"
        self.inputs_for_NN = "probability_for_each_state"
        # Hyperparameter initialization
        self.epochs = 100000
        self.learning_rate = 5e-4
        self.n_train = 3200
        self.batch_size = int(self.n_train/1)
        self.evaluation_frequency_epochs = self.epochs

        # Layer initialization
        self.num_input_nodes = 2**13
        self.num_output_nodes = 2**13
        n_hidden_neurons = 32
        self.linear1 = nn.Linear(self.num_input_nodes, n_hidden_neurons)
        self.linear2 = nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.linear3 = nn.Linear(n_hidden_neurons, self.num_output_nodes)
        self.dropout = nn.Dropout(p=0.)

        self.num_trainable_parameters = self._count_parameters()

    def forward(self, x: list[TensorType]):
        """
        Forward pass of the model.
        """
        x = self.linear1(x[0])
        x = self.dropout(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
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
          "qubits": [0,1,2,3,4,5,6,7,8,9,10,11,12],  # Array of used qubits, e.g., [0,1,2,3,7,8]
          "max_age": 1400
        }
        return request

if __name__ == '__main__':
    """
    Main method of this model.
    To train the model, run this main method from this files directory.
    If you want to train models for different numbers of qubits, it is easier to store the respective files in seperate directories on your disk to avoid errors caused by loading a not existing model.
    """
    model = Fully_connected_Neural_Network()
    # model = model.load_model_weights_from_file()
    training_inputs, training_solutions = model.load_training_data(path="4000_samples_rotation_13_qubits")
    datasets_train, datasets_val = model.prepare_training_data(training_inputs, training_solutions, inputs_for_NN = "probability_for_each_state")  # inputs for NN eiter register_probability_repeat or probability_for_each_state or k_most_frequent_states

    model.train_model(datasets_train, datasets_val, evaluation_path="Plots/Loss-Training-Plots")
    # model.learning_rate_range_test(datasets_train, datasets_val)
