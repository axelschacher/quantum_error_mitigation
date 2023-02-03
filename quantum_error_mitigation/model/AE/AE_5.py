import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType
from quantum_error_mitigation.model.NN_Base_Class.Neural_Network_Base_Class import Neural_Network_Base_Class


class Autoencoder(Neural_Network_Base_Class):
    def __init__(self):
        """
        Constructor of class Autoencoder.
        """
        super().__init__()
        self.model_name = "AE"
        self.inputs_for_NN = "probability_for_each_state"
        # Hyperparameter initialization
        self.epochs = 75000
        self.learning_rate = 1e-3
        self.n_train = 80
        self.batch_size = int(self.n_train/2)
        self.evaluation_frequency_epochs = self.epochs

        # Layer initialization
        self.num_input_nodes = 32
        self.num_output_nodes = 32
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=40, out_channels=80, kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=80, out_channels=2, kernel_size=7)
        self.linear1 = nn.Linear(40, 15)
        self.linear2 = nn.Linear(15, 40)

        self.deconv1 = nn.ConvTranspose1d(in_channels=2, out_channels=80, kernel_size=7)
        self.deconv2 = nn.ConvTranspose1d(in_channels=80, out_channels=40, kernel_size=5)
        self.deconv3 = nn.ConvTranspose1d(in_channels=40, out_channels=1, kernel_size=3)

    def forward(self, x: list[TensorType]):
        """
        Forward pass of the model.
        """
        x = x[0]
        x = x.unsqueeze(1)  # add a channel dimension
        x = self.conv1(x)
        x = torch.sigmoid(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        x = self.conv3(x)
        x = torch.sigmoid(x)
        x = x.flatten(start_dim=1, end_dim=-1)
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = x.unflatten(dim=1, sizes=(2,20))
        x = self.deconv1(x)
        x = torch.sigmoid(x)
        x = self.deconv2(x)
        x = torch.sigmoid(x)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        x = x.squeeze(1)
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
          "qubits": [0,1,2,3,4],  # Array of used qubits, e.g., [0,1,2,3,7,8]
          "max_age": 1400
        }
        return request


if __name__ == '__main__':
    """
    Main method of this model.
    To train the model, run this main method from this files directory.
    If you want to train models for different numbers of qubits, it is easier to store the respective files in seperate directories on your disk to avoid errors caused by loading a not existing model.
    """
    model = Autoencoder()
    # model = model.load_model_weights_from_file()
    training_inputs, training_solutions = model.load_training_data(path="4000_samples_rotation_5_qubits")
    dataset_train, dataset_val = model.prepare_training_data(training_inputs, training_solutions, inputs_for_NN = "probability_for_each_state")  # inputs for NN eiter register_probability_repeat or probability_for_each_state or k_most_frequent_states
    model.train_model(dataset_train, dataset_val, evaluation_path="Plots/Loss-Training-Plots")
    # model.learning_rate_range_test(dataset_train, dataset_val)
