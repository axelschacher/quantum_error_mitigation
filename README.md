# Deep Learning-Based Quantum Readout Error Mitigation Framework

This Framework is intended to mitigate quantum readout errors.
Different deep learning-based models are implemented and new models can easily be integrated in our framework.
Our implementations can solve this task with high quality and low resource requirements in the online phase.

## Setup Guide

In order to take advantage of the models, you have to integrate this Framework into your existing project.
We provide a setup file to install the whole project via pip using

```bash
git clone https://github.com/axelschacher/quantum_error_mitigation.git
cd quantum_error_mitigation
pip install quantum_error_mitigation
```

Then, all modules of the project can be imported to other python projects.
Exemplary, if you have a Qiskit counts object you want to mitigate on the QPU ibmq quito, just add the following lines to your code:

```python
from quantum_error_mitigation.model.FCN.FCN_5 import Fully_connected_Neural_Network  # choose the directory and file where the desired model is stored

model = Fully_connected_Neural_Network()
model = model.load_model_weights_from_file()
mitigated_counts = model.mitigate_counts(counts)
```

If no pretrained model is available or the model is outdated, a dataset to train a new model can be generated using

```python
from quantum_error_mitigation.data.training_data.Data_Generator import Data_Generator

n_qubits = 5  # number of qubits
n_samples = 4000  # number of samples in the dataset
n_different_measurements = 32 # crops the probability distribution to store only the most frequent counts. Choose min(2**n_qubits, n_shots) as a maximum value
method = "rotation_angles"  # choose between 'rotation_angles' and 'calibration_bits'
backend = "ibmq_quito"  # choose between 'ibmq_quito', 'aer_quito' or 'noise_model_auckland'
path = "./4000_samples_rotation_5_qubits/"  # path where the dataset is stored

generator = Data_Generator(n_qubits, n_samples, n_different_measurements, method, backend, path)
generator.generate_training_data()
```

and the training is initiated by the following code:

```python
model = Fully_connected_Neural_Network()
training_inputs, training_solutions = model.load_training_data(path)
inputs_for_NN = "probability_for_each_state"  # choose between 'probability_for_each_state' or 'k_most_frequent_states'
datasets_train, datasets_val = model.prepare_training_data(training_inputs, training_solutions, inputs_for_NN)
evaluation_path = "evaluation"
model.train_model(datasets_train, datasets_val, evaluation_path)
```

In principle, that is everything needed for a quick start.
If the model parameters shall be changed, or more advanced adjustments are to be made, search the respective file:
All models can be found in `quantum_error_mitigation.model`
All methods needed to generate training and validation data can be found in `quantum_error_mitigation.data`
All methods in the framework can be adjusted, if desired to build custom input or output features.
We support a comparison of results with the [error mitigation service from the QuAntiL project](https://github.com/UST-QuAntiL/error-mitigation-service), which is explained in the paper [Configurable Readout Error Mitigation in Quantum Workflows](https://www.mdpi.com/2079-9292/11/19/2983).

## License
Copyright 2023 Axel Schumacher

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
