import os
import torch
import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import Union, Optional
from torchtyping import TensorType


def quantum_register_to_integer(register: Union[str, list, npt.NDArray, TensorType], big_endian: bool = True) -> int:
    """
    Computes the integer value of a register.
    """
    result = 0
    length = len(register)
    if isinstance(register, str):
        for i, bit in enumerate(register):
            bit_int = int(bit)
            result += get_value_of_qubit(bit_int, i, big_endian=big_endian, length=length)
    elif isinstance(register, list):
        for i in range(len(register)):
            bit = register[i]
            bit_int = int(np.rint(bit))
            result += get_value_of_qubit(bit_int, i, big_endian=big_endian, length=length)
    elif isinstance(register, np.ndarray) or isinstance(register, torch.Tensor):
        for i in range(register.shape[0]):
            bit = register[i]
            bit_int = int(np.rint(bit))
            result += get_value_of_qubit(bit_int, i, big_endian=big_endian, length=length)
    else:
        raise TypeError(f"Type {type(register)} not supported. Ensure the register is of type 'str', 'list', 'np.ndarray' or 'torch.Tensor'.")
    return result


def integer_value_to_classical_register(value: int, n_bits: int, big_endian: bool = True) -> list[int]:
    """
    Computes the (classical) register corresponding to a given value.
    Returned register is either big-endian (we used this in our code) or little-endian (qiskit results are little-endian).

    Args:
    value: the value to represent in the register
    n_qubits: the number of bits in the register
    """
    binary_code = format(value, "b")  # little-endian
    big_endian_binary_code = binary_code[::-1]
    length = len(binary_code)
    if length < n_bits:
        zeros_to_add = n_bits-len(binary_code)
        big_endian_binary_code = big_endian_binary_code + "0"*zeros_to_add
    elif length > n_bits:
        raise ValueError(f"integer value {value} is larger than maximum value for a {n_bits}-bit register.")
    big_endian_binary_code = [int(bit) for bit in big_endian_binary_code]
    if not big_endian:
        return big_endian_binary_code[::-1]
    return big_endian_binary_code


def get_value_of_qubit(bit: int, index: int, big_endian: bool, length: Optional[int] = 0) -> int:
    """
    Computes the value of a qubit at a given position (index) in a register.
    """
    if big_endian:
        value = bit * (2 ** index)
    else:
        assert length > 0
        value = bit * (2 ** (length-index-1))  # the last bit (index=length-1) has to be 2**0.
    return value


def split_register_by_indices(register: npt.NDArray, indices: list) -> list:
    """
    Splits a register into multiple sub-registers. Indices not in indices are omitted, all others are returned as groups as specified by indices.

    Args:
    register: the register to split.
    indices: List of sublists where each sublist contains the indices of the bits in a group.
    """
    subregisters = []
    for subindices in indices:
        subregister = register[subindices]
        subregisters.append(subregister)
    return subregisters


def initialize_knowledge_database() -> pd.DataFrame:
    """
    Setup empty knowledge base containing the losses for different hyperparameters (e.g. model types, metrics, n_qubits, n_trained_epochs)

    Returns:
    knowledge_database: pd.DataFrame that contains columns for each feature and where samples can be added.
    """
    knowledge_database = pd.DataFrame(columns=['model_name', 'metric', 'n_qubits', 'n_epochs', 'loss'])
    knowledge_database = knowledge_database.astype({'model_name': 'str', 'metric': 'str', 'n_qubits': 'int', 'n_epochs': 'int', 'loss': 'float'})
    return knowledge_database


def get_knowledge_database(path: Optional[str] = None) -> pd.DataFrame:
    """
    Either initializes a new DataFrame or loads the exising knowledge database.
    """
    store = pd.HDFStore(get_knowledge_database_path(path))
    try:
        knowledge_database = store["knowledge_database"]
    except KeyError:
        knowledge_database = initialize_knowledge_database()
    store.close()
    return knowledge_database


def get_knowledge_database_path(path: Optional[str] = None) -> os.path:
    """
    Returns the path where the knowledge_database is stored.
    """
    if path is None:
        return os.path.relpath("Plots/Loss-Training-Plots/knowledge_database.hdf5")  # relatively from call of the model.
    else:
        return os.path.relpath(os.path.join(path, "knowledge_database.hdf5"))


def save_knowledge_database(knowledge_database: pd.DataFrame, path: Optional[str] = None) -> None:
    """
    Saves the knowledge database to a HDF5 file on disk.
    """
    store = pd.HDFStore(get_knowledge_database_path(path))
    store["knowledge_database"] = knowledge_database
    store.close()


def add_sample_to_knowledge_database(sample: dict, path: Optional[str] = None) -> None:
    """
    Extends the knowledge_database with the given sample.
    """
    # since we don't open the database often, to add a sample, we open it, write the sample and close it again.
    knowledge_database = get_knowledge_database(path)
    sample = pd.DataFrame([sample])
    knowledge_database = pd.concat([knowledge_database, sample], ignore_index=True)
    knowledge_database = clean_up_knowledge_database(knowledge_database)
    save_knowledge_database(knowledge_database, path)


def clean_up_knowledge_database(knowledge_database: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates (losses of older runs) from the knowledge database.
    """
    dups_boolean = knowledge_database.duplicated(subset=['model_name', 'metric', 'n_qubits', 'n_epochs'], keep='last')
    remove_idx = knowledge_database.index[dups_boolean]
    return knowledge_database.drop(index=remove_idx)


def get_idx(knowledge_database: pd.DataFrame, column_name: str, value: any) -> pd.Index:
    """
    Computes the index where a sample in column_name matches the given value.
    """
    rows_boolean = knowledge_database[column_name].apply(lambda criterion: criterion==value)
    return knowledge_database.index[rows_boolean]
