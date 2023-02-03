import os
import time
import torch
import pickle
from typing import Any

def save_object(obj: Any, filename: str) -> None:
    """
    Saves an object to file.
    """
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(filename: str) -> Any:
    """
    Loads an object from a file.
    """
    with open(filename, 'rb') as inp:
        return pickle.load(inp)

def save_model(model: Any):
    """
    Saves a torch model to file.
    """
    systime = time.localtime()
    timestr = str(systime[0]) + '_' +  str(systime[1]) + '_' + str(systime[2]) + '_' + str(systime[3]) + '_' + str(systime[4]) + '_' + str(systime[5])
    # to store a model with a timestamp use:
    # torch.save(model.state_dict(), os.path.join('Trained_model', 'model_state_dict'+timestr+'.pth'))
    # else to store it as 'final', use:
    torch.save(model.state_dict(), os.path.join('Trained_model', 'model_state_dict_trained.pth'))
    print("Model saved successfully.")

def load_model(model: Any, name: str='model_state_dict_trained.pth') -> Any:
    """
    Loads the model weights from a file into a torch model.
    """
    model.load_state_dict(torch.load(os.path.join('Trained_model', name)))
    model.to(model.device)
    return model
