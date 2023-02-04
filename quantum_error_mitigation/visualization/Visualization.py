import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Tuple, Optional
from quantum_error_mitigation.data.information_handler import Information_Handler


def plot_performance_measures(visualization_path: Optional[str] = None) -> None:
    """
    Use the knowledge_database to plot different performance measures of the model(s).
    Commonly used to plot the loss decrease during training.
    """
    knowledge_database = Information_Handler.get_knowledge_database(visualization_path)
    # print(knowledge_database.tail(15))
    losses, models, metrics, n_qubits_in_kndb, n_epochs_in_kndb, max_loss_value = _compute_loss_from_knowledge_database(knowledge_database)
    max_epochs = n_epochs_in_kndb[-1]

    for model in models:
        # epochs-loss-plots
        for n_qubits in n_qubits_in_kndb:
            fig, ax = plt.subplots()
            epochs_mse_unmitigated, loss_per_epoch_mse_unmitigated = _get_epochs_and_loss(losses, model, metric='mse_unmitigated', n_qubits=n_qubits)
            epochs_kl_div_unmitigated, loss_per_epoch_kl_div_unmitigated = _get_epochs_and_loss(losses, model, metric='kl_div_unmitigated', n_qubits=n_qubits)
            epochs_if_unmitigated, loss_per_epoch_if_unmitigated = _get_epochs_and_loss(losses, model, metric='if_unmitigated', n_qubits=n_qubits)
            epochs_mse_train, loss_per_epoch_mse_train = _get_epochs_and_loss(losses, model, metric='mse_train', n_qubits=n_qubits)
            epochs_kl_div_train, loss_per_epoch_kl_div_train = _get_epochs_and_loss(losses, model, metric='kl_div_train', n_qubits=n_qubits)
            epochs_if_train, loss_per_epoch_if_train = _get_epochs_and_loss(losses, model, metric='if_train', n_qubits=n_qubits)
            epochs_mse_val, loss_per_epoch_mse_val = _get_epochs_and_loss(losses, model, metric='mse_val', n_qubits=n_qubits)
            epochs_kl_div_val, loss_per_epoch_kl_div_val = _get_epochs_and_loss(losses, model, metric='kl_div_val', n_qubits=n_qubits)
            epochs_if_val, loss_per_epoch_if_val = _get_epochs_and_loss(losses, model, metric='if_val', n_qubits=n_qubits)
            epochs_mse_li, loss_per_epoch_mse_li = _get_epochs_and_loss(losses, model, metric='mse_li', n_qubits=n_qubits)
            epochs_kl_div_li, loss_per_epoch_kl_div_li = _get_epochs_and_loss(losses, model, metric='kl_div_li', n_qubits=n_qubits)
            epochs_if_li, loss_per_epoch_if_li = _get_epochs_and_loss(losses, model, metric='if_li', n_qubits=n_qubits)
            epochs_mse_tpnm, loss_per_epoch_mse_tpnm = _get_epochs_and_loss(losses, model, metric='mse_tpnm', n_qubits=n_qubits)
            epochs_kl_div_tpnm, loss_per_epoch_kl_div_tpnm = _get_epochs_and_loss(losses, model, metric='kl_div_tpnm', n_qubits=n_qubits)
            epochs_if_tpnm, loss_per_epoch_if_tpnm = _get_epochs_and_loss(losses, model, metric='if_tpnm', n_qubits=n_qubits)

            ax.plot(epochs_mse_unmitigated.iloc[0], loss_per_epoch_mse_unmitigated.iloc[0],
                color='red', marker='x', linestyle='-', label='MSE, unmitigated')
            ax.plot(epochs_mse_unmitigated, loss_per_epoch_mse_unmitigated,
                color='red', marker=None, linestyle='-', label='MSE, unmitigated')

            ax.plot(epochs_kl_div_unmitigated.iloc[0], loss_per_epoch_kl_div_unmitigated.iloc[0],
                color='red', marker='1', linestyle='--', label='KL div, unmitigated')
            ax.plot(epochs_kl_div_unmitigated, loss_per_epoch_kl_div_unmitigated,
                color='red', marker=None, linestyle='--', label='KL div, unmitigated')

            ax.plot(epochs_if_unmitigated.iloc[0], loss_per_epoch_if_unmitigated.iloc[0],
                color='red', marker='o', fillstyle='none', linestyle='-.', label='IF, unmitigated')
            ax.plot(epochs_if_unmitigated, loss_per_epoch_if_unmitigated,
                color='red', marker=None, fillstyle='none', linestyle='-.', label='IF, unmitigated')

            stepsize=1
            ax.plot(epochs_mse_train.iloc[::stepsize], loss_per_epoch_mse_train.iloc[::stepsize],
                color='green', marker='x', linestyle='-', label='MSE, train set')
            ax.plot(epochs_kl_div_train.iloc[::stepsize], loss_per_epoch_kl_div_train.iloc[::stepsize],
                color='green', marker='1', linestyle='--', label='Kl div, train set')
            ax.plot(epochs_if_train.iloc[::stepsize], loss_per_epoch_if_train.iloc[::stepsize],
                color='green', marker='o', fillstyle='none', linestyle='-.', label='IF, train set')
            ax.plot(epochs_mse_val.iloc[::stepsize], loss_per_epoch_mse_val.iloc[::stepsize],
                color='orange', marker='x', linestyle='-', label='MSE, validation set')
            ax.plot(epochs_kl_div_val.iloc[::stepsize], loss_per_epoch_kl_div_val.iloc[::stepsize],
                color='orange', marker='1', linestyle='--', label='KL div, validation set')
            ax.plot(epochs_if_val.iloc[::stepsize], loss_per_epoch_if_val.iloc[::stepsize],
                color='orange', marker='o', fillstyle='none', linestyle='-.', label='IF, validation set')

            ax.plot(epochs_mse_li.iloc[0], loss_per_epoch_mse_li.iloc[0],
                color='palevioletred', marker='x', linestyle='-', linewidth=1.5, label='MSE, linear inversion')
            ax.plot(epochs_mse_li, loss_per_epoch_mse_li,
                color='palevioletred', marker=None, linestyle='-', linewidth=1.5, label='MSE, linear inversion')

            ax.plot(epochs_kl_div_li.iloc[0], loss_per_epoch_kl_div_li.iloc[0],
                color='palevioletred', marker='1', linestyle='--', linewidth=1.5, label='KL div, linear inversion')
            ax.plot(epochs_kl_div_li, loss_per_epoch_kl_div_li,
                color='palevioletred', marker=None, linestyle='--', linewidth=1.5, label='KL div, linear inversion')

            ax.plot(epochs_if_li.iloc[0], loss_per_epoch_if_li.iloc[0],
                color='palevioletred', marker='o', fillstyle='none', linestyle='-.', linewidth=1.5, label='IF, linear inversion')
            ax.plot(epochs_if_li, loss_per_epoch_if_li,
                color='palevioletred', marker=None, fillstyle='none', linestyle='-.', linewidth=1.5, label='IF, linear inversion')

            ax.plot(epochs_mse_tpnm.iloc[0], loss_per_epoch_mse_tpnm.iloc[0],
                color='mediumturquoise', marker='x', linestyle='-', linewidth=1.2, alpha=0.5, label='MSE, TPNM')
            ax.plot(epochs_mse_tpnm, loss_per_epoch_mse_tpnm,
                color='mediumturquoise', marker=None, linestyle='-', linewidth=1.2, alpha=0.5, label='MSE, TPNM')

            ax.plot(epochs_kl_div_tpnm.iloc[0], loss_per_epoch_kl_div_tpnm.iloc[0],
                color='mediumturquoise', marker='1', linestyle='--', linewidth=1.2, alpha=0.5, label='KL div, TPNM')
            ax.plot(epochs_kl_div_tpnm, loss_per_epoch_kl_div_tpnm,
                color='mediumturquoise', marker=None, linestyle='--', linewidth=1.2, alpha=0.5, label='KL div, TPNM')

            ax.plot(epochs_if_tpnm.iloc[0], loss_per_epoch_if_tpnm.iloc[0],
                color='mediumturquoise', marker='o', fillstyle='none', linestyle='-.', linewidth=1.2, alpha=0.5, label='IF, TPNM')
            ax.plot(epochs_if_tpnm, loss_per_epoch_if_tpnm,
                color='mediumturquoise', marker=None, fillstyle='none', linestyle='-.', linewidth=1.2, alpha=0.5, label='IF, TPNM')

            grey_patch, = [Line2D([0], [0], color='grey', linewidth=1.2, linestyle='-', marker='x')]
            grey_dashed_patch, = [Line2D([0], [0], color='grey', linewidth=1.2, linestyle='--', marker='1')]
            grey_dash_dotted_patch, = [Line2D([0], [0], color='grey', linewidth=1.2, linestyle='-.', marker='o', fillstyle='none')]
            red_patch, = [Line2D([0], [0], color='red', linewidth=1.2, linestyle='-')]
            green_patch, = [Line2D([0], [0], color='green', linewidth=1.2, linestyle='-')]
            orange_patch, = [Line2D([0], [0], color='orange', linewidth=1.2, linestyle='-')]
            purple_patch, = [Line2D([0], [0], color='palevioletred', linewidth=1.2, linestyle='-')]
            turquoise_patch, = [Line2D([0], [0], color='mediumturquoise', linewidth=1.2, linestyle='-')]

            ax.legend([red_patch, green_patch, orange_patch, purple_patch, turquoise_patch, grey_patch, grey_dashed_patch, grey_dash_dotted_patch],
                  ['unmitigated', 'train', 'validation', 'li. inversion', 'tpnm', 'MSE', 'KL div', 'IF'], loc='upper right', ncol=2, fancybox=True, shadow=False)

            plt.xlabel('Number of training epochs [-]')
            plt.ylabel('Loss [-]')
            x_locs, x_labels = plt.xticks()
            plt.xticks(np.linspace(0, max_epochs, num=5))
            plt.title(f'Qualitative performance measures for model {model} with {n_qubits} qubits')
            if visualization_path is None:
                figure_path = os.path.join("Plots/Loss-Training-Plots")
            else:
                figure_path = os.path.join(visualization_path)
            outpath = os.path.relpath(os.path.join(figure_path, f'Performance_measures_{model}_{n_qubits}_qubits_variable_n_epochs.pdf'))
            plt.savefig(outpath)

            # cropped and logscale plot for better visibility
            plt.ylim(bottom=1e-4, top=2.0)
            plt.yscale("log")
            outpath = os.path.relpath(os.path.join(figure_path, f'Performance_measures_{model}_{n_qubits}_qubits_variable_n_epochs_cropped_logscale.pdf'))
            plt.savefig(outpath)
            plt.close()


def plot_double_dataset_evaluation_data_generation_method(model_name_where_trained: str, model_name_to_validate: str, visualization_path: Optional[str] = None) -> None:
        """
        Visualizes the losses on two different datasets.
        Only needed for evaluation of a model on a dataset, where it was not trained.
        """
        knowledge_database = Information_Handler.get_knowledge_database(visualization_path)
        # print(knowledge_database.tail(15))
        losses, models, metrics, n_qubits_in_kndb, n_epochs_in_kndb, max_loss_value = _compute_loss_from_knowledge_database(knowledge_database)
        model_name_1 = model_name_where_trained
        model_name_2 = model_name_to_validate

        # epochs-loss-plots
        for n_qubits in n_qubits_in_kndb:
            fig, ax = plt.subplots()
            epochs_mse_train_1, loss_per_epoch_mse_train_1 = _get_epochs_and_loss(losses, model_name_1, metric='mse_train', n_qubits=n_qubits)
            epochs_kl_div_train_1, loss_per_epoch_kl_div_train_1 = _get_epochs_and_loss(losses, model_name_1, metric='kl_div_train', n_qubits=n_qubits)
            epochs_if_train_1, loss_per_epoch_if_train_1 = _get_epochs_and_loss(losses, model_name_1, metric='if_train', n_qubits=n_qubits)
            epochs_mse_val_1, loss_per_epoch_mse_val_1 = _get_epochs_and_loss(losses, model_name_1, metric='mse_val', n_qubits=n_qubits)
            epochs_kl_div_val_1, loss_per_epoch_kl_div_val_1 = _get_epochs_and_loss(losses, model_name_1, metric='kl_div_val', n_qubits=n_qubits)
            epochs_if_val_1, loss_per_epoch_if_val_1 = _get_epochs_and_loss(losses, model_name_1, metric='if_val', n_qubits=n_qubits)

            epochs_mse_train_2, loss_per_epoch_mse_train_2 = _get_epochs_and_loss(losses, model_name_2, metric='mse_train', n_qubits=n_qubits)
            epochs_kl_div_train_2, loss_per_epoch_kl_div_train_2 = _get_epochs_and_loss(losses, model_name_2, metric='kl_div_train', n_qubits=n_qubits)
            epochs_if_train_2, loss_per_epoch_if_train_2 = _get_epochs_and_loss(losses, model_name_2, metric='if_train', n_qubits=n_qubits)
            epochs_mse_val_2, loss_per_epoch_mse_val_2 = _get_epochs_and_loss(losses, model_name_2, metric='mse_val', n_qubits=n_qubits)
            epochs_kl_div_val_2, loss_per_epoch_kl_div_val_2 = _get_epochs_and_loss(losses, model_name_2, metric='kl_div_val', n_qubits=n_qubits)
            epochs_if_val_2, loss_per_epoch_if_val_2 = _get_epochs_and_loss(losses, model_name_2, metric='if_val', n_qubits=n_qubits)

            ax.plot(epochs_mse_train_1, loss_per_epoch_mse_train_1,
                color='green', marker='x', linestyle='-', label='MSE, train set')
            ax.plot(epochs_mse_val_1, loss_per_epoch_mse_val_1,
                color='orange', marker='x', linestyle='-', label='MSE, validation set')

            ax.plot(epochs_mse_train_2, loss_per_epoch_mse_train_2,
                color='springgreen', marker='x', linestyle='-', label='MSE, other train set')
            ax.plot(epochs_mse_val_2, loss_per_epoch_mse_val_2,
                color='lightcoral', marker='x', linestyle='-', label='MSE, other validation set')

            grey_patch, = [Line2D([0], [0], color='grey', linewidth=1.2, linestyle='-', marker='x')]
            light_green_patch, = [Line2D([0], [0], color='springgreen', linewidth=1.2, linestyle='-')]
            green_patch, = [Line2D([0], [0], color='green', linewidth=1.2, linestyle='-')]
            orange_patch, = [Line2D([0], [0], color='orange', linewidth=1.2, linestyle='-')]
            light_orange_patch, = [Line2D([0], [0], color='lightcoral', linewidth=1.2, linestyle='-')]

            ax.legend([green_patch, orange_patch, light_green_patch, light_orange_patch, grey_patch],
                  ['train', 'validation', 'other train set', 'other val set', 'MSE'], loc='upper right', ncol=1, bbox_to_anchor=(0.97, 0.9), fancybox=True, shadow=True)
            plt.xlabel('Number of training epochs [-]')
            plt.ylabel('Loss [-]')
            x_locs, x_labels = plt.xticks()
            plt.xticks(np.linspace(0, 100000, num=5))
            plt.title('Comparison of the data generation method')
            if visualization_path is None:
                figure_path = os.path.join("Plots/Loss-Training-Plots")
            else:
                figure_path = os.path.join(visualization_path)
            plt.ylim(bottom=5e-4, top=2.0)
            plt.yscale("log")
            outpath = os.path.relpath(os.path.join(figure_path, f'Double_dataset_evaluation_{model_name_1}_{n_qubits}_qubits_variable_n_epochs_cropped_logscale.pdf'))
            plt.savefig(outpath)
            plt.close()


def _get_entity_and_loss(knowledge_database: pd.DataFrame, entity: str, idx: pd.Index) -> Tuple[list, list]:
    """
    Returns the entity (e.g. n_epochs or n_qubits) and the losses for a given index.
    """
    entity_values = knowledge_database[entity][idx]
    loss_values = knowledge_database['loss'][idx]
    return entity_values, loss_values


def _compute_loss_from_knowledge_database(knowledge_database: pd.DataFrame) -> dict:
    """
    Returns all losses from the knowledge_database such that they can be used for visualization purposes.
    """
    # find unique features
    models = pd.unique(knowledge_database['model_name'])
    metrics = pd.unique(knowledge_database['metric'])
    n_qubits_in_kndb = pd.unique(knowledge_database['n_qubits'])
    n_epochs_in_kndb = pd.unique(knowledge_database['n_epochs'])
    max_loss_value = 0.
    losses = {}  # for all models
    for model in models:
        model_idx = Information_Handler.get_idx(knowledge_database, column_name='model_name', value=model)
        model_losses = {}  # for a specific model and all metrics
        for metric in metrics:
            metric_idx = Information_Handler.get_idx(knowledge_database, column_name='metric', value=metric)
            intersection_idx = model_idx.intersection(metric_idx)
            metric_losses = {} # for a specific metric and all n_qubits/n_epochs
            losses_variable_n_qubits = {}  # for a specific metric and all n_qubits
            losses_variable_n_epochs = {}  # for a specific metric and all n_epochs
            for n_qubits in n_qubits_in_kndb:
                n_qubits_idx = Information_Handler.get_idx(knowledge_database, column_name='n_qubits', value=n_qubits)
                intersection_n_qubits_idx = intersection_idx.intersection(n_qubits_idx)  # model, metric and n_qubits are fixed, epochs variable
                epochs, loss_per_epoch = _get_entity_and_loss(knowledge_database, entity='n_epochs', idx=intersection_n_qubits_idx)
                if loss_per_epoch.max() > max_loss_value:
                    max_loss_value = loss_per_epoch.max()
                losses_variable_n_epochs[n_qubits] = {'epochs': epochs, 'loss_per_epoch': loss_per_epoch}
            for n_epochs in n_epochs_in_kndb:
                n_epochs_idx = Information_Handler.get_idx(knowledge_database, column_name='n_epochs', value=n_epochs)
                intersection_n_epochs_idx = intersection_idx.intersection(n_epochs_idx)  # model, metric and n_epochs are fixed, n_qubits variable
                qubits, loss_per_qubit = _get_entity_and_loss(knowledge_database, entity='n_qubits', idx=intersection_n_epochs_idx)
                if loss_per_qubit.max() > max_loss_value:
                    max_loss_value = loss_per_qubit.max()
                losses_variable_n_qubits[n_epochs] = {'qubits': qubits, 'loss_per_qubit': loss_per_qubit}
            metric_losses['variable_n_qubits'] = losses_variable_n_qubits
            metric_losses['variable_n_epochs'] = losses_variable_n_epochs
            model_losses[metric] = metric_losses
        losses[model] = model_losses
    return losses, models, metrics, n_qubits_in_kndb, n_epochs_in_kndb, max_loss_value


def _get_epochs_and_loss(losses: dict, model:str, metric: str, n_qubits: int) -> Tuple[pd.Series, pd.Series]:
    """
    Getter for the epoch and corresponding loss for given model, metric and n_qubits.
    """
    epochs_metric = losses[model][metric]['variable_n_epochs'][n_qubits]['epochs']
    loss_per_epoch_metric = losses[model][metric]['variable_n_epochs'][n_qubits]['loss_per_epoch']
    return epochs_metric, loss_per_epoch_metric


if __name__ == "__main__":
    """
    Call the main function, if desired, from the model's base directory using ```python ../../visualization/Visualization.py```.
    """
    plot_performance_measures()
