import os
import sys

sys.path.append(os.getcwd())
import argparse

import optuna
import torch
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.trial import TrialState
from tqdm import tqdm

import src.models.config as cfg
from src.models.config import *
from src.models.lstm_attention import LSTMWithAttention


def define_model(trial):
    args = read_arguments_train()

    args.dataset_name = "wdbench_nl_sparql"
    args.experiment_name = "lstm"
    args.num_epochs = 50
    args.wandb = False
    args.model_name = "lstm"

    input_size, output_size = 4096, 1
    hidden_size = trial.suggest_int("hidden_size", 16, 128)
    args.num_layers = trial.suggest_int("num_layers", 2, 6)
    args.learning_rate = trial.suggest_float('lr', 1e-5, 1e-3)
    args.weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2)
    args.use_validation_set = True

    args.bidirectional = False
    lstm = LSTMWithAttention(input_size, hidden_size, output_size, args)

    return lstm

def objective(trial):
    lstm = define_model(trial)

    lstm.train()
    total_loss, prev_epoch_loss = 0, 0
    accs = []
    early_stopping_patience = 5
    patience_counter = 0
    best_val_loss = float('inf')
    for epoch in tqdm(range(lstm.num_epochs)):
        sample_correct, sample_count = 0, 0
        for i, data in enumerate(lstm.train_data_loader):

            # todo
            ids = data['encoding'].to(lstm.device)
            labels = data['labels'].to(lstm.device)
            queries = data['query']
            query_id = data['query_id']
        
            lstm.optimizer.zero_grad()
            h_0, c_0 = lstm.init_hidden(batch_size=ids.size(0))


            output, _ = lstm(ids, h_0, c_0)
            # import code; code.interact(local=dict(globals(), **locals()))
            loss = lstm.loss_fn(output.squeeze(1), labels)
            loss.backward()

            total_loss += loss.item()

            predicted = torch.sigmoid(output.squeeze(1)) >= 0.5
            sample_correct += (predicted == labels).sum().item()
            sample_count += labels.size(0)


            lstm.optimizer.step()
            if (lstm.checkpoint_after_n is not None) and (i % lstm.checkpoint_after_n == 0):
                print(f'Epoch: {epoch}, Loss: {loss.item()}')
                lstm.save_checkpoint(epoch)
        
        train_loss = round((total_loss - prev_epoch_loss) / len(lstm.train_data_loader), 4)
        train_acc = round(sample_correct / sample_count, 4)

        accs.append(train_acc)

        prev_epoch_loss = total_loss
        _, val_loss = lstm.validate_model(epoch=epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # # Save the best model
            # torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping, epoch: {epoch}, val_loss: {val_loss}, best_val_loss: {best_val_loss}")
            break

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return val_loss

    # torch.manau



def main():
        # create_training_data()
    study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=100, timeout=600)
    wandb_kwargs = {
        "project": "mtrdf",
        "entity": "gehd"
    }
    wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)
    study.optimize(objective, n_trials=1000, callbacks=[wandbc])

    pruned_trials = study.get_trials(
        deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    main()