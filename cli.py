from mortality_part_preprocessing import load_pad_separate
from mortality_classification import train_test
import os
import click
import torch
import random
import numpy as np
import json


@click.command()
@click.option(
    "--output_path",
    default="./ehr_classification_results/",
    help="Path to output folder",
)
@click.option("--pooling", default="max", help="pooling function")
@click.option("--epochs", default=300, help="model dropout rate")
@click.option("--dropout", default=0.4, help="model dropout rate")
@click.option("--attn_dropout", default=0.4, help="model attention dropout rate")
@click.option(
    "--model_type", default="transformer", help="model_type"
)
@click.option("--heads", default=1, help="number of attention heads")
@click.option("--batch_size", default=64, help="batch size")
@click.option("--layers", default=1, help="number of attention layers")
@click.option("--dataset_id", default="physionet2012", help="filename id of dataset")
@click.option("--base_path", default="/dtu/blackhole/1b/203515/P12data", help="Path to data folder")
@click.option("--lr", default=0.001, help="learning rate")
@click.option("--patience", default=10, help="patience for early stopping")
@click.option(
    "--loss_function",
    default="cross_entropy",
    help="Loss function to use. Options are: cross_entropy, bce_with_logits, mse",
)
@click.option(
    "--use_mask",
    default=False,
    help="boolean, use mask for timepoints with no measurements",
)
@click.option(
    "--early_stop_criteria",
    default="auroc",
    help="what to early stop on. Options are: auroc, auprc, auprc+auroc, f1-score or loss",
)
# Existing model parameters...
@click.option("--seft_n_phi_layers", default=3)
@click.option("--seft_phi_width", default=32)
@click.option("--seft_phi_dropout", default=0.)
@click.option("--seft_n_psi_layers", default=2)
@click.option("--seft_psi_width", default=64)
@click.option("--seft_psi_latent_width", default=128)
@click.option("--seft_dot_prod_dim", default=128)
@click.option("--seft_latent_width", default=128)
@click.option("--seft_n_rho_layers", default=3)
@click.option("--seft_rho_width", default=32)
@click.option("--seft_rho_dropout", default=0.)
@click.option("--seft_max_timescales", default=100)
@click.option("--seft_n_positional_dims", default=4)
@click.option("--ipnets_imputation_stepsize", default=0.25)
@click.option("--ipnets_reconst_fraction", default=0.25)
@click.option("--recurrent_dropout", default=0.3)
@click.option("--recurrent_n_units", default=100)
# New Mamba-specific parameters
@click.option("--mamba_d_model", default=256, help="Mamba model dimension")
@click.option("--mamba_state_size", default=16, help="Mamba state space size")
@click.option("--mamba_expand_factor", default=2, help="Mamba expansion factor")
@click.option("--mamba_conv_size", default=4, help="Mamba convolution size")
def core_function(
    output_path,
    base_path,
    model_type,
    epochs,
    dataset_id,
    batch_size,
    lr,
    patience,
    early_stop_criteria,
    loss_function,
    **kwargs
):

    model_args = kwargs

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    accum_loss = []
    accum_accuracy = []
    accum_auprc = []
    accum_auroc = []
    accum_f1score = []

    for split_index in range(1, 6):

        base_path_new = f"{base_path}/split_{split_index}"
        train_pair, val_data, test_data = load_pad_separate(
            dataset_id, base_path_new, split_index
        )

        # make necessary folders
        if not os.path.exists(output_path):
            try:
                os.mkdir(output_path)
            except OSError as err:
                print("OS error:", err)
                
        # make run folder
        base_run_path = os.path.join(output_path, f"split_{split_index}")
        run_path = base_run_path
        os.makedirs(run_path, exist_ok=True)

        # save model settings
        model_settings = {
            "model_type": model_type,
            "batch_size": batch_size,
            "epochs": epochs,
            "dataset": dataset_id,
            "learning_rate": lr,
            "patience": patience,
            "early_stop_criteria": early_stop_criteria,
            "base_path": base_path,
            "pooling_fxn": model_args["pooling"],
            "loss_function": loss_function,
        }
        # Add model-specific settings
        if model_type == "transformer":
            model_settings["layers"] = model_args["layers"]
            model_settings["dropout"] = model_args["dropout"]
            model_settings["attn_dropout"] = model_args["attn_dropout"]
            model_settings["use_timepoint_mask"] = model_args["use_mask"]
            model_settings["heads"] = model_args["heads"]
        elif model_type == "mamba":  # New Mamba settings
            model_settings.update({
                "layers": model_args["layers"],
                "dropout": model_args["dropout"],
                "d_model": model_args["mamba_d_model"],
                "state_size": model_args["mamba_state_size"],
                "expand_factor": model_args["mamba_expand_factor"],
                "conv_size": model_args["mamba_conv_size"],
                "use_timepoint_mask": model_args["use_mask"]
            })
        elif model_type == "seft":
            model_settings.update({
                "dropout": model_args["dropout"],
                "attn_dropout": model_args["attn_dropout"],
                "use_timepoint_mask": model_args["use_mask"],
                "heads": model_args["heads"],
                "seft_n_phi_layers": model_args["seft_n_phi_layers"],
                "seft_phi_width": model_args["seft_phi_width"],
                "seft_phi_dropout": model_args["seft_phi_dropout"],
                "seft_n_psi_layers": model_args["seft_n_psi_layers"],
                "seft_psi_width": model_args["seft_psi_width"],
                "seft_psi_latent_width": model_args["seft_psi_latent_width"],
                "seft_dot_prod_dim": model_args["seft_dot_prod_dim"],
                "seft_latent_width": model_args["seft_latent_width"],
                "seft_n_rho_layers": model_args["seft_n_rho_layers"],
                "seft_rho_width": model_args["seft_rho_width"],
                "seft_rho_dropout": model_args["seft_rho_dropout"]
            })
        elif model_type in ("grud", "ipnets"):
            model_settings.update({
                "recurrent_dropout": model_args["recurrent_dropout"],
                "recurrent_n_units": model_args["recurrent_n_units"]
            })
            if model_type == "ipnets":
                model_settings.update({
                    "ipnets_imputation_stepsize": model_args["ipnets_imputation_stepsize"],
                    "ipnets_reconst_fraction": model_args["ipnets_reconst_fraction"]
                })

        with open(f"{run_path}/model_settings.json", "w") as fp:
            json.dump(model_settings, fp)

        # run training
        loss, accuracy_score, auprc_score, auc_score, f1_score = train_test(
            train_pair,
            val_data,
            test_data,
            output_path=run_path,
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            early_stop_criteria=early_stop_criteria,
            model_args=model_args,
            loss_function=loss_function
        )

        accum_loss.append(loss)
        accum_accuracy.append(accuracy_score)
        accum_auprc.append(auprc_score)
        accum_auroc.append(auc_score)
        accum_f1score.append(f1_score)

    # Save summary statistics
    with open(f"{output_path}/summary.json", "w") as f:
        json.dump(
            {
                "mean_loss": float(np.mean(accum_loss)),
                "mean_accuracy": float(np.mean(accum_accuracy)),
                "mean_auprc": float(np.mean(accum_auprc)),
                "mean_auroc": float(np.mean(accum_auroc)),
                "mean_f1score": float(np.mean(accum_f1score)),
                "std_loss": float(np.std(accum_loss)),
                "std_accuracy": float(np.std(accum_accuracy)),
                "std_auprc": float(np.std(accum_auprc)),
                "std_auroc": float(np.std(accum_auroc)),
                "std_f1score": float(np.std(accum_f1score)),
            }, f, indent=4,
        )

if __name__ == "__main__":
    core_function()