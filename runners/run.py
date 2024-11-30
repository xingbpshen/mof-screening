import torch
from datasets import get_dataset
from models.wrapper import Model
import torch.utils.data
from functions import get_optimizer, compute_weights, compute_metrics, plot_gt_pred, plot_x
from functions.losses import calculate_loss
import os
from utils import SimpleLogger, load_latest_model


class Runner:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        if args.device is None:
            device = (
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.device = device
        else:
            self.device = args.device
        self.log_path = os.path.join(args.exp, args.doc)
        self.local_logger = SimpleLogger(self.log_path)

    def train(self):
        # Log args and config with local logger
        self.local_logger.log(str(self.args))
        self.local_logger.log(str(self.config))

        # Get datasets
        if self.config.data.dataset == "MOF":
            train_dataset, val_dataset, _ = get_dataset(self.args, self.config)
        else:
            raise ValueError("Unknown dataset in train()")
        # Get data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            drop_last=False
        )
        # Get model
        model = Model(self.args, self.config).to(self.device)
        # Data parallelism
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        # Get optimizer
        optimizer = get_optimizer(self.config, model.parameters())
        # Train
        model.train()
        total_train_loss = 0
        best_val_loss = float("inf")
        best_saved_model_epoch = 0
        variable_validation_freq = self.config.training.validation_freq
        patient_cnt = 0
        for epoch in range(self.config.training.n_epochs):
            for i, (x, wc, sel, weights_wc, weights_sel) in enumerate(train_loader):

                # Add noise to the input
                if self.config.training.add_input_noise > 0.0:
                    x = x + self.config.training.add_input_noise * torch.randn_like(x)

                # Density reweighing
                if self.config.training.density_reweighing:
                    weights_wc, weights_sel = compute_weights(wc.clone()), compute_weights(sel.clone())

                x, wc, sel, weights_wc, weights_sel = x.to(self.device), wc.to(self.device), sel.to(
                    self.device), weights_wc.to(self.device), weights_sel.to(self.device)
                optimizer.zero_grad()

                if self.config.training.density_reweighing:
                    loss = calculate_loss(model, x, wc, sel, weights_wc, weights_sel)
                else:
                    loss = calculate_loss(model, x, wc, sel, 1, 1)

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
            total_train_loss /= len(train_dataset)

            # Logging
            msg = f"Epoch {epoch}, train loss: {total_train_loss}"
            self.local_logger.log(msg)

            # Validation for early stopping
            if epoch % variable_validation_freq == 0 and epoch > 0:
                # Get validation data loader
                val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=self.config.validation.batch_size,
                    shuffle=False,
                    drop_last=False
                )
                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for i, (x, wc, sel, _, _) in enumerate(val_loader):
                        x, wc, sel = x.to(self.device), wc.to(self.device), sel.to(self.device)
                        # No need to density reweigh validation data
                        loss = calculate_loss(model, x, wc, sel, 1, 1)
                        total_val_loss += loss.item()
                total_val_loss /= len(val_dataset)

                # Logging
                msg = f"Epoch {epoch}, val loss: {total_val_loss}"
                self.local_logger.log(msg)

                if total_val_loss <= best_val_loss:
                    best_val_loss = total_val_loss
                    patient_cnt = 0
                    variable_validation_freq = self.config.training.validation_freq
                    best_saved_model_epoch = epoch
                    # Save model
                    self.local_logger.save_model_checkpoint(model, epoch)
                else:
                    variable_validation_freq = 1
                    if patient_cnt >= self.config.training.patience:
                        # End training
                        # Logging
                        msg=f"Early stopping at epoch {epoch}, best saved model at epoch {best_saved_model_epoch}"
                        self.local_logger.log(msg)
                        break
                    else:
                        patient_cnt += 1
                model.train()

    def test(self):
        # Get model
        model = Model(self.args, self.config).to(self.device)
        model.to(self.device)
        # Data parallelism
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = load_latest_model(model, self.log_path)
        model.eval()

        # Get datasets
        if self.config.data.dataset == "MOF":
            _, _, test_dataset = get_dataset(self.args, self.config)
        else:
            raise ValueError("Unknown dataset in test()")

        # Perform testing steps
        # Evaluate testset loss
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.testing.batch_size,
            shuffle=False,
            drop_last=False
        )
        total_test_loss = 0
        # Collect ground truth and predicted values
        gt_wc = []
        gt_sel = []
        hat_wc = []
        hat_sel = []
        if self.args.mc_dropout > 0:
            uncertainty_wc = []
            uncertainty_sel = []
        with torch.no_grad():
            for i, (x, wc, sel, _, _) in enumerate(test_loader):
                x, wc, sel = x.to(self.device), wc.to(self.device), sel.to(self.device)
                # No need to density reweigh testing data
                loss = calculate_loss(model, x, wc, sel, 1, 1)
                total_test_loss += loss.item()
                gt_wc.append(wc)
                gt_sel.append(sel)
                if self.args.mc_dropout > 0:
                    # TODO: Implement MC Dropout based on model.sample(x=x, n_samples=self.args.mc_dropout)
                    wcs, sels = model.sample(x=x, n_samples=self.args.mc_dropout)   # wcs, sels: (batch_size, n_samples)
                    mean_wc, mean_sel = wcs.mean(dim=1), sels.mean(dim=1)
                    unbiased_var_wc, unbiased_var_sel = wcs.var(dim=1, unbiased=True), sels.var(dim=1, unbiased=True)
                    hat_wc.append(mean_wc)
                    hat_sel.append(mean_sel)
                    uncertainty_wc.append(unbiased_var_wc)
                    uncertainty_sel.append(unbiased_var_sel)
                else:
                    hat_wc.append(model(x)[0])
                    hat_sel.append(model(x)[1])
        total_test_loss /= len(test_dataset)
        print(f"Test loss: {total_test_loss}")

        # Convert to numpy and compute metrics
        gt_wc = torch.cat(gt_wc, dim=0).cpu().numpy()
        gt_sel = torch.cat(gt_sel, dim=0).cpu().numpy()
        hat_wc = torch.cat(hat_wc, dim=0).cpu().numpy()
        hat_sel = torch.cat(hat_sel, dim=0).cpu().numpy()
        if self.args.mc_dropout > 0:
            uncertainty_wc = torch.cat(uncertainty_wc, dim=0).cpu().numpy()
            uncertainty_sel = torch.cat(uncertainty_sel, dim=0).cpu().numpy()

        # Metrics computation
        metrics_wc = compute_metrics(gt_wc, hat_wc)
        metrics_sel = compute_metrics(gt_sel, hat_sel)
        # Logging
        self.local_logger.log(f"Metrics for wc: \n {metrics_wc}")
        self.local_logger.log(f"Metrics for sel: \n {metrics_sel}")
        self.local_logger.save_tensor_as_np(gt_wc, "gt_wc.npy")
        self.local_logger.save_tensor_as_np(hat_wc, "hat_wc.npy")
        self.local_logger.save_tensor_as_np(gt_sel, "gt_sel.npy")
        self.local_logger.save_tensor_as_np(hat_sel, "hat_sel.npy")
        if self.args.mc_dropout > 0:
            self.local_logger.save_tensor_as_np(uncertainty_wc, "uncertainty_wc.npy")
            self.local_logger.save_tensor_as_np(uncertainty_sel, "uncertainty_sel.npy")
        # Plot
        plot_gt_pred(gt_wc, hat_wc, 'Capacity', os.path.join(self.log_path, 'gt_pred_wc.pdf'))
        plot_gt_pred(gt_sel, hat_sel, 'Selectivity', os.path.join(self.log_path, 'gt_pred_sel.pdf'))
