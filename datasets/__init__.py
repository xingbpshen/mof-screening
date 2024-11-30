from sklearn.model_selection import train_test_split
from datasets.mof import MOFDataset
import os
import numpy as np
from functions import compute_weights
import torch


def get_dataset(args, config):
    if config.data.dataset == "MOF":
        train_path = os.path.join(config.data.data_dir, 'train_npy/')
        label_path = os.path.join(config.data.data_dir, 'output.txt')
        data_list = np.array(os.listdir(train_path))
        with open(label_path) as f:
            y_csv = list(map(lambda x: x.strip().split(','), f.readlines()))
        y_label_wc, y_label_sel = {}, {}
        for i in y_csv:
            try:
                y_label_wc[i[0][:-4]] = i[1]
                y_label_sel[i[0][:-4]] = i[2]
            except:
                continue
        x_path, y_wc, y_sel = [], [], []
        for i in data_list:
            if i[:-4] in y_label_wc:
                x_path.append(os.path.join(train_path, i))
                y_wc.append(float(y_label_wc[i[:-4]]))
                y_sel.append(float(y_label_sel[i[:-4]]))
        y_wc, y_sel = np.array(y_wc), np.array(y_sel)

        # Load stats
        x_mean, x_std = config.data.x_min, (config.data.x_max - config.data.x_min)
        y_wc_mean, y_wc_std = config.data.wc_mean, config.data.wc_std
        y_sel_mean, y_sel_std = config.data.sel_mean, config.data.sel_std

        # # Save y_wc and y_sel
        # y_wc = (y_wc - y_wc_mean) / y_wc_std
        # y_sel = (y_sel - y_sel_mean) / y_sel_std
        # np.save(os.path.join('./figs/', 'y_wc.npy'), y_wc)
        # np.save(os.path.join('./figs/', 'y_sel.npy'), y_sel)
        # exit()

        # Split the data into training, validation, and testing sets, according to the specified ratios in config.data
        x_train, x_temp, y_wc_train, y_wc_temp, y_sel_train, y_sel_temp = train_test_split(
            x_path, y_wc, y_sel, test_size=(1.0 - config.data.train_ratio), random_state=42, shuffle=True
        )
        x_val, x_test, y_wc_val, y_wc_test, y_sel_val, y_sel_test = train_test_split(
            x_temp, y_wc_temp, y_sel_temp, test_size=(config.data.test_ratio / (1.0 - config.data.train_ratio)),
            random_state=42, shuffle=True
        )

        ## Compute weights on the whole set is intracable, so we compute weights on the batch level
        # weights_wc = compute_weights(y_wc_train)
        # torch.save(weights_wc, os.path.join(config.data.data_dir, 'train_weights_wc.pt'))
        # weights_sel = compute_weights(y_sel_train)
        # torch.save(weights_sel, os.path.join(config.data.data_dir, 'train_weights_sel.pt'))
        train_dataset = MOFDataset(x_train, y_wc_train, y_sel_train, x_stats=(x_mean, x_std), wc_stats=(y_wc_mean, y_wc_std),
                                   sel_stats=(y_sel_mean, y_sel_std), input_normalization=config.data.input_normalization,
                                   target_normalization=config.data.target_normalization)
        val_dataset = MOFDataset(x_val, y_wc_val, y_sel_val, x_stats=(x_mean, x_std), wc_stats=(y_wc_mean, y_wc_std),
                                 sel_stats=(y_sel_mean, y_sel_std), input_normalization=config.data.input_normalization,
                                   target_normalization=config.data.target_normalization)
        test_dataset = MOFDataset(x_test, y_wc_test, y_sel_test, x_stats=(x_mean, x_std), wc_stats=(y_wc_mean, y_wc_std),
                                  sel_stats=(y_sel_mean, y_sel_std), input_normalization=config.data.input_normalization,
                                   target_normalization=config.data.target_normalization)
        print(f"Train size: {len(train_dataset)}")
        print(f"Validation size: {len(val_dataset)}")
        print(f"Test size: {len(test_dataset)}")

        return train_dataset, val_dataset, test_dataset

    else:
        raise NotImplementedError(f"Invalid dataset {config.data.dataset}.")
