# Description: This file contains the loss functions used in the training of the model.

def calculate_loss(model, x, wc, sel, weights_wc, weights_sel):
    # RMSE loss
    wc_hat, sel_hat = model(x)
    loss_wc = (wc - wc_hat).pow(2)
    loss_sel = (sel - sel_hat).pow(2)
    loss = (loss_wc * weights_wc + loss_sel * weights_sel).mean()
    return loss
