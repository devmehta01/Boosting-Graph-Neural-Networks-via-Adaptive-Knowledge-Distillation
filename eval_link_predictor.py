def eval_link_predictor(model, data, edge_label, edge_label_index):

    model.eval()
    z = model.forward(data, edge_label, edge_label_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(data.edge_label.cpu().detach().numpy(), out.cpu().detach().numpy())