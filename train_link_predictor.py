def train_link_predictor(model, train_data, val_data, optimizer, criterion, n_epochs=100):

    for epoch in range(1, n_epochs + 1):

        model.train()
        optimizer.zero_grad()
        
        # sampling training negatives for every training epoch
        from torch_geometric.utils import negative_sampling
        import torch

        z = model.forward(train_data, train_data.edge_label, train_data.edge_label_index)

        

        out = model.decode(z, train_data.edge_label_index).view(-1)
        #print(out)
        loss = criterion(out, train_data.edge_label)
        loss.backward()
        optimizer.step()

        from eval_link_predictor import eval_link_predictor 
        val_auc = eval_link_predictor(model, val_data, val_data.edge_label, val_data.edge_label_index)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val AUC: {val_auc:.3f}")

    return model, train_data.edge_label, train_data.edge_label_index