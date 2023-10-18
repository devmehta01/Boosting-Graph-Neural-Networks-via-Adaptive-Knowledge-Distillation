def train_link_predictor(model, train_data, val_data, optimizer, criterion, n_epochs=100):

    for epoch in range(1, n_epochs + 1):

        model.train()
        optimizer.zero_grad()
        
        # sampling training negatives for every training epoch
        from torch_geometric.utils import negative_sampling
        import torch
        # neg_edge_index = negative_sampling(
        #     edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        #     num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

        # edge_label_index = torch.cat(
        #     [train_data.edge_label_index, neg_edge_index],
        #     dim=-1,
        # )
        # edge_label = torch.cat([
        #     train_data.edge_label,
        #     train_data.edge_label.new_zeros(neg_edge_index.size(1))
        # ], dim=0)

        loss = model.forward(train_data, train_data.edge_label, train_data.edge_label_index)
        #print(epoch)
        

        #out = model.decode(z, train_data.edge_label_index).view(-1)
        #loss = criterion(out, train_data.edge_label)
        loss.backward()
        #print(model.trainable_parameters())
        optimizer.step()
        

        #from eval_link_predictor import eval_link_predictor 
        #val_auc = eval_link_predictor(model, val_data, val_data.edge_label, val_data.edge_label_index)
        #print("epochnum: ", epoch)
        val_auc = model.evaluate(val_data, val_data.edge_label, val_data.edge_label_index)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val AUC: {val_auc:.3f}")
            #print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}")

    return model, train_data.edge_label, train_data.edge_label_index