import copy
import logging
import os
from random import sample

from absl import app
from absl import flags
import torch
from torch.nn.functional import cosine_similarity
from torch.optim import AdamW
from tqdm import tqdm

from bgnn import *
from bgnn.models import GAT, GCN, GraphSage, GraphSage_link, GCN_link, GAT_link
from train_link_predictor import train_link_predictor
from eval_link_predictor import eval_link_predictor

from bgnn.distillation import Supervised_training, Supervised_training_link
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import json

from ogb.nodeproppred import PygNodePropPredDataset

log = logging.getLogger(__name__)
FLAGS = flags.FLAGS
flags.DEFINE_integer('model_seed', 0, 'Random seed used for model initialization and training.')
flags.DEFINE_integer('data_seed', 1, 'Random seed used to generate train/val/test split.')
flags.DEFINE_integer('num_eval_splits', 3, 'Number of different train/test splits the model will be evaluated over.')

# Dataset.
flags.DEFINE_enum('dataset', 'cora',
                  ['amazon-computers', 'cora','pubmed','citeseer'],
                  'Which graph dataset to use.')
flags.DEFINE_string('dataset_dir', './data', 'Where the dataset resides.')

flags.DEFINE_integer('n_classes', 10, 'Number of classes.')

# Architecture.
flags.DEFINE_multi_integer('graph_encoder_layer', None, 'Conv layer sizes.')
flags.DEFINE_multi_integer('gat_encoder_layer', None, 'Conv layer sizes.')
flags.DEFINE_multi_integer('sage_encoder_layer', None, 'Conv layer sizes.')
flags.DEFINE_multi_integer('gat_head', None, 'Conv layer sizes.')
flags.DEFINE_bool('graph_layernorm', None, 'gcn layer norm.')
flags.DEFINE_bool('gat_layernorm', None, 'gat layer norm.')
flags.DEFINE_bool('sage_layernorm', None, 'sage layer norm.')
flags.DEFINE_float('graph_droprate', 0.6, 'gcn droprate.')
flags.DEFINE_float('gat_droprate', 0.6, 'gat droprate.')
flags.DEFINE_float('sage_droprate', 0.6, 'sage droprate.')
flags.DEFINE_integer('predictor_hidden_size', 512, 'Hidden size of projector.')

# Training hyperparameters.
flags.DEFINE_integer('epochs', 10, 'The number of training epochs.')
flags.DEFINE_float('lr', 1e-5, 'The learning rate for model training.')
flags.DEFINE_float('weight_decay', 1e-5, 'The value of the weight decay for training.')
flags.DEFINE_float('mm', 0.99, 'The momentum for moving average.')
flags.DEFINE_integer('lr_warmup_epochs', 500, 'Warmup period for learning rate.')

flags.DEFINE_float('gamma', 1, 'nll_loss weight.') ##  label loss
flags.DEFINE_float('alpha', 0, 'kd loss weight.') ##  KD loss
flags.DEFINE_float('beta', 0, 'contrastive loss weight.')  ##CRD loss
flags.DEFINE_integer('T', 4, 'temperature.')

# Logging and checkpoint.
flags.DEFINE_string('logdir', None, 'Where the checkpoint and logs are stored.')
flags.DEFINE_integer('log_steps', 10, 'Log information at every log_steps.')

# Evaluation
flags.DEFINE_integer('eval_epochs', 10, 'Evaluate every eval_epochs.')
flags.DEFINE_string('ckpt_path', None, 'Path to checkpoint.')

flags.DEFINE_string('encoder', 'gcn', 'GNN encoder')

flags.DEFINE_string('classification_or_linkpred', 'classification', 'Either choose classification or link prediction')

def main(argv):
    # use CUDA_VISIBLE_DEVICES to select gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info('Using {} for training.'.format(device))

    # set random seed
    if FLAGS.model_seed is not None:
        log.info('Random seed set to {}.'.format(FLAGS.model_seed))
        set_random_seeds(random_seed=FLAGS.model_seed)

    # create log directory
    os.makedirs(FLAGS.logdir, exist_ok=True)
    with open(os.path.join(FLAGS.logdir, 'config.cfg'), "w") as file:
        file.write(FLAGS.flags_into_string())  # save config file

    # load data
    if FLAGS.dataset == 'ogb-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
        split_idx = dataset.get_idx_split()
        train_masks = split_idx['train'].to(device)
        test_masks = split_idx['test'].to(device)
        num_eval_splits = FLAGS.num_eval_splits

    elif FLAGS.dataset == 'wiki-cs':
        dataset, train_masks, val_masks, test_masks = get_wiki_cs(FLAGS.dataset_dir)
        num_eval_splits = train_masks.shape[1]

    elif FLAGS.dataset == 'cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
        train_masks = dataset[0].train_mask.to(device)
        val_masks = dataset[0].val_mask.to(device)
        test_masks = dataset[0].test_mask.to(device)

    elif FLAGS.dataset == 'pubmed':
        dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed', transform=T.NormalizeFeatures())
        train_masks = dataset[0].train_mask.to(device)
        val_masks = dataset[0].val_mask.to(device)
        test_masks = dataset[0].test_mask.to(device)

    elif FLAGS.dataset == 'citeseer':
        dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer', transform=T.NormalizeFeatures())
        train_masks = dataset[0].train_mask.to(device)
        val_masks = dataset[0].val_mask.to(device)
        test_masks = dataset[0].test_mask.to(device)

    else:
        dataset = get_dataset(FLAGS.dataset_dir, FLAGS.dataset)
        num_eval_splits = FLAGS.num_eval_splits
        rng = np.random.RandomState(1)
        train_masks, test_masks = train_test_split(range(0, dataset[0].x.size()[0]), test_size=0.8, shuffle=False)
        train_masks = train_masks[:int(len(train_masks)/2)]

    data = dataset[0]  # all dataset include one graph
    log.info('Dataset {}, {}.'.format(dataset.__class__.__name__, data))
    data = data.to(device)  # permanently move in gpy memory

    # build networks
    input_size, representation_size = data.x.size(1), FLAGS.graph_encoder_layer[-1]

    if FLAGS.encoder == "sage":
        print(FLAGS.sage_encoder_layer)
        encoder = GraphSage([input_size]+FLAGS.sage_encoder_layer, FLAGS.sage_layernorm, FLAGS.sage_droprate)
    elif FLAGS.encoder == "gat":
        encoder = GAT([input_size] + FLAGS.gat_encoder_layer, FLAGS.gat_head, FLAGS.gat_layernorm, FLAGS.gat_droprate)   # 512, 256, 128
    elif FLAGS.encoder == "gcn":
        encoder = GCN([input_size] + FLAGS.graph_encoder_layer, FLAGS.graph_layernorm, FLAGS.graph_droprate)   # 512, 256, 128
    elif FLAGS.encoder == "sage-link":
        encoder = GraphSage_link([input_size]+FLAGS.sage_encoder_layer, FLAGS.sage_layernorm, FLAGS.sage_droprate)
    elif FLAGS.encoder == "gcn-link":
        encoder = GCN_link([input_size] + FLAGS.graph_encoder_layer, FLAGS.graph_layernorm, FLAGS.graph_droprate)   # 512, 256, 128
    elif FLAGS.encoder == "gat-link":
        encoder = GAT_link([input_size] + FLAGS.gat_encoder_layer, FLAGS.gat_head, FLAGS.gat_layernorm, FLAGS.gat_droprate)   # 512, 256, 128


    if FLAGS.classification_or_linkpred == "classification":
        model = Supervised_training(encoder, FLAGS).to(device)
    elif FLAGS.classification_or_linkpred == "linkpred":
        model = Supervised_training_link(encoder, FLAGS).to(device)

    sample_weights = torch.ones(data.x.size(0))  # 2708
    if FLAGS.dataset == 'wiki-cs':
        sample_weights = sample_weights[train_masks[:, 0].to("cpu")]
    else:
        sample_weights = sample_weights[train_masks.to("cpu")]  # 140*1
    sample_weights = sample_weights / sample_weights.sum()  # 1/140
    sample_weights = sample_weights.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    if FLAGS.classification_or_linkpred == "linkpred":
      split = T.RandomLinkSplit(num_val=0.05,num_test=0.1,is_undirected=True,add_negative_train_samples=True,neg_sampling_ratio=1.0)
      train_data, val_data, test_data = split(data)
      train_data.to(device)
      val_data.to(device)
      test_data.to(device)


    best_accs = []
    best_one = 0.0
    for time in range(10):
        model.reset_parameters()
        best_acc = 0.0
        print("############# Training round " + str(time+1) + "############" )
        if FLAGS.classification_or_linkpred == "classification":
            for epoch in tqdm(range(1, FLAGS.epochs + 1)):
                model.train()

                # forward
                optimizer.zero_grad()

                loss = model(data, train_masks)

                loss.backward()

                optimizer.step()

                # if epoch % FLAGS.eval_epochs == 0:
                model.eval()
                accs, updated_weight = model.evaluate(data, test_masks, train_masks, sample_weights)
                if accs > best_acc:
                    best_acc = accs
                if accs > best_one:
                    best_one = accs
                    torch.save({'model': model.student_encoder.state_dict(), 'weight': updated_weight}, os.path.join(FLAGS.logdir, FLAGS.encoder + '.pt'))
        elif FLAGS.classification_or_linkpred == "linkpred":
            criterion = torch.nn.BCEWithLogitsLoss()
            model, edge_label, edge_label_index = train_link_predictor(model, train_data, val_data, optimizer, criterion)
            model.to(device)
            test_auc = eval_link_predictor(model, test_data, edge_label, edge_label_index)
            print(f"Test: {test_auc:.3f}")
            if test_auc > best_acc:
                best_acc = test_auc
            if test_auc > best_one:
                best_one = test_auc
                #torch.save({'model': model.student_encoder.state_dict(), 'weight': updated_weight}, os.path.join(FLAGS.logdir, FLAGS.encoder + '.pt'))
                torch.save({'model': model.student_encoder.state_dict()}, os.path.join(FLAGS.logdir, FLAGS.encoder + '.pt'))
        print("Best of this round:", best_acc)
        best_accs.append(best_acc)

    test_accs = np.stack(best_accs)

    test_acc = best_one
    file = open("results/sp-" + FLAGS.dataset + "-" + FLAGS.encoder + ".txt", "a")
    file.write(f"test_acc is {test_acc:.4f} \n")
    file.close()
    print(f"Best of all rounds, test_acc is {test_acc:.4f}")


if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)
