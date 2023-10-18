#### train gcn student
#CUDA_VISIBLE_DEVICES=3 python train.py --flagfile=config-boosting/citeseer.cfg --teacher=gat --student=gcn --gamma=0.2 --alpha=10.0 --lr=0.005
#CUDA_VISIBLE_DEVICES=3 python train.py --flagfile=config-boosting/pubmed.cfg --teacher=sage --student=gcn --gamma=0.2 --alpha=0.2 --lr=0.05

### train sage student
#python train.py --flagfile=config-boosting/citeseer.cfg --teacher=gcn --student=sage --gamma=0.1 --alpha=5.0 --lr=0.05
#CUDA_VISIBLE_DEVICES=3 python train.py --flagfile=config-boosting/pubmed.cfg --teacher=gcn --student=sage --gamma=0.2 --alpha=10.0 --lr=0.05

### train gat student
#CUDA_VISIBLE_DEVICES=3 python train.py --flagfile=config-boosting/citeseer.cfg --teacher=gcn --student=gat --gamma=0.1 --alpha=5.0 --lr=0.01
#CUDA_VISIBLE_DEVICES=3 python train.py --flagfile=config-boosting/pubmed.cfg --teacher=sage --student=gat --gamma=0.1 --alpha=1.0 --lr=0.05



#python train.py --flagfile=config-boosting/citeseer.cfg --teacher=sage-link --student=sage-link --gamma=0.1 --alpha=5.0 --lr=1e-5 --classification_or_linkpred="linkpred" --boosting=0
#python train.py --flagfile=config-boosting/citeseer.cfg --teacher=sage-link --student=gat-link --gamma=1 --alpha=0.5 --lr=1e-3 --classification_or_linkpred="linkpred" --boosting=0 --T=1 --weight_decay=1e-5

python train.py --flagfile=config-boosting/cora.cfg --teacher=sage-link --student=gcn-link --gamma=1 --alpha=0.5 --lr=1e-3 --classification_or_linkpred="linkpred" --boosting=0 --T=1 --weight_decay=1e-5