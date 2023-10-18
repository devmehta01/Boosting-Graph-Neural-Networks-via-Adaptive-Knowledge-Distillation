for data in "cora" #"pubmed"
do
    for encoder in "sage-link" #"sage" "gat" "gcn"
    do
    python train_supervised.py --flagfile=config-boosting/$data.cfg --encoder=$encoder --classification_or_linkpred="linkpred" --lr=5e-3
    done
done