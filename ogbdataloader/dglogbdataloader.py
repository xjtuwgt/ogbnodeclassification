import sys
import os


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import pandas as pd
import os.path as osp
import torch
from dgl.data.utils import load_graphs

class DglNodePropPredDataset(object):
    def __init__(self, name, root):
        super(DglNodePropPredDataset, self).__init__()
        self.name = name ## original name, e.g., ogbn-proteins
        self.dir_name = "_".join(name.split("_")) + "_dgl" ## replace hyphen with underline, e.g., ogbn_proteins_dgl
        self.root = osp.join(root, self.dir_name)
        print(self.root)
        self.pre_process()

    def pre_process(self):
        processed_dir = osp.join(self.root, 'processed')
        pre_processed_file_path = osp.join(processed_dir, 'dgl_data_processed')
        self.graph, label_dict = load_graphs(pre_processed_file_path)
        self.labels = label_dict['labels']

    def get_idx_split(self):
        if 'arxiv' in self.name:
            split_type = 'time'
        elif 'protein' in self.name:
            split_type = 'species'
        else:
            return
        path = osp.join(self.root, "split", split_type)
        train_idx = pd.read_csv(osp.join(path, "train.csv.gz"), compression="gzip", header = None).values.T[0]
        valid_idx = pd.read_csv(osp.join(path, "valid.csv.gz"), compression="gzip", header = None).values.T[0]
        test_idx = pd.read_csv(osp.join(path, "test.csv.gz"), compression="gzip", header = None).values.T[0]
        return {"train": torch.tensor(train_idx, dtype=torch.long), "valid": torch.tensor(valid_idx, dtype=torch.long), "test": torch.tensor(test_idx, dtype=torch.long)}

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.graph[idx], self.labels

    def __len__(self):
        return 1

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))

if __name__ == "__main__":
    dgl_dataset = DglNodePropPredDataset(name = "ogbn_arxiv", root='../dataset')
    splitted_index = dgl_dataset.get_idx_split()
    print(dgl_dataset[0])
    print(splitted_index)
