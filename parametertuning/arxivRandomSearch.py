import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from parametertuning.randomsearch import RandomSearchJob
from codes.ioutils import remove_all_files

def HypeParameterSpace(data_name):
    learning_rate = {'name': 'learning_rate', 'type': 'range', 'bounds': [0.001, 0.01]}
    feat_drop = {'name': 'fea_drop', 'type': 'range', 'bounds': [0.1, 0.6]}
    att_drop = {'name': 'att_drop', 'type': 'range', 'bounds': [0.1, 0.6]}
    # hidden_dim = {'name': 'hidden_dim', 'type': 'choice', 'values': [128, 256]}
    hidden_dim = {'name': 'hidden_dim', 'type': 'fixed', 'value': 128}
    project_dim = {'name': 'project_dim', 'type': 'fixed', 'value': -1}
    edge_drop = {'name': 'edge_drop', 'type': 'range', 'bounds': [0.1, 0.6]}
    num_heads = {'name': 'num_heads', 'type': 'fixed', 'value': 8}
    top_k = {'name': 'top_k', 'type': 'choice', 'values': [5, 10, 15]}
    topk_type = {'name': 'topk_type', 'type': 'fixed', 'value': 'local'}
    epochs = {'name': 'epochs', 'type': 'fixed', 'value': 1000}
    num_layers = {'name': 'num_layers', 'type': 'fixed', 'value': 3}
    self_loop = {'name': 'self_loop', 'type': 'fixed', 'value': 1}
    alpha = {'name': 'alpha', 'type': 'range', 'bounds': [0.1, 0.6]}
    hop_num = {'name': 'hop_num', 'type': 'choice', 'values': [3, 4]}
    neg_slope = {'name': 'negative_slope', 'type': 'fixed', 'value': 0.2}
    adam_weight_decay = {'name': 'adam_weight_decay', 'type': 'range', 'bounds': [1e-5, 1e-4], "log_scale": True}
    data_name = {'name': 'dataset', 'type': 'fixed', 'value': data_name}
    #++++++++++++++++++++++++++++++++++
    search_space = [learning_rate, adam_weight_decay, att_drop, feat_drop, project_dim, topk_type, edge_drop, self_loop,
                      top_k, hop_num, num_heads, num_layers, alpha,  neg_slope, epochs, hidden_dim, data_name]
    search_space = dict((x['name'], x) for x in search_space)
    return search_space

def generate_random_search_bash(task_num, data_name):
    bash_save_path = '../' + data_name + '_jobs/'
    if os.path.exists(bash_save_path):
        remove_all_files(bash_save_path)
    if bash_save_path and not os.path.exists(bash_save_path):
        os.makedirs(bash_save_path)
    search_space = HypeParameterSpace(data_name)
    random_search_job =RandomSearchJob(data_name=data_name, search_space=search_space)
    for i in range(task_num):
        task_id, parameter_id = random_search_job.single_task_trial(i+1)
        with open(bash_save_path + 'run_' + task_id +'.sh', 'w') as rsh_i:
            command_i = 'bash runarxiv.sh ' + parameter_id
            rsh_i.write(command_i)
    print('{} jobs have been generated'.format(task_num))

if __name__ == '__main__':
    generate_random_search_bash(data_name='ogbn_arxiv', task_num=60)