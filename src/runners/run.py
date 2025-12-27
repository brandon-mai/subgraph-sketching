"""
main module
"""
import argparse
import time
import warnings
from math import inf
import sys
import random

sys.path.insert(0, '..')

import numpy as np
import torch
from ogb.linkproppred import Evaluator
import matplotlib.pyplot as plt
try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


torch.set_printoptions(precision=4)
import wandb
# when generating subgraphs the supervision edge is deleted, which triggers a SparseEfficiencyWarning, but this is
# not a performance bottleneck, so suppress for now
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

from src.data import get_data, get_loaders
from src.models.elph import ELPH, BUDDY
from src.models.seal import SEALDGCNN, SEALGCN, SEALGIN, SEALSAGE
from src.utils import ROOT_DIR, print_model_params, select_embedding, str2bool
from src.wandb_setup import initialise_wandb
from src.runners.train import get_train_func
from src.runners.inference import test

def print_results_list(results_list):
    for idx, res in enumerate(results_list):
        print(f'repetition {idx}: test {res[0]:.2f}, val {res[1]:.2f}, train {res[2]:.2f}')

def set_seed(seed):
    """
    setting a random seed for reproducibility and in accordance with OGB rules
    @param seed: an integer seed
    @return: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run(args):
    args = initialise_wandb(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"executing on {device}")
    results_list = []
    train_func = get_train_func(args)
    for rep in range(args.reps):
        set_seed(rep)
        dataset, splits, directed, eval_metric = get_data(args)
        train_loader, train_eval_loader, val_loader, test_loader = get_loaders(args, dataset, splits, directed)
        if args.dataset_name.startswith('ogbl'):  # then this is one of the ogb link prediction datasets
            evaluator = Evaluator(name=args.dataset_name)
        else:
            evaluator = Evaluator(name='ogbl-ppa')  # this sets HR@100 as the metric
        emb = select_embedding(args, dataset.data.num_nodes, device)
        model, optimizer = select_model(args, dataset, emb, device)
        val_res = test_res = best_epoch = 0
        epoch_losses = {'train': [], 'val': [], 'epochs': []}
        pull_targets = None
        
        # PULL checks and dynamic setup
        current_k = 0
        k_growth = 0
        if args.use_pull:
            if hasattr(train_loader.dataset, 'labels'):
                # Check if labels is tensor or list training dataset usually has .labels
                if torch.is_tensor(train_loader.dataset.labels):
                    num_pos_edges = (train_loader.dataset.labels == 1).sum().item()
                else:
                    # Assuming numpy or list
                    num_pos_edges = int(sum(np.array(train_loader.dataset.labels) == 1))
            else:
                num_pos_edges = args.pull_k # Fallback to arg if logic fails
            
            current_k = num_pos_edges
            k_growth = int(0.05 * num_pos_edges)
            print(f"PULL Enabled: Ep={num_pos_edges}, Initial K={current_k}, Growth={k_growth}, Interval={args.pull_interval}")

        print(f'running repetition {rep}')
        # if rep == 0:
        #     print_model_params(model)
        for epoch in range(args.epochs):
            t0 = time.time()
            
            # PULL updates
            if args.use_pull and (epoch % args.pull_interval == 0):
                if epoch == 0:
                    pass 
                else:
                    # Update targets
                    print(f"Updating PULL targets with K={current_k}")
                    pull_targets = update_pull_targets(model, train_loader, device, current_k)
                    current_k += k_growth
            
            loss = train_func(model, optimizer, train_loader, args, device, pull_targets=pull_targets)
            if (epoch + 1) % args.eval_steps == 0:
                results, losses = test(model, evaluator, train_eval_loader, val_loader, test_loader, args, device,
                               eval_metric=eval_metric)
                
                # Track losses
                epoch_losses['train'].append(loss) # Use training loss from train_func (running avg)
                epoch_losses['val'].append(losses['val'])
                epoch_losses['epochs'].append(epoch)

                for key, result in results.items():
                    train_res, tmp_val_res, tmp_test_res = result
                    if tmp_val_res > val_res:
                        val_res = tmp_val_res
                        test_res = tmp_test_res
                        best_epoch = epoch
                    res_dic = {f'rep{rep}_loss': loss, f'rep{rep}_Train' + key: 100 * train_res,
                               f'rep{rep}_Val' + key: 100 * val_res, f'rep{rep}_tmp_val' + key: 100 * tmp_val_res,
                               f'rep{rep}_tmp_test' + key: 100 * tmp_test_res,
                               f'rep{rep}_Test' + key: 100 * test_res, f'rep{rep}_best_epoch': best_epoch,
                               f'rep{rep}_epoch_time': time.time() - t0, 'epoch_step': epoch,
                               f'rep{rep}_Val_Loss': losses['val']}
                    if args.wandb:
                        wandb.log(res_dic)
                    to_print = f'Epoch: {epoch:02d}, Best epoch: {best_epoch}, Loss: {loss:.4f}, Val Loss: {losses["val"]:.4f}, Train: {100 * train_res:.2f}%, Valid: ' \
                               f'{100 * val_res:.2f}%, Test: {100 * test_res:.2f}%, epoch time: {time.time() - t0:.1f}'
                    print(key)
                    print(to_print)
        
        # Plotting Learning Curve
        plt.figure()
        plt.plot(epoch_losses['epochs'], epoch_losses['train'], label='Train Loss')
        plt.plot(epoch_losses['epochs'], epoch_losses['val'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Learning Curve - Rep {rep}')
        plt.legend()
        plot_path = f'{ROOT_DIR}/saved_models/{args.dataset_name}_rep{rep}_loss.png'
        # Ensure dir exists (already checked before saving model but safe to recheck or assume exists if save_model is on, 
        # but to be sure for plotting regardless of save_model flag:)
        import os
        if not os.path.exists(os.path.dirname(plot_path)):
             os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

        if args.wandb:
             wandb.log({f"learning_curve_rep{rep}": wandb.Image(plot_path)})

        if args.save_model:
            path = f'{ROOT_DIR}/saved_models/{args.dataset_name}'
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(model.state_dict(), path)
            print(f"Model saved to {path}")

            if args.hf_repo_id:
                if HfApi is None:
                    print("Warning: HfApi not imported. Install huggingface_hub to push model.")
                else:
                    try:
                        api = HfApi(token=args.hf_token)
                        # Upload model
                        api.upload_file(
                            path_or_fileobj=path,
                            path_in_repo=f"{args.dataset_name}_model.pt",
                            repo_id=args.hf_repo_id,
                            repo_type="model"
                        )
                        print(f"Model pushed to {args.hf_repo_id}")
                        # Upload plot
                        api.upload_file(
                            path_or_fileobj=plot_path,
                            path_in_repo=f"{args.dataset_name}_rep{rep}_loss.png",
                            repo_id=args.hf_repo_id,
                            repo_type="model"
                        )
                        print(f"Plot pushed to {args.hf_repo_id}")
                    except Exception as e:
                        print(f"Failed to push to Hugging Face: {e}")



def select_model(args, dataset, emb, device):
    if args.model == 'SEALDGCNN':
        model = SEALDGCNN(args.hidden_channels, args.num_seal_layers, args.max_z, args.sortpool_k,
                          dataset, args.dynamic_train, use_feature=args.use_feature,
                          node_embedding=emb).to(device)
    elif args.model == 'SEALSAGE':
        model = SEALSAGE(args.hidden_channels, args.num_seal_layers, args.max_z, dataset.num_features,
                         args.use_feature, node_embedding=emb, dropout=args.dropout).to(device)
    elif args.model == 'SEALGCN':
        model = SEALGCN(args.hidden_channels, args.num_seal_layers, args.max_z, dataset.num_features,
                        args.use_feature, node_embedding=emb, dropout=args.dropout, pooling=args.seal_pooling).to(
            device)
    elif args.model == 'SEALGIN':
        model = SEALGIN(args.hidden_channels, args.num_seal_layers, args.max_z, dataset.num_features,
                        args.use_feature, node_embedding=emb, dropout=args.dropout).to(device)
    elif args.model == 'BUDDY':
        model = BUDDY(args, dataset.num_features, node_embedding=emb).to(device)
    elif args.model == 'ELPH':
        model = ELPH(args, dataset.num_features, node_embedding=emb).to(device)
    else:
        raise NotImplementedError
    parameters = list(model.parameters())
    if args.train_node_embedding:
        torch.nn.init.xavier_uniform_(emb.weight)
        parameters += list(emb.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=args.weight_decay)
    total_params = sum(p.numel() for param in parameters for p in param)
    print(f'Total number of parameters is {total_params}')
    if args.model == 'DGCNN':
        print(f'SortPooling k is set to {model.k}')
    return model, optimizer


@torch.no_grad()
def update_pull_targets(model, train_loader, device, k):
    """
    Construct PULL target matrix.
    Positives (observed) = 1
    Top-K confident Negatives = Predicted Probability (Pseudo-Positive)
    Other Negatives = 0
    """
    model.eval()
    
    data = train_loader.dataset
    if hasattr(data, 'links'):
        links = data.links
        labels = torch.tensor(data.labels)
    else:
        # Fallback for datasets without explicit links attribute, though BUDDY/ELPH use it
        return None

    dataset_len = len(links)
    # Create a non-shuffling loader for consistency
    if hasattr(train_loader, 'batch_size'):
        bs = train_loader.batch_size
    else:
        bs = 1024
        
    seq_loader = DataLoader(range(dataset_len), batch_size=bs, shuffle=False, num_workers=0)
    
    pull_targets = torch.zeros(dataset_len, dtype=torch.float, device=device)
    pull_targets[labels == 1] = 1.0 # Keep positives as 1
    
    neg_mask = (labels == 0)
    if neg_mask.sum() == 0:
        return pull_targets

    preds = []
    indices_list = []
    
    for indices in tqdm(seq_loader, desc="Updating PULL targets", disable=len(seq_loader) <= 1):
        curr_links = links[indices].to(device)
        
        if isinstance(model, ELPH):
            node_features, hashes, cards = model(data.x.to(device), data.edge_index.to(device))
            # Just extract what we need
            batch_node_features = None if node_features is None else node_features[curr_links]
            batch_emb = None if model.node_embedding is None else model.node_embedding.weight[curr_links].to(device)
            if hasattr(model, 'elph_hashes') and model.elph_hashes is not None:
                subgraph_features = model.elph_hashes.get_subgraph_features(curr_links, hashes, cards).to(device)
            else:
                 subgraph_features = torch.zeros((len(indices), 0)).to(device)
            logits = model.predictor(subgraph_features, batch_node_features, batch_emb)
            
        elif isinstance(model, BUDDY):
            # For BUDDY, subgraph features are precomputed in data.subgraph_features
            # But we need them aligned with indices
            subgraph_features = data.subgraph_features[indices].to(device)
            node_features = data.x[curr_links].to(device)
            degrees = data.degrees[curr_links].to(device)
            if hasattr(data, 'RA') and data.RA is not None:
                RA = data.RA[indices].to(device)
            else:
                RA = None
            batch_emb = None 
            if model.node_embedding is not None:
                batch_emb = model.node_embedding.weight[curr_links].to(device)
            
            logits = model(subgraph_features, node_features, degrees[:, 0], degrees[:, 1], RA, batch_emb)
        else:
            # Skip for other models
            continue

        preds.append(torch.sigmoid(logits).view(-1).detach())
        indices_list.append(indices)
        
    if len(preds) > 0:
        all_preds = torch.cat(preds).to(device)
        
        # Filter for negatives only
        neg_indices = torch.nonzero(neg_mask.to(device), as_tuple=True)[0]
        neg_scores = all_preds[neg_indices]
        
        # Top K
        if len(neg_scores) > k:
            topk_vals, topk_idx = torch.topk(neg_scores, k)
            global_topk_idx = neg_indices[topk_idx]
            pull_targets[global_topk_idx] = topk_vals # Soft target
        else:
            pull_targets[neg_indices] = neg_scores

    return pull_targets



if __name__ == '__main__':
    # Data settings
    parser = argparse.ArgumentParser(description='Efficient Link Prediction with Hashes (ELPH)')
    parser.add_argument('--dataset_name', type=str, default='Cora',
                        choices=['Cora', 'Citeseer', 'Pubmed', 'ogbl-ppa', 'ogbl-collab', 'ogbl-ddi',
                                 'ogbl-citation2'])
    parser.add_argument('--val_pct', type=float, default=0.1,
                        help='the percentage of supervision edges to be used for validation. These edges will not appear'
                             ' in the training set and will only be used as message passing edges in the test set')
    parser.add_argument('--test_pct', type=float, default=0.2,
                        help='the percentage of supervision edges to be used for test. These edges will not appear'
                             ' in the training or validation sets for either supervision or message passing')
    parser.add_argument('--train_samples', type=float, default=inf, help='the number of training edges or % if < 1')
    parser.add_argument('--val_samples', type=float, default=inf, help='the number of val edges or % if < 1')
    parser.add_argument('--test_samples', type=float, default=inf, help='the number of test edges or % if < 1')
    parser.add_argument('--preprocessing', type=str, default=None)
    parser.add_argument('--sign_k', type=int, default=0)
    parser.add_argument('--load_features', action='store_true', help='load node features from disk')
    parser.add_argument('--load_hashes', action='store_true', help='load hashes from disk')
    parser.add_argument('--cache_subgraph_features', action='store_true',
                        help='write / read subgraph features from disk')
    parser.add_argument('--train_cache_size', type=int, default=inf, help='the number of training edges to cache')
    parser.add_argument('--year', type=int, default=0, help='filter training data from before this year')
    # GNN settings
    parser.add_argument('--model', type=str, default='BUDDY')
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_batch_size', type=int, default=1000000,
                        help='eval batch size should be largest the GPU memory can take - the same is not necessarily true at training time')
    parser.add_argument('--label_dropout', type=float, default=0.5)
    parser.add_argument('--feature_dropout', type=float, default=0.5)
    parser.add_argument('--sign_dropout', type=float, default=0.5)
    parser.add_argument('--save_model', action='store_true', help='save the model to use later for inference')
    parser.add_argument('--feature_prop', type=str, default='gcn',
                        help='how to propagate ELPH node features. Values are gcn, residual (resGCN) or cat (jumping knowledge networks)')
    # SEAL settings
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_seal_layers', type=int, default=3)
    parser.add_argument('--sortpool_k', type=float, default=0.6)
    parser.add_argument('--label_pooling', type=str, default='add', help='add or mean')
    parser.add_argument('--seal_pooling', type=str, default='edge', help='how SEAL pools features in the subgraph')
    # Subgraph settings
    parser.add_argument('--num_hops', type=int, default=1)
    parser.add_argument('--ratio_per_hop', type=float, default=1.0)
    parser.add_argument('--max_nodes_per_hop', type=int, default=None)
    parser.add_argument('--node_label', type=str, default='drnl')
    parser.add_argument('--max_dist', type=int, default=4)
    parser.add_argument('--max_z', type=int, default=1000,
                        help='the size of the label embedding table. ie. the maximum number of labels possible')
    parser.add_argument('--use_feature', type=str2bool, default=True,
                        help="whether to use raw node features as GNN input")
    parser.add_argument('--use_struct_feature', type=str2bool, default=True,
                        help="whether to use structural graph features as GNN input")
    parser.add_argument('--use_edge_weight', action='store_true',
                        help="whether to consider edge weight in GNN")
    # Training settings
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimization')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_negs', type=int, default=1, help='number of negatives for each positive')
    parser.add_argument('--train_node_embedding', action='store_true',
                        help="also train free-parameter node embeddings together with GNN")
    parser.add_argument('--propagate_embeddings', action='store_true',
                        help='propagate the node embeddings using the GCN diffusion operator')
    parser.add_argument('--loss', default='bce', type=str, help='bce, auc, hauc, or rank')
    parser.add_argument('--add_normed_features', dest='add_normed_features', type=str2bool,
                        help='Adds a set of features that are normalsied by sqrt(d_i*d_j) to calculate cosine sim')
    parser.add_argument('--use_RA', type=str2bool, default=False, help='whether to add resource allocation features')
    # SEAL specific args
    parser.add_argument('--dynamic_train', action='store_true',
                        help="dynamically extract enclosing subgraphs on the fly")
    parser.add_argument('--dynamic_val', action='store_true')
    parser.add_argument('--dynamic_test', action='store_true')
    parser.add_argument('--pretrained_node_embedding', type=str, default=None,
                        help="load pretrained node embeddings as additional node features")
    # Testing settings
    parser.add_argument('--reps', type=int, default=1, help='the number of repetition of the experiment to run')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--eval_metric', type=str, default='hits',
                        choices=('hits', 'mrr', 'auc'))
    parser.add_argument('--K', type=int, default=100, help='the hit rate @K')
    # hash settings
    parser.add_argument('--use_zero_one', type=str2bool, default=0,
                        help="whether to use the counts of (0,1) and (1,0) neighbors")
    parser.add_argument('--floor_sf', type=str2bool, default=0,
                        help='the subgraph features represent counts, so should not be negative. If --floor_sf the min is set to 0')
    parser.add_argument('--hll_p', type=int, default=8, help='the hyperloglog p parameter')
    parser.add_argument('--minhash_num_perm', type=int, default=128, help='the number of minhash perms')
    parser.add_argument('--max_hash_hops', type=int, default=2, help='the maximum number of hops to hash')
    parser.add_argument('--subgraph_feature_batch_size', type=int, default=11000000,
                        help='the number of edges to use in each batch when calculating subgraph features. '
                             'Reduce or this or increase system RAM if seeing killed messages for large graphs')
    # wandb settings
    parser.add_argument('--wandb', action='store_true', help="flag if logging to wandb")
    parser.add_argument('--wandb_offline', dest='use_wandb_offline',
                        action='store_true')  # https://docs.wandb.ai/guides/technical-faq

    parser.add_argument('--wandb_sweep', action='store_true',
                        help="flag if sweeping")  # if not it picks up params in greed_params
    parser.add_argument('--wandb_watch_grad', action='store_true', help='allows gradient tracking in train function')
    parser.add_argument('--wandb_track_grad_flow', action='store_true')

    parser.add_argument('--wandb_entity', default="link-prediction", type=str)
    parser.add_argument('--wandb_project', default="link-prediction", type=str)
    parser.add_argument('--wandb_group', default="testing", type=str, help="testing,tuning,eval")
    parser.add_argument('--wandb_run_name', default=None, type=str)
    parser.add_argument('--wandb_output_dir', default='./wandb_output',
                        help='folder to output results, images and model checkpoints')
    parser.add_argument('--wandb_log_freq', type=int, default=1, help='Frequency to log metrics.')
    parser.add_argument('--wandb_epoch_list', nargs='+', default=[0, 1, 2, 4, 8, 16],
                        help='list of epochs to log gradient flow')
    parser.add_argument('--log_features', action='store_true', help="log feature importance")
    parser.add_argument('--hf_token', type=str, default=None, help='Hugging Face token for pushing the model')
    parser.add_argument('--hf_repo_id', type=str, default=None, help='Hugging Face repo ID to push to')
    # PULL settings
    parser.add_argument('--use_pull', action='store_true', help='whether to use PULL (Positive Unlabeled Learning)')
    parser.add_argument('--pull_k', type=int, default=1000, help='number of pseudo-positives for PULL')
    parser.add_argument('--pull_interval', type=int, default=10, help='epoch interval for updating PULL targets')

    args = parser.parse_args()
    if (args.max_hash_hops == 1) and (not args.use_zero_one):
        print("WARNING: (0,1) feature knock out is not supported for 1 hop. Running with all features")
        args.use_zero_one = True
    if args.use_pull and args.pull_k <= 0:
        print("WARNING: PULL enabled but pull_k is <= 0. Disabling PULL.")
        args.use_pull = False
    if args.use_pull and args.epochs / args.pull_interval > 10:
        print("WARNING: Max 10 updates of PULL targets only. Setting new interval to fit 10 updates.")
        args.pull_interval = args.epochs // 10
    if args.dataset_name == 'ogbl-ddi':
        args.use_feature = 0  # dataset has no features
        assert args.sign_k > 0, '--sign_k must be set to > 0 i.e. 1,2 or 3 for ogbl-ddi'
    print(args)
    run(args)
