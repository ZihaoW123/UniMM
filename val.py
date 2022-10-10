import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.optim as optim
import copy

from dataloader.dataloader_visdial import VisdialDataset
import options
from models.visual_dialog_encoder import VisualDialogEncoder
from utils.visualize import VisdomVisualize
from utils.visdial_metrics import SparseGTMetrics, NDCG, scores_to_ranks
from pytorch_transformers.tokenization_bert import BertTokenizer
from utils.data_utils import sequence_mask, batch_iter
from utils.optim_utils import WarmupLinearScheduleNonZero

import pprint
from time import gmtime, strftime
from utils.data_parallel import DataParallelImbalance 
from timeit import default_timer as timer
from pytorch_transformers.optimization import AdamW
import os

import json
import logging

from train import forward


def my_topk(input, k):
    a, _ = input.topk(k=k, dim=-1)
    a_min = torch.min(a, dim=-1).values.unsqueeze(-1).expand_as(input)
    ge = torch.ge(input, a_min)
    zero = torch.zeros_like(input).to(input.device)
    result = torch.where(ge, input, zero)
    return result

def visdial_evaluate(dataloader, params, eval_batch_size, dialog_encoder_nsp_models):
    sparse_metrics = SparseGTMetrics()
    ndcg = NDCG()
    for dialog_encoder_nsp in dialog_encoder_nsp_models:
        dialog_encoder_nsp.eval()
    
    batch_idx = 0
    with torch.no_grad():
        # we can fit approximately 500 sequences of length 256 in 8 gpus with 12 GB of memory during inference.
        batch_size = 250 * (params['n_gpus'] / 1)
        batch_size = min([1, 2, 4, 5, 100, 1000, 200, 8, 10, 40, 50, 500, 20, 25, 250, 125], \
                         key=lambda x: abs(x - batch_size) if x <= batch_size else float("inf"))
        print("batch size for evaluation", batch_size)
        for epoch_id, _, batch in batch_iter(dataloader, params):
            if epoch_id == 1:
                break
            tokens = batch['tokens']
            num_rounds = tokens.shape[1]
            num_options = tokens.shape[2]
            tokens = tokens.view(-1, tokens.shape[-1])
            segments = batch['segments']
            segments = segments.view(-1, segments.shape[-1])
            positions = batch['positions']
            positions = positions.view(-1, positions.shape[-1])
            weights = batch['weights']
            weights = weights.view(-1, weights.shape[-1])
            sep_indices = batch['sep_indices']
            sep_indices = sep_indices.view(-1, sep_indices.shape[-1])
            mask = batch['mask']
            mask = mask.view(-1, mask.shape[-1])
            hist_len = batch['hist_len']
            hist_len = hist_len.view(-1)
            gt_option_inds = batch['gt_option_inds']
            gt_relevance = batch['gt_relevance']
            gt_relevance_round_id = batch['round_id'].squeeze(1)
            txt_attention_mask = batch['txt_attention_mask']
            co_attention_mask = batch['co_attention_mask']
            txt_attention_mask = txt_attention_mask.view(-1, txt_attention_mask.shape[-2], txt_attention_mask.shape[-1])
            co_attention_mask = co_attention_mask.view(-1, co_attention_mask.shape[-2], co_attention_mask.shape[-1])

            # get image features
            features = batch['image_feat']
            spatials = batch['image_loc']
            image_mask = batch['image_mask']
            max_num_regions = features.shape[-2]
            features = features.unsqueeze(1).unsqueeze(1).expand(eval_batch_size, num_rounds, num_options,
                                                                 max_num_regions, 2048).contiguous()
            spatials = spatials.unsqueeze(1).unsqueeze(1).expand(eval_batch_size, num_rounds, num_options,
                                                                 max_num_regions, 5).contiguous()
            image_mask = image_mask.unsqueeze(1).unsqueeze(1).expand(eval_batch_size, num_rounds, num_options,
                                                                     max_num_regions).contiguous()

            features = features.view(-1, max_num_regions, 2048)
            spatials = spatials.view(-1, max_num_regions, 5)
            image_mask = image_mask.view(-1, max_num_regions)

            assert tokens.shape[0] == segments.shape[0] == sep_indices.shape[0] == mask.shape[0] == \
                   hist_len.shape[0] == features.shape[0] == spatials.shape[0] == \
                   image_mask.shape[0] == num_rounds * num_options * eval_batch_size

            output = []
            
            output_nsp_lists = [ []for _ in range(len(dialog_encoder_nsp_models))]
            #output_ce = []
            assert (eval_batch_size * num_rounds * num_options) // batch_size == (
                        eval_batch_size * num_rounds * num_options) / batch_size
            for j in range((eval_batch_size * num_rounds * num_options) // batch_size):
                # create chunks of the original batch
                item = {}
                item['tokens'] = tokens[j * batch_size:(j + 1) * batch_size, :]
                item['segments'] = segments[j * batch_size:(j + 1) * batch_size, :]
                item['positions'] = positions[j * batch_size:(j + 1) * batch_size, :]
                item['weights'] = weights[j * batch_size:(j + 1) * batch_size, :]
                item['sep_indices'] = sep_indices[j * batch_size:(j + 1) * batch_size, :]
                item['mask'] = mask[j * batch_size:(j + 1) * batch_size, :]
                item['hist_len'] = hist_len[j * batch_size:(j + 1) * batch_size]
                item['txt_attention_mask'] = txt_attention_mask[j * batch_size:(j + 1) * batch_size]
                item['co_attention_mask'] = co_attention_mask[j * batch_size:(j + 1) * batch_size]

                item['image_feat'] = features[j * batch_size:(j + 1) * batch_size, :, :]
                item['image_loc'] = spatials[j * batch_size:(j + 1) * batch_size, :, :]
                item['image_mask'] = image_mask[j * batch_size:(j + 1) * batch_size, :]
                res = None
                for ids, dialog_encoder_nsp in enumerate(dialog_encoder_nsp_models):
                    _, _, _, _, nsp_scores = forward(dialog_encoder_nsp, item, params, output_nsp_scores=True, output_lm_scores=False, evaluation=True)
                    # normalize nsp scores
                    nsp_probs = F.softmax(nsp_scores, dim=1)
                    
                    #masked_lm_labels = item['mask'].to(nsp_scores.device)
                    #a, b, c = lm_scores.size()
                    # print('nsp_scores', nsp_scores.size())
                    # print('lm_scores', lm_scores.size())
                    # print('masked_lm_labels', masked_lm_labels.size())
                    # print('lm_scores', lm_scores.size())
                    # print('masked_lm_labels', masked_lm_labels.size())
                    #input = lm_scores.view(a*b, c)
                    #target = masked_lm_labels.view(-1)
                    #lm_scores = -F.cross_entropy(input=input, target=target, ignore_index=-1, reduction='none')
                    #lm_scores = lm_scores.view(a, b)
                    # print('lm_scores', lm_scores)
                    #lm_probs = lm_scores.sum(-1) / (masked_lm_labels != -1).sum(-1)
                    #lm_probs = lm_probs.exp()
                    # print('lm_probs', lm_probs)
                    # print('nsp_probs', nsp_probs[:, 0])
                 
                    assert nsp_probs.shape[-1] == 2 
                    output_nsp_lists[ids].append(nsp_probs[:, 0])
                #output_ce.append(lm_probs[:])

            #output = torch.cat(output, 0).view(eval_batch_size, num_rounds, num_options)
            res = None
            for tmp_list in output_nsp_lists:
                tmp = torch.cat(tmp_list, 0).view(eval_batch_size, num_rounds, num_options)  
                a = tmp.min(dim=-1)[0].unsqueeze(-1)
                b = tmp.max(dim=-1)[0].unsqueeze(-1)
                e_x = (tmp-a)/(b-a)
                res_tmp = e_x / (e_x.sum(dim=-1).unsqueeze(-1))
                if res is None:
                    res = res_tmp
                else:
                    res += res_tmp
            
            
            output = res
            sparse_metrics.observe(output, gt_option_inds)
            output = output[torch.arange(output.size(0)), gt_relevance_round_id - 1, :]
            ndcg.observe(output, gt_relevance)
            batch_idx += 1
            if batch_idx % 10 == 0:
                print("eval batches", batch_idx)
                cur_metrics = {}
                cur_metrics.update(sparse_metrics.retrieve(reset=False))
                cur_metrics.update(ndcg.retrieve(reset=False))
                print("eval batches: ", batch_idx,
                      "r@1", cur_metrics['r@1'],
                      "r@5", cur_metrics['r@5'],
                      "r@10", cur_metrics['r@10'],
                      "mean", cur_metrics['mean'],
                      "mrr", cur_metrics['mrr'],
                      "ndcg", cur_metrics['ndcg'])
    print("tot eval batches", batch_idx)
    all_metrics = {}
    all_metrics.update(sparse_metrics.retrieve(reset=True))
    all_metrics.update(ndcg.retrieve(reset=True))

    return all_metrics


if __name__ == '__main__':

    params = options.read_command_line()
    os.makedirs('checkpoints', exist_ok=True)
    if not os.path.exists(params['save_path']):
        os.mkdir(params['save_path'])
    viz = VisdomVisualize(
        enable=bool(params['enable_visdom']),
        env_name=params['visdom_env'],
        server=params['visdom_server'],
        port=params['visdom_server_port'])
    pprint.pprint(params)
    viz.addText(pprint.pformat(params, indent=4))

    dataset = VisdialDataset(params)

    dataset.split = 'val'


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params['device'] = device
     

    # dialog_encoder_ce = VisualDialogEncoder(params['model_config'])


    start_iter_id = 0
    dialog_encoder_nsp_pathes = [
                                "checkpoints/dense_neuralNDCG_transposed_loss1/visdial_dialog_encoder_5997.ckpt",      
                                "checkpoints/dense_neuralNDCG_transposed_loss0/visdial_dialog_encoder_3998.ckpt", 
                                "checkpoints/dense_neuralNDCG_transposed_loss0_1/visdial_dialog_encoder_3998.ckpt", 
                                "checkpoints/dense_neuralNDCG_transposed_loss0_2/visdial_dialog_encoder_3998.ckpt", 
                                "checkpoints/dense_neuralNDCG_transposed_loss0_2/visdial_dialog_encoder_5997.ckpt",  
                                ]
                                
    #dialog_encoder_nsp_pathes = [
    #                            "checkpoints/dense_ce_1.0_loss/visdial_dialog_encoder_7996.ckpt",  
    #                            "checkpoints/dense_qfocal_1.0_loss5/visdial_dialog_encoder_7996.ckpt",  
    #                            #"checkpoints/dense_neuralNDCG_transposed_loss1/visdial_dialog_encoder_3998.ckpt",  
    #                            "checkpoints/dense_listNet_loss1/visdial_dialog_encoder_9995.ckpt",  
    #                            "checkpoints/dense_neuralNDCG_loss2/visdial_dialog_encoder_5997.ckpt", 
    #                            "checkpoints/dense_neuralNDCG_transposed_loss1/visdial_dialog_encoder_5997.ckpt",   
    #                            ]

    ######dialog_encoder_nsp
    dialog_encoder_nsp_models = []
    for ids, dialog_encoder_nsp_path in enumerate(dialog_encoder_nsp_pathes):
        dialog_encoder_nsp = VisualDialogEncoder(params['model_config'])
        pretrained_dict = torch.load(dialog_encoder_nsp_path)
        #if 'model_state_dict' in pretrained_dict:
        #    pretrained_dict = pretrained_dict['model_state_dict']
      
        #model_dict = dialog_encoder_nsp.state_dict()
        #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #print("number of keys transferred", len(pretrained_dict))
        #assert len(pretrained_dict.keys()) > 0
        #model_dict.update(pretrained_dict)
        dialog_encoder_nsp.load_state_dict(pretrained_dict['model_state_dict'])
        print("dialog_encoder_nsp_path: ", dialog_encoder_nsp_path)
        del pretrained_dict
        
        
        dialog_encoder_nsp = DataParallelImbalance(dialog_encoder_nsp.cuda(0)) #nn.DataParallel(dialog_encoder_nsp.cuda(0))
        dialog_encoder_nsp.eval()
        dialog_encoder_nsp_models.append(copy.deepcopy(dialog_encoder_nsp))
 
    torch.cuda.empty_cache() 

    params['nsp_weight'] = torch.FloatTensor([[params['num_negative_samples'], 1.0]]).repeat(params['n_gpus'], 1).to(
        device)

    torch.cuda.empty_cache()
    eval_batch_size = 4
    if params['overfit']:
        eval_batch_size = 5

    dataset.split = 'val'
    # each image will need 1000 forward passes, (100 at each round x 10 rounds).
    dataloader_eval = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=params['num_workers'],
        drop_last=True,
        pin_memory=False)
    all_metrics = visdial_evaluate(dataloader_eval, params, eval_batch_size, dialog_encoder_nsp_models)
    # all_metrics = train_visdial_evaluate(dataloader_eval, params, eval_batch_size, dialog_encoder_nsp_models)
    for metric_name, metric_value in all_metrics.items():
        print(f"{metric_name}: {metric_value}")
        if 'round' in metric_name:
            viz.linePlot(start_iter_id, metric_value,
                         'Retrieval Round Val Metrics Round -' + metric_name.split('_')[-1], metric_name)
        else:
            viz.linePlot(start_iter_id, metric_value, 'Retrieval Val Metrics', metric_name)

    torch.cuda.empty_cache()
