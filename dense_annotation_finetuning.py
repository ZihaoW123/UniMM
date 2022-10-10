import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import options
from models.visual_dialog_encoder import VisualDialogEncoder
import torch.optim as optim
from utils.visualize import VisdomVisualize
import pprint
from time import gmtime, strftime
from timeit import default_timer as timer
from pytorch_transformers.optimization import AdamW
import os
from utils.visdial_metrics import SparseGTMetrics, NDCG, scores_to_ranks
from pytorch_transformers.tokenization_bert import BertTokenizer
from utils.data_utils import sequence_mask, batch_iter
from utils.optim_utils import WarmupLinearScheduleNonZero
import json
import logging
from dataloader.dataloader_dense_annotations import VisdialDatasetDense
from dataloader.dataloader_visdial import VisdialDataset
from utils.data_parallel import DataParallelImbalance
from train import forward, visdial_evaluate
from utils.rank_loss import *




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

    dataset = VisdialDatasetDense(params)

    num_images_batch = 1

    dataset.split = 'train'
    dataloader = DataLoader(
        dataset,
        batch_size= num_images_batch,
        shuffle=True,
        num_workers=params['num_workers'],
        drop_last=True,
        pin_memory=True)

    eval_dataset = VisdialDataset(params)
    eval_dataset.split = 'val'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params['device'] = device
    dialog_encoder = VisualDialogEncoder(params['model_config'])

    param_optimizer = list(dialog_encoder.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    langauge_weights = None
    with open('config/language_weights.json') as f:
        langauge_weights = json.load(f)

    optimizer_grouped_parameters = []
    for key, value in dict(dialog_encoder.named_parameters()).items():
        if value.requires_grad:
            if key in langauge_weights:
                lr = params['lr'] 
            else:
                lr = params['image_lr']

            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0}
                ]

            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.01}
                ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=params['lr'])
    scheduler = WarmupLinearScheduleNonZero(optimizer, warmup_steps=10000, t_total=200000)
    startIterID = 0

    if params['start_path']:

        pretrained_dict = torch.load(params['start_path'])

        if not params['continue']:
            if 'model_state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['model_state_dict']

            model_dict = dialog_encoder.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            print("number of keys transferred", len(pretrained_dict))
            assert len(pretrained_dict.keys()) > 0
            model_dict.update(pretrained_dict)
            dialog_encoder.load_state_dict(model_dict)
        else:
            model_dict = dialog_encoder.state_dict()
            optimizer_dict = optimizer.state_dict()
            pretrained_dict_model = pretrained_dict['model_state_dict']
            pretrained_dict_optimizer = pretrained_dict['optimizer_state_dict']
            pretrained_dict_scheduler = pretrained_dict['scheduler_state_dict']
            pretrained_dict_model = {k: v for k, v in pretrained_dict_model.items() if k in model_dict}
            pretrained_dict_optimizer = {k: v for k, v in pretrained_dict_optimizer.items() if k in optimizer_dict}
            model_dict.update(pretrained_dict_model)
            optimizer_dict.update(pretrained_dict_optimizer)
            dialog_encoder.load_state_dict(model_dict)
            optimizer.load_state_dict(optimizer_dict)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            scheduler = WarmupLinearScheduleNonZero(optimizer, warmup_steps=10000, \
                 t_total=200000, last_epoch=pretrained_dict["iterId"])
            scheduler.load_state_dict(pretrained_dict_scheduler)
            startIterID = pretrained_dict['iterId']

            del pretrained_dict, pretrained_dict_model, pretrained_dict_optimizer, pretrained_dict_scheduler, \
                model_dict, optimizer_dict
            torch.cuda.empty_cache()

    num_iter_epoch = dataset.numDataPoints['train'] // num_images_batch if not params['overfit'] else 1
    print('\n%d iter per epoch.' % num_iter_epoch)

    # dialog_encoder = nn.DataParallel(dialog_encoder)
    dialog_encoder = DataParallelImbalance(dialog_encoder.cuda(0))
    # dialog_encoder.to(device)

    start_t = timer()
    optimizer.zero_grad()
    # kl div reduces to ce if the target distribution is fixed
    ce_loss_fct = nn.KLDivLoss(reduction='batchmean')
    params['nsp_weight'] = None
    for epoch_id, idx, batch in batch_iter(dataloader, params):
        iter_id = startIterID + idx + (epoch_id * num_iter_epoch)
        dialog_encoder.train()
        # expand image features, 
        features = batch['image_feat'] 
        spatials = batch['image_loc'] 
        image_mask = batch['image_mask']
        image_label = batch['image_label']
        image_target = batch['image_target']

        num_rounds = batch["tokens"].shape[1]
        num_samples = batch["tokens"].shape[2]

        # sample 80 options including the gt option due to memory constraints
        assert num_images_batch == 1
        num_options = 100

        gt_option_ind = batch['gt_option'].item()
        all_inds_minus_gt = torch.cat([torch.arange(gt_option_ind), torch.arange(gt_option_ind + 1,100)],0)
        all_inds_minus_gt = all_inds_minus_gt[torch.randperm(99)[:num_options-1]]
        option_indices = torch.cat([batch['gt_option'].view(-1), all_inds_minus_gt], 0)

        assert num_options == len(option_indices)

        features = features.unsqueeze(1).unsqueeze(1).expand(features.shape[0], num_rounds, num_options, features.shape[1], features.shape[2])
        spatials = spatials.unsqueeze(1).unsqueeze(1).expand(spatials.shape[0], num_rounds, num_options, spatials.shape[1], spatials.shape[2])
        image_mask = image_mask.unsqueeze(1).unsqueeze(1).expand(image_mask.shape[0], num_rounds, num_options, image_mask.shape[1])
        image_label = image_label.unsqueeze(1).unsqueeze(1).expand(image_label.shape[0], num_rounds, num_options, image_label.shape[1])
        image_target = image_target.unsqueeze(1).unsqueeze(1).expand(image_target.shape[0], num_rounds, num_options, image_target.shape[1],image_target.shape[2])

        # print('features', features.size())
        # print('spatials', spatials.size())
        # print('image_mask', image_mask.size())
        # print('image_label', image_label.size())
        # print('image_target', image_target.size())

        features = features.view(-1, features.shape[-2], features.shape[-1])
        spatials = spatials.view(-1, spatials.shape[-2], spatials.shape[-1])
        image_mask = image_mask.view(-1, image_mask.shape[-1])
        image_label = image_label.view(-1, image_label.shape[-1])
        image_target = image_target.view(-1, image_target.shape[-2], image_target.shape[-1])

        # reshape text features
        tokens = batch['tokens']
        segments = batch['segments']
        positions = batch['positions']
        weights = batch['weights']
        sep_indices = batch['sep_indices'] 
        mask = batch['mask']
        hist_len = batch['hist_len']
        nsp_labels = batch['next_sentence_labels']
        txt_attention_mask = batch['txt_attention_mask']
        co_attention_mask = batch['co_attention_mask']
        # print(txt_attention_mask)
        # print('tokens', tokens.size())
        # print('segments', segments.size())
        # print('positions', positions.size())
        # print('weights', weights.size())
        # print('mask', mask.size())
        # print('nsp_labels', nsp_labels.size())
        # print('txt_attention_mask', txt_attention_mask.size())
        # print('co_attention_mask', co_attention_mask.size())

        # select 80 options from the 100 options including the GT option
        tokens = tokens[:, :, option_indices, :]
        segments = segments[:, :, option_indices, :]
        positions = positions[:, :, option_indices, :]
        weights = weights[:, :, option_indices, :]
        sep_indices = sep_indices[:, :, option_indices, :]
        mask = mask[:, :, option_indices, :]
        hist_len = hist_len[:, :, option_indices]
        nsp_labels = nsp_labels[:, :, option_indices]
        txt_attention_mask = txt_attention_mask[:, :, option_indices, ...]
        co_attention_mask = co_attention_mask[:, :, option_indices, ...]

        tokens = tokens.view(-1, tokens.shape[-1])
        segments = segments.view(-1, segments.shape[-1])
        positions = positions.view(-1, positions.shape[-1])
        weights = weights.view(-1, weights.shape[-1])
        sep_indices = sep_indices.view(-1, sep_indices.shape[-1])
        mask = mask.view(-1, mask.shape[-1])
        hist_len = hist_len.view(-1)
        nsp_labels = nsp_labels.view(-1)
        nsp_labels = nsp_labels.to(params['device'])
        txt_attention_mask = txt_attention_mask.view(-1, txt_attention_mask.shape[-2], txt_attention_mask.shape[-1])
        co_attention_mask = co_attention_mask.view(-1, co_attention_mask.shape[-2], co_attention_mask.shape[-1])

        batch['tokens'] = tokens
        batch['segments'] = segments
        batch['positions'] = positions
        batch['weights'] = weights
        batch['sep_indices'] = sep_indices
        batch['mask'] = mask
        batch['hist_len'] = hist_len
        batch['next_sentence_labels'] = nsp_labels
        batch['txt_attention_mask'] = txt_attention_mask
        batch['co_attention_mask'] = co_attention_mask

        batch['image_feat'] = features.contiguous()
        batch['image_loc'] = spatials.contiguous()
        batch['image_mask'] = image_mask.contiguous()
        batch['image_target'] = image_target.contiguous()
        batch['image_label'] = image_label.contiguous()

        # print("token shape", tokens.shape)
        loss = 0
        nsp_loss = 0
        _, lm_loss_, _, _, nsp_scores = forward(dialog_encoder, batch, params, sample_size=None,
                                                output_nsp_scores=True, evaluation=False)
        # print('loss_', loss_)
        # print('lm_loss_', lm_loss_)
        # print('nsp_loss_', nsp_loss_)
        # print('img_loss_', img_loss_)
        # print('nsp_scores', nsp_scores)

        logging.info("nsp scores: {}".format(nsp_scores))        
        # calculate dense annotation ce loss
        nsp_scores = nsp_scores.view(-1, num_options, 2)
        nsp_loss = F.cross_entropy(nsp_scores.view(-1,2), nsp_labels.view(-1))
        
        gt_relevance = batch['gt_relevance'].to(device)
        # shuffle the gt relevance scores as well
        # gt_relevance = gt_relevance * 0.95
        #if gt_relevance[:, gt_option_ind] <= 0.5:
        #    gt_relevance[:, gt_option_ind] += 0.5 
        gt_relevance = gt_relevance[:, option_indices]
        nsp_probs = F.softmax(nsp_scores, dim=-1)


        ce_loss = ce_loss_fct(F.log_softmax(nsp_probs[:, :, 0], dim=1), F.softmax(gt_relevance, dim=1))


        nsp_log_probs = F.log_softmax(nsp_scores, dim=-1)
        qfocal_loss = -((torch.abs(gt_relevance-nsp_probs[:, :, 0]) ** 2.0) * \
                       ((gt_relevance * nsp_log_probs[:, :,0]) + ((1-gt_relevance) * (nsp_log_probs[:, :, 1])))).mean()


        #preds_log = F.log_softmax(nsp_probs[:, :, 0], dim=1)
        #true_smax = F.softmax(gt_relevance, dim=1)
        #listnet_loss = torch.mean(-torch.sum(true_smax * preds_log, dim=1))
        y_pred = nsp_probs[:, :, 0]
        y_true = gt_relevance
        target_loss = neuralNDCG_transposed(y_pred, y_true)
        target_loss_name = 'neuralNDCG_transposed loss'
        lm_loss = lm_loss_.mean() 
        if torch.isnan(lm_loss):
            loss = target_loss + params['nsp_loss_coeff'] * nsp_loss
        else:
            loss = target_loss + lm_loss + params['nsp_loss_coeff'] * nsp_loss
        loss /= params['batch_multiply']
        loss.backward()        
        scheduler.step()

        if iter_id % params['batch_multiply'] == 0 and iter_id > 0:
            optimizer.step()
            optimizer.zero_grad()
        
        if iter_id % 10 == 0:
            # Update line plots
            viz.linePlot(iter_id, loss.item(), 'loss', 'tot loss')
            viz.linePlot(iter_id, lm_loss.item(), 'loss', 'lm loss')
            viz.linePlot(iter_id, nsp_loss.item(), 'loss', 'nsp loss')
            viz.linePlot(iter_id, ce_loss.item(), 'loss', 'ce loss')
            viz.linePlot(iter_id, qfocal_loss.item(), 'loss', 'qfocal_loss')
            viz.linePlot(iter_id, target_loss.item(), 'loss', target_loss_name)

            end_t = timer()

            print_format = f'[Ep: %.2f][Iter: %d][Time: %5.2fs][loss: %.3g][LM Loss: %.3g][NSP Loss: %.3g][CE Loss: %.3g][qfocal_loss: %.3g][{target_loss_name}: %.3g]'
            print_info = [
                epoch_id, iter_id, end_t - start_t, loss.item(), lm_loss.item(), nsp_loss.item(), ce_loss.item(),  qfocal_loss.item(), target_loss.item(),
            ]
            print(print_format % tuple(print_info))
            start_t = end_t
        
        old_num_iter_epoch = num_iter_epoch
        if params['overfit']:
            num_iter_epoch = 100
        if iter_id % num_iter_epoch == 0:# or iter_id % num_iter_epoch == num_iter_epoch//2:
            torch.save({'model_state_dict' : dialog_encoder.module.state_dict(),'scheduler_state_dict':scheduler.state_dict() \
                 ,'optimizer_state_dict': optimizer.state_dict(), 'iter_id':iter_id}, os.path.join(params['save_path'], 'visdial_dialog_encoder_%d.ckpt'%iter_id))

        if (iter_id % num_iter_epoch == 0) and iter_id > 0:
            viz.save()
        # fire evaluation
        # print("num iteration for eval", num_iter_epoch)
        if (iter_id % num_iter_epoch == 0) and iter_id > 0 and iter_id // num_iter_epoch >= 2:
            torch.cuda.empty_cache()
            eval_batch_size = 4
            if params['overfit']:
                eval_batch_size = 5
            
            # each image will need 1000 forward passes, (100 at each round x 10 rounds).
            dataloader = DataLoader(
                eval_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=params['num_workers'],
                drop_last=True,
                pin_memory=False)
            all_metrics = visdial_evaluate(dataloader, params, eval_batch_size, dialog_encoder)
            for metric_name, metric_value in all_metrics.items():
                print(f"{metric_name}: {metric_value}")
                if 'round' in metric_name:
                    viz.linePlot(iter_id, metric_value, 'Retrieval Round Val Metrics Round -' + metric_name.split('_')[-1], metric_name)
                else:
                    viz.linePlot(iter_id, metric_value, 'Retrieval Val Metrics', metric_name)
            torch.cuda.empty_cache()
        
        num_iter_epoch = old_num_iter_epoch