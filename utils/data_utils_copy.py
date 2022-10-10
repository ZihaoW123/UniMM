import torch
from torch.autograd import Variable
import random
import numpy as np


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def repeat_last_utterance(tokens, tokens_masked, segments, sep_indices, hist_len, attention_mask, sequence_length,
                          max_len=None):
    if max_len is None:
        max_len = tokens.size(-1)
    batch_size = tokens.size(0)
    tops = torch.topk(sep_indices, k=2, dim=-1, largest=True, sorted=False).values
    top1 = tops[:, 0].unsqueeze(-1)
    top2 = tops[:, -1].unsqueeze(-1)
    seq_range = torch.arange(0, max_len).long().to(tokens.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)

    ini_position_ids = seq_range.unsqueeze(0).expand_as(tokens)
    tmp_position_ids = (ini_position_ids >= top1.unsqueeze(-1)).long()
    position_ids = ini_position_ids - tmp_position_ids * (top1 - top2)

    att_position_ids = seq_range.unsqueeze(0).unsqueeze(0).expand_as(attention_mask)
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    attention_mask = seq_range_expand < seq_length_expand

    tokens = torch.gather(tokens, dim=-1, index=position_ids)
    segments = torch.gather(segments, dim=-1, index=position_ids)
    tokens_masked = torch.where(tmp_position_ids, tokens, tokens_masked)

    tmp = torch.zeros_like(sep_indices).to(sep_indices.device).scatter(dim=1,
                                                                       index=sep_indices.topk(1, dim=-1).indices + 1,
                                                                       value=1)
    sep_indices = tmp * (2 * top1 - top2) + sep_indices
    hist_len = hist_len + (top1 - top2).unsqueeze(-1)
    return tokens, tokens_masked, segments, sep_indices, hist_len, position_ids, attention_mask


def batch_iter(dataloader, params):
    for epochId in range(params['num_epochs']):
        for idx, batch in enumerate(dataloader):
            yield epochId, idx, batch


def list2tensorpad(inp_list, max_seq_len):
    inp_tensor = torch.LongTensor([inp_list])
    inp_tensor_zeros = torch.zeros(1, max_seq_len, dtype=torch.long)
    inp_tensor_zeros[0, :inp_tensor.shape[1]] = inp_tensor
    inp_tensor = inp_tensor_zeros
    return inp_tensor


def encode_input(utterances, start_segment, CLS, SEP, MASK, max_seq_len=256, max_sep_len=25, mask_prob=0.2):
    cur_segment = start_segment
    token_id_list = []
    segment_id_list = []
    sep_token_indices = []
    masked_token_list = []

    token_id_list.append(CLS)
    segment_id_list.append(cur_segment)
    masked_token_list.append(0)

    cur_sep_token_index = 0
    cur_id = 1
    utt_len = len(utterances)
    for cur_utterance in utterances:
        # add the masked token and keep track
        cur_masked_index = [1 if (random.random() < mask_prob and cur_id != utt_len) else 0 for _ in
                            range(len(cur_utterance))]
        masked_token_list.extend(cur_masked_index)
        token_id_list.extend(cur_utterance)
        segment_id_list.extend([cur_segment] * len(cur_utterance))

        token_id_list.append(SEP)
        segment_id_list.append(cur_segment)
        masked_token_list.append(0)
        cur_sep_token_index = cur_sep_token_index + len(cur_utterance) + 1
        sep_token_indices.append(cur_sep_token_index)
        cur_segment = cur_segment ^ 1  # cur segment osciallates between 0 and 1
        cur_id += 1

    assert len(segment_id_list) == len(token_id_list) == len(masked_token_list) == sep_token_indices[-1] + 1
    # convert to tensors and pad to maximum seq length
    tokens = list2tensorpad(token_id_list, max_seq_len)
    masked_tokens = list2tensorpad(masked_token_list, max_seq_len)
    masked_tokens[0, masked_tokens[0, :] == 0] = -1
    mask = masked_tokens[0, :] == 1
    masked_tokens[0, mask] = tokens[0, mask]
    tokens[0, mask] = MASK

    # print("mask", mask)
    # print("tokens", tokens)
    # print("masked tokens", masked_tokens)
    # print("num mask tokens", torch.sum(mask))

    segment_id_list = list2tensorpad(segment_id_list, max_seq_len)
    # segment_id_list += 2 
    return tokens, segment_id_list, list2tensorpad(sep_token_indices, max_sep_len), masked_tokens


def encode_image_input(features, num_boxes, boxes, image_target, max_regions=37, mask_prob=0.15):
    output_label = []
    num_boxes = min(int(num_boxes), max_regions)

    mix_boxes_pad = np.zeros((max_regions, boxes.shape[-1]))
    mix_features_pad = np.zeros((max_regions, features.shape[-1]))
    mix_image_target = np.zeros((max_regions, image_target.shape[-1]))

    mix_boxes_pad[:num_boxes] = boxes[:num_boxes]
    mix_features_pad[:num_boxes] = features[:num_boxes]
    mix_image_target[:num_boxes] = image_target[:num_boxes]

    boxes = mix_boxes_pad
    features = mix_features_pad
    image_target = mix_image_target

    for i in range(num_boxes):
        prob = random.random()
        # mask token with 15% probability
        if prob < mask_prob:
            prob /= mask_prob

            # 80% randomly change token to mask token
            if prob < 0.9:
                features[i] = 0
            output_label.append(1)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    image_mask = [1] * (int(num_boxes))
    while len(image_mask) < max_regions:
        image_mask.append(0)
        output_label.append(-1)

    # ensure we have atleast one region being predicted
    output_label[random.randint(1, len(output_label) - 1)] = 1
    image_label = torch.LongTensor(output_label)
    image_label[0] = 0  # make sure the <IMG> token doesn't contribute to the masked loss
    image_mask = torch.tensor(image_mask).float()

    features = torch.tensor(features).float()
    spatials = torch.tensor(boxes).float()
    image_target = torch.tensor(image_target).float()
    return features, spatials, image_mask, image_target, image_label
