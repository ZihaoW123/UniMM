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
        seq_range_expand = seq_range_expand.to(sequence_length.device)
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


def encode_input_gen_test(utterances, start_segment, CLS, SEP, MASK, max_seq_len=256, max_sep_len=25):
    cur_segment = start_segment
    token_id_list = []
    segment_id_list = []
    position_id_list = []
    sep_token_indices = []
    seq_ids = torch.arange(max_seq_len)

    txt_attention_mask = seq_ids[None, :].repeat(max_seq_len, 1) == seq_ids[:, None]
    co_txt_attention_mask = torch.zeros(max_seq_len, dtype=torch.long)
    # print('encode_input /*********')
    # print('txt_attention_mask', txt_attention_mask.size())
    # print('co_txt_attention_mask', co_txt_attention_mask.size())
    # print('encode_input *********/')

    token_id_list.append(CLS)
    segment_id_list.append(cur_segment)
    position_id_list.append(0)

    cur_sep_token_index = 0
    cur_id = 1
    context_list = utterances[:-1]
    target_list = utterances[-1]
    utt_len = len(context_list)
    for cur_utterance in context_list:
        # add the masked token and keep track
        cur_len = len(cur_utterance)
        token_id_list.extend(cur_utterance)
        segment_id_list.extend([cur_segment] * cur_len)

        token_id_list.append(SEP)
        segment_id_list.append(cur_segment)

        cur_position = range(len(position_id_list), len(position_id_list) + cur_len + 1)
        position_id_list.extend(cur_position)

        cur_sep_token_index = cur_sep_token_index + cur_len + 1
        sep_token_indices.append(cur_sep_token_index)
        if cur_id == utt_len:
            orig_length = len(token_id_list)
            txt_attention_mask[0, :orig_length] = 1
            txt_attention_mask[1:orig_length, 1:orig_length] = 1
            co_txt_attention_mask[1:orig_length] = 1


        cur_segment = cur_segment ^ 1  # cur segment osciallates between 0 and 1
        cur_id += 1

    assert len(segment_id_list) == len(token_id_list) == sep_token_indices[-1] + 1
    # convert to tensors and pad to maximum seq length
    length = len(token_id_list)
    if length > max_seq_len:
        token_id_list = token_id_list[:max_seq_len]
        segment_id_list = segment_id_list[:max_seq_len]
        position_id_list = position_id_list[:max_seq_len]
        sep_token_indices[-1] = max_seq_len - 1

    context = list2tensorpad(token_id_list, max_seq_len)
    target = list2tensorpad(target_list, max_seq_len)
    # print("mask", mask)
    # print("tokens", tokens)
    # print("masked tokens", masked_tokens)
    # print("num mask tokens", torch.sum(mask))

    segment_id_list = list2tensorpad(segment_id_list, max_seq_len)
    position_id_list = list2tensorpad(position_id_list, max_seq_len)
    sep_token_indices = list2tensorpad(sep_token_indices, max_sep_len)
    txt_attention_mask = txt_attention_mask.unsqueeze(0)
    co_txt_attention_mask = co_txt_attention_mask.unsqueeze(0)
    # print('tokens_size', tokens.size())
    return context, target, segment_id_list, position_id_list, sep_token_indices, txt_attention_mask, co_txt_attention_mask


def encode_input_gen(utterances, start_segment, CLS, SEP, MASK, max_seq_len=256, max_sep_len=25, mask_prob=0.1, is_negtive=0, weight=1, vocab_size=None):
    cur_segment = start_segment
    token_id_list = []
    segment_id_list = []
    position_id_list = []
    sep_token_indices = []
    masked_token_list = []
    token_weight_list = []
    seq_ids = torch.arange(max_seq_len)
    
    causal_mask = seq_ids[None, :].repeat(max_seq_len, 1) < seq_ids[:, None]
    causal_mask2 = seq_ids[None, :].repeat(max_seq_len, 1) <= seq_ids[:, None]
    txt_attention_mask = seq_ids[None, :].repeat(max_seq_len, 1) == seq_ids[:, None]
    co_txt_attention_mask = torch.zeros(max_seq_len, dtype=torch.long)
    # print('encode_input /*********')
    # print('txt_attention_mask', txt_attention_mask.size())
    # print('co_txt_attention_mask', co_txt_attention_mask.size())
    # print('encode_input *********/')

    token_id_list.append(CLS)
    segment_id_list.append(cur_segment)
    position_id_list.append(0)
    masked_token_list.append(0)
    token_weight_list.append(0)

    cur_sep_token_index = 0
    cur_id = 1
    utt_len = len(utterances)
    orig_length = 0
    for cur_utterance in utterances:
        # add the masked token and keep track
        cur_len = len(cur_utterance)
        # cur_masked_index = [1 if (np.random.rand() < mask_prob and cur_id != utt_len) else 0 for _ in
        #                     range(cur_len)]

        if cur_id == utt_len and cur_len <= 1:
            cur_masked_index = [0 for _ in range(cur_len)]
        else:
            cur_masked_index = [1 if (np.random.rand() < mask_prob) else 0 for _ in range(cur_len)]

        masked_token_list.extend(cur_masked_index)
        token_id_list.extend(cur_utterance)
        segment_id_list.extend([cur_segment] * cur_len)
        # token_weight_list.extend(cur_masked_index)
        if cur_id == utt_len and is_negtive:
            token_weight_list.extend([0 for _ in range(cur_len)])
        else:
            token_weight_list.extend(cur_masked_index)


        token_id_list.append(SEP)
        segment_id_list.append(cur_segment)
        masked_token_list.append(0)
        token_weight_list.append(0)

        cur_position = range(len(position_id_list), len(position_id_list) + cur_len + 1)
        position_id_list.extend(cur_position)

        cur_sep_token_index = cur_sep_token_index + cur_len + 1
        sep_token_indices.append(cur_sep_token_index)
        if cur_id == utt_len:
            last_len = cur_len + 1
            orig_length = len(token_id_list)
            txt_attention_mask[0, :orig_length+last_len] = 1
            txt_attention_mask[1:orig_length-last_len, 1:orig_length-last_len] = 1
            txt_attention_mask[orig_length-last_len:orig_length, 1:orig_length] = causal_mask2[orig_length-last_len:orig_length, 1:orig_length] 
            if orig_length+last_len <= max_seq_len:
                txt_attention_mask[orig_length:orig_length+last_len, 1:orig_length] = causal_mask[orig_length-last_len:orig_length, 1:orig_length]
                txt_attention_mask[orig_length+last_len:, :] = 0
            else:
                txt_attention_mask[orig_length:max_seq_len, 1:orig_length] = causal_mask[orig_length-last_len: max_seq_len-last_len, 1:orig_length]
            co_txt_attention_mask[1:orig_length-last_len] = 1

            cur_masked_index = [1] * len(cur_utterance)
            masked_token_list.extend(cur_masked_index)
            token_id_list.extend(cur_utterance)
            segment_id_list.extend([cur_segment] * len(cur_utterance))

            token_id_list.append(SEP)
            segment_id_list.append(cur_segment)
            masked_token_list.append(1)

            if is_negtive:
                token_weight_list.extend([-weight] * (len(cur_utterance)+1))
                #token_weight_list.extend([0] * (len(cur_utterance)+1))
            else:
                token_weight_list.extend([weight] * (len(cur_utterance)+1))

            position_id_list.extend(cur_position)
            cur_sep_token_index = cur_sep_token_index + len(cur_utterance) + 1
            sep_token_indices.append(cur_sep_token_index)

        cur_segment = cur_segment ^ 1  # cur segment osciallates between 0 and 1
        cur_id += 1


    assert len(segment_id_list) == len(token_id_list) == len(masked_token_list) == sep_token_indices[-1] + 1
    # convert to tensors and pad to maximum seq length
    length = len(token_id_list)
    if length > max_seq_len:
        token_id_list = token_id_list[:max_seq_len]
        segment_id_list = segment_id_list[:max_seq_len]
        position_id_list = position_id_list[:max_seq_len]
        masked_token_list = masked_token_list[:max_seq_len]
        token_weight_list = token_weight_list[:max_seq_len]
        sep_token_indices[-1] = max_seq_len-1
    tokens = list2tensorpad(token_id_list, max_seq_len)
    masked_tokens = list2tensorpad(masked_token_list, max_seq_len)
    masked_tokens[0, masked_tokens[0, :] == 0] = -1
    mask = masked_tokens[0, :] == 1
    masked_tokens[0, mask] = tokens[0, mask]
    tokens[0, mask] = MASK
    
    for pos, tmp in enumerate(mask):
        if tmp == 1:
            if np.random.rand() < 0.8 or vocab_size is None or pos >= orig_length:  # 80%
                tokens[0, pos] = MASK
            elif (np.random.rand()) < 0.5:  # 10%
                tokens[0, pos] = np.random.randint(0, vocab_size)


    # print("mask", mask)
    # print("tokens", tokens)
    # print("masked tokens", masked_tokens)
    # print("num mask tokens", torch.sum(mask))

    segment_id_list = list2tensorpad(segment_id_list, max_seq_len)
    position_id_list = list2tensorpad(position_id_list, max_seq_len)
    sep_token_indices = list2tensorpad(sep_token_indices, max_sep_len)
    token_weight_list = list2tensorpad(token_weight_list, max_seq_len)
    txt_attention_mask = txt_attention_mask.unsqueeze(0)
    co_txt_attention_mask = co_txt_attention_mask.unsqueeze(0)
    # print('tokens_size', tokens.size())
    # print('segment_id_list_size', segment_id_list.size())
    # print('position_id_list_size', position_id_list.size())
    # print('sep_token_indices_size', sep_token_indices.size())
    # print('masked_tokens_size', masked_tokens.size())
    # print('token_weight_list_size', token_weight_list.size())
    # print('txt_attention_mask_size', txt_attention_mask.size())
    # print('co_txt_attention_mask_size', co_txt_attention_mask.size())
    # print('tokens', tokens)
    # print('segment_id_list', segment_id_list)
    # print('position_id_list', position_id_list)
    # print('sep_token_indices', sep_token_indices)
    # print('masked_tokens', masked_tokens)
    # print('token_weight_list', token_weight_list)
    # print('co_txt_attention_mask', co_txt_attention_mask)
    # for i in range(length+2):
    #     print('txt_attention_mask', i, txt_attention_mask[0, i, :length+10])
    return tokens, segment_id_list, position_id_list, sep_token_indices, masked_tokens, token_weight_list, txt_attention_mask, co_txt_attention_mask


def encode_input_dis(utterances, start_segment, CLS, SEP, MASK, max_seq_len=256, max_sep_len=25, mask_prob=0.1, is_negtive=0, weight=1, vocab_size=None):
    cur_segment = start_segment
    token_id_list = []
    segment_id_list = []
    position_id_list = []
    sep_token_indices = []
    masked_token_list = []
    token_weight_list = []

    txt_attention_mask = torch.zeros((max_seq_len, max_seq_len), dtype=torch.long)
    co_txt_attention_mask = torch.zeros(max_seq_len, dtype=torch.long)
    # print('encode_input /*********')
    # print('txt_attention_mask', txt_attention_mask.size())
    # print('co_txt_attention_mask', co_txt_attention_mask.size())
    # print('encode_input *********/')

    token_id_list.append(CLS)
    segment_id_list.append(cur_segment)
    position_id_list.append(0)
    masked_token_list.append(0)
    token_weight_list.append(0)

    cur_sep_token_index = 0
    cur_id = 1
    utt_len = len(utterances)
    orig_length = 0
    for cur_utterance in utterances:
        # add the masked token and keep track
        cur_len = len(cur_utterance)
        # cur_masked_index = [1 if (np.random.rand() < mask_prob and cur_id != utt_len) else 0 for _ in
        #                     range(cur_len)]

        if cur_id == utt_len and cur_len <= 1:
            cur_masked_index = [0 for _ in range(cur_len)]
        else:
            cur_masked_index = [1 if (np.random.rand() < mask_prob) else 0 for _ in range(cur_len)]

        masked_token_list.extend(cur_masked_index)
        token_id_list.extend(cur_utterance)
        segment_id_list.extend([cur_segment] * cur_len)
        # token_weight_list.extend(cur_masked_index)
        if cur_id == utt_len and is_negtive:
            token_weight_list.extend([0 for _ in range(cur_len)])
        else:
            token_weight_list.extend(cur_masked_index)

        token_id_list.append(SEP)
        segment_id_list.append(cur_segment)
        masked_token_list.append(0)
        token_weight_list.append(0)

        cur_position = range(len(position_id_list), len(position_id_list) + cur_len + 1)
        position_id_list.extend(cur_position)

        cur_sep_token_index = cur_sep_token_index + cur_len + 1
        sep_token_indices.append(cur_sep_token_index)
        if cur_id == utt_len:
            last_len = cur_len + 1
            orig_length = len(token_id_list)
            #txt_attention_mask[:orig_length+last_len, :orig_length+last_len] = 1
            #txt_attention_mask[orig_length:orig_length + last_len, orig_length - last_len:orig_length] = 0 
            #co_txt_attention_mask[:orig_length+last_len] = 1
            co_txt_attention_mask[:orig_length] = 1
            txt_attention_mask[:orig_length, :orig_length] = 1

            #cur_masked_index = [1] * len(cur_utterance)
            #masked_token_list.extend(cur_masked_index)
            #token_id_list.extend(cur_utterance)
            #segment_id_list.extend([cur_segment] * len(cur_utterance))

            #token_id_list.append(SEP)
            #segment_id_list.append(cur_segment)
            #masked_token_list.append(1)

            #if is_negtive:
            #    token_weight_list.extend([-weight] * (len(cur_utterance) + 1))
            #else:
            #    token_weight_list.extend([weight] * (len(cur_utterance) + 1))

            #position_id_list.extend(cur_position)
            #cur_sep_token_index = cur_sep_token_index + len(cur_utterance) + 1
            #sep_token_indices.append(cur_sep_token_index)

        cur_segment = cur_segment ^ 1  # cur segment osciallates between 0 and 1
        cur_id += 1

    assert len(segment_id_list) == len(token_id_list) == len(masked_token_list) == sep_token_indices[-1] + 1
    # convert to tensors and pad to maximum seq length
    length = len(token_id_list)
    if length > max_seq_len:
        token_id_list = token_id_list[:max_seq_len]
        segment_id_list = segment_id_list[:max_seq_len]
        position_id_list = position_id_list[:max_seq_len]
        masked_token_list = masked_token_list[:max_seq_len]
        token_weight_list = token_weight_list[:max_seq_len]
        sep_token_indices[-1] = max_seq_len - 1
    tokens = list2tensorpad(token_id_list, max_seq_len)
    masked_tokens = list2tensorpad(masked_token_list, max_seq_len)
    masked_tokens[0, masked_tokens[0, :] == 0] = -1
    mask = masked_tokens[0, :] == 1
    masked_tokens[0, mask] = tokens[0, mask]
    tokens[0, mask] = MASK
    for pos, tmp in enumerate(mask):
        if tmp == 1:
            if np.random.rand() < 0.8 or vocab_size is None or pos >= orig_length:  # 80%
                tokens[0, pos] = MASK
            elif (np.random.rand()) < 0.5:  # 10%
                tokens[0, pos] = np.random.randint(0, vocab_size)

    # print("mask", mask)
    # print("tokens", tokens)
    # print("masked tokens", masked_tokens)
    # print("num mask tokens", torch.sum(mask))

    segment_id_list = list2tensorpad(segment_id_list, max_seq_len)
    position_id_list = list2tensorpad(position_id_list, max_seq_len)
    sep_token_indices = list2tensorpad(sep_token_indices, max_sep_len)
    token_weight_list = list2tensorpad(token_weight_list, max_seq_len)
    txt_attention_mask = txt_attention_mask.unsqueeze(0)
    co_txt_attention_mask = co_txt_attention_mask.unsqueeze(0)
    # print('tokens_size', tokens.size())
    # print('segment_id_list_size', segment_id_list.size())
    # print('position_id_list_size', position_id_list.size())
    # print('sep_token_indices_size', sep_token_indices.size())
    # print('masked_tokens_size', masked_tokens.size())
    # print('token_weight_list_size', token_weight_list.size())
    # print('txt_attention_mask_size', txt_attention_mask.size())
    # print('co_txt_attention_mask_size', co_txt_attention_mask.size())
    # print('tokens', tokens)
    # print('segment_id_list', segment_id_list)
    # print('position_id_list', position_id_list)
    # print('sep_token_indices', sep_token_indices)
    # print('masked_tokens', masked_tokens)
    # print('token_weight_list', token_weight_list)
    # print('co_txt_attention_mask', co_txt_attention_mask)
    # for i in range(length+2):
    #     print('txt_attention_mask', i, txt_attention_mask[0, i, :length+10])
    return tokens, segment_id_list, position_id_list, sep_token_indices, masked_tokens, token_weight_list, txt_attention_mask, co_txt_attention_mask

def encode_input(dis_rate, utterances, start_segment, CLS, SEP, MASK, max_seq_len=256, max_sep_len=25, mask_prob=0.15, is_negtive=0, weight=1, vocab_size=None):
    if np.random.rand() < dis_rate:
        return encode_input_dis(utterances, start_segment, CLS, SEP, MASK, max_seq_len=max_seq_len, max_sep_len=max_sep_len, mask_prob=mask_prob,
                         is_negtive=is_negtive, weight=weight, vocab_size=vocab_size)
    else:
        return encode_input_gen(utterances, start_segment, CLS, SEP, MASK, max_seq_len=max_seq_len, max_sep_len=max_sep_len, mask_prob=mask_prob,
                         is_negtive=is_negtive, weight=weight, vocab_size=vocab_size)

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
