import torch
import torch.nn.functional as F
import numpy as np
from itertools import product
from torch.nn import BCEWithLogitsLoss
DEFAULT_EPS = 1e-8
PADDED_Y_VALUE = -1


def __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator=PADDED_Y_VALUE):
    mask = y_true == padding_indicator

    y_pred[mask] = float('-inf')
    y_true[mask] = 0.0

    _, indices = y_pred.sort(descending=True, dim=-1)
    return torch.gather(y_true, dim=1, index=indices)
def dcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=PADDED_Y_VALUE):
    """
    Discounted Cumulative Gain at k.

    Compute DCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for DCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: DCG values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    actual_length = y_true.shape[1]

    if ats is None:
        ats = [actual_length]
    ats = [min(at, actual_length) for at in ats]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    discounts = (torch.tensor(1) / torch.log2(torch.arange(true_sorted_by_preds.shape[1], dtype=torch.float) + 2.0)).to(
        device=true_sorted_by_preds.device)

    gains = gain_function(true_sorted_by_preds)

    discounted_gains = (gains * discounts)[:, :np.max(ats)]

    cum_dcg = torch.cumsum(discounted_gains, dim=1)

    ats_tensor = torch.tensor(ats, dtype=torch.long) - torch.tensor(1)

    dcg = cum_dcg[:, ats_tensor]

    return dcg
def sinkhorn_scaling(mat, mask=None, tol=1e-6, max_iter=50):
    """
    Sinkhorn scaling procedure.
    :param mat: a tensor of square matrices of shape N x M x M, where N is batch size
    :param mask: a tensor of masks of shape N x M
    :param tol: Sinkhorn scaling tolerance
    :param max_iter: maximum number of iterations of the Sinkhorn scaling
    :return: a tensor of (approximately) doubly stochastic matrices
    """
    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)
        mat = mat.masked_fill(mask[:, None, :] & mask[:, :, None], 1.0)

    for _ in range(max_iter):
        mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=DEFAULT_EPS)
        mat = mat / mat.sum(dim=2, keepdim=True).clamp(min=DEFAULT_EPS)

        if torch.max(torch.abs(mat.sum(dim=2) - 1.)) < tol and torch.max(torch.abs(mat.sum(dim=1) - 1.)) < tol:
            break

    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)

    return mat
def deterministic_neural_sort(s, tau, mask):
    """
    Deterministic neural sort.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param s: values to sort, shape [batch_size, slate_length]
    :param tau: temperature for the final softmax function
    :param mask: mask indicating padded elements
    :return: approximate permutation matrices of shape [batch_size, slate_length, slate_length]
    """
    dev = get_torch_device()

    n = s.size()[1]
    one = torch.ones((n, 1), dtype=torch.float32, device=dev)
    s = s.masked_fill(mask[:, :, None], -1e8)
    A_s = torch.abs(s - s.permute(0, 2, 1))
    A_s = A_s.masked_fill(mask[:, :, None] | mask[:, None, :], 0.0)

    B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1)))

    temp = [n - m + 1 - 2 * (torch.arange(n - m, device=dev) + 1) for m in mask.squeeze(-1).sum(dim=1)]
    temp = [t.type(torch.float32) for t in temp]
    temp = [torch.cat((t, torch.zeros(n - len(t), device=dev))) for t in temp]
    scaling = torch.stack(temp).type(torch.float32).to(dev)  # type: ignore

    s = s.masked_fill(mask[:, :, None], 0.0)
    C = torch.matmul(s, scaling.unsqueeze(-2))

    P_max = (C - B).permute(0, 2, 1)
    P_max = P_max.masked_fill(mask[:, :, None] | mask[:, None, :], -np.inf)
    P_max = P_max.masked_fill(mask[:, :, None] & mask[:, None, :], 1.0)
    sm = torch.nn.Softmax(-1)
    P_hat = sm(P_max / tau)
    return P_hat
def sample_gumbel(samples_shape, device, eps=1e-10) -> torch.Tensor:
    """
    Sampling from Gumbel distribution.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param samples_shape: shape of the output samples tensor
    :param device: device of the output samples tensor
    :param eps: epsilon for the logarithm function
    :return: Gumbel samples tensor of shape samples_shape
    """
    U = torch.rand(samples_shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)
def stochastic_neural_sort(s, n_samples, tau, mask, beta=1.0, log_scores=True, eps=1e-10):
    """
    Stochastic neural sort. Please note that memory complexity grows by factor n_samples.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param s: values to sort, shape [batch_size, slate_length]
    :param n_samples: number of samples (approximations) for each permutation matrix
    :param tau: temperature for the final softmax function
    :param mask: mask indicating padded elements
    :param beta: scale parameter for the Gumbel distribution
    :param log_scores: whether to apply the logarithm function to scores prior to Gumbel perturbation
    :param eps: epsilon for the logarithm function
    :return: approximate permutation matrices of shape [n_samples, batch_size, slate_length, slate_length]
    """
    dev = get_torch_device()

    batch_size = s.size()[0]
    n = s.size()[1]
    s_positive = s + torch.abs(s.min())
    samples = beta * sample_gumbel([n_samples, batch_size, n, 1], device=dev)
    if log_scores:
        s_positive = torch.log(s_positive + eps)

    s_perturb = (s_positive + samples).view(n_samples * batch_size, n, 1)
    mask_repeated = mask.repeat_interleave(n_samples, dim=0)

    P_hat = deterministic_neural_sort(s_perturb, tau, mask_repeated)
    P_hat = P_hat.view(n_samples, batch_size, n, n)
    return P_hat

def get_torch_device():
    """
    Getter for an available pyTorch device.
    :return: CUDA-capable GPU if available, CPU otherwise
    """
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def ndcgLoss1_scheme(G, D, *args):
    return (G / D)[:, :, None]


def ndcgLoss2_scheme(G, D, *args):
    pos_idxs = torch.arange(1, G.shape[1] + 1, device=G.device)
    delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
    deltas = torch.abs(torch.pow(torch.abs(D[0, delta_idxs - 1]), -1.) - torch.pow(torch.abs(D[0, delta_idxs]), -1.))
    deltas.diagonal().zero_()

    return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])


def lambdaRank_scheme(G, D, *args):
    return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(G[:, :, None] - G[:, None, :])


def ndcgLoss2PP_scheme(G, D, *args):
    return args[0] * ndcgLoss2_scheme(G, D) + lambdaRank_scheme(G, D)


def rankNet_scheme(G, D, *args):
    return 1.


def rankNetWeightedByGTDiff_scheme(G, D, *args):
    return torch.abs(args[1][:, :, None] - args[1][:, None, :])


def rankNetWeightedByGTDiffPowed_scheme(G, D, *args):
    return torch.abs(torch.pow(args[1][:, :, None], 2) - torch.pow(args[1][:, None, :], 2))


def listMLE(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    return torch.mean(torch.mean(observation_loss, dim=1))


def approxNDCGLoss(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, alpha=1.):
    """
    Loss based on approximate NDCG introduced in "A General Approximation Framework for Direct Optimization of
    Information Retrieval Measures". Please note that this method does not implement any kind of truncation.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param alpha: score difference weight used in the sigmoid function
    :return: loss value, a torch.Tensor
    """
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)
    padded_pairs_mask.diagonal(dim1=-2, dim2=-1).zero_()

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum((torch.pow(2, y_true_sorted) - 1) / D, dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # Here we approximate the ranking positions according to Eqs 19-20 and later approximate NDCG (Eq 21)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :])
    scores_diffs[~padded_pairs_mask] = 0.
    approx_pos = 1. + torch.sum(padded_pairs_mask.float() * (torch.sigmoid(-alpha * scores_diffs).clamp(min=eps)),
                                dim=-1)
    approx_D = torch.log2(1. + approx_pos)
    approx_NDCG = torch.sum((G / approx_D), dim=-1)

    return -torch.mean(approx_NDCG)
    # return -torch.mean(approx_NDCG)


def rankNet_weightByGTDiff(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE):
    """
    Wrapper for RankNet employing weighing by the differences of ground truth values.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    return rankNet(y_pred, y_true, padded_value_indicator, weight_by_diff=True)


def rankNet_weightByGTDiff_pow(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE):
    """
    Wrapper for RankNet employing weighing by the squared differences of ground truth values.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    return rankNet(y_pred, y_true, padded_value_indicator, weight_by_diff=False, weight_by_diff_powed=True)


def rankNet(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE, weight_by_diff=False, weight_by_diff_powed=False):
    """
    RankNet loss introduced in "Learning to Rank using Gradient Descent".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param weight_by_diff: flag indicating whether to weight the score differences by ground truth differences.
    :param weight_by_diff_powed: flag indicating whether to weight the score differences by the squared ground truth differences.
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    # here we generate every pair of indices from the range of document length in the batch
    document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, document_pairs_candidates]
    selected_pred = y_pred[:, document_pairs_candidates]

    # here we calculate the relative true relevance of every candidate pair
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

    # here we filter just the pairs that are 'positive' and did not involve a padded instance
    # we can do that since in the candidate pairs we had symetric pairs so we can stick with
    # positive ones for a simpler loss function formulation
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    pred_diffs = pred_diffs[the_mask]

    weight = None
    if weight_by_diff:
        abs_diff = torch.abs(true_diffs)
        weight = abs_diff[the_mask]
    elif weight_by_diff_powed:
        true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
        abs_diff = torch.abs(true_pow_diffs)
        weight = abs_diff[the_mask]

    # here we 'binarize' true relevancy diffs since for a pairwise loss we just need to know
    # whether one document is better than the other and not about the actual difference in
    # their relevancy levels
    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[the_mask]

    return BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)


def listNet(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    preds_smax = F.softmax(y_pred, dim=1)
    true_smax = F.softmax(y_true, dim=1)

    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)

    return torch.mean(-torch.sum(true_smax * preds_log, dim=1))


def lambdaLoss(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, weighing_scheme=None, k=None, sigma=1., mu=10.,
               reduction="mean", reduction_log="binary"):
    """
    LambdaLoss framework for LTR losses implementations, introduced in "The LambdaLoss Framework for Ranking Metric Optimization".
    Contains implementations of different weighing schemes corresponding to e.g. LambdaRank or RankNet.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param weighing_scheme: a string corresponding to a name of one of the weighing schemes
    :param k: rank at which the loss is truncated
    :param sigma: score difference weight used in the sigmoid function
    :param mu: optional weight used in NDCGLoss2++ weighing scheme
    :param reduction: losses reduction method, could be either a sum or a mean
    :param reduction_log: logarithm variant used prior to masking and loss reduction, either binary or natural
    :return: loss value, a torch.Tensor
    """
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)

    if weighing_scheme != "ndcgLoss1_scheme":
        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

    ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
    ndcg_at_k_mask[:k, :k] = 1

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # Here we apply appropriate weighing scheme - ndcgLoss1, ndcgLoss2, ndcgLoss2++ or no weights (=1.0)
    if weighing_scheme is None:
        weights = 1.
    else:
        weights = globals()[weighing_scheme](G, D, mu, true_sorted_by_preds)  # type: ignore

    # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
    scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
    weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
    if reduction_log == "natural":
        losses = torch.log(weighted_probas)
    elif reduction_log == "binary":
        losses = torch.log2(weighted_probas)
    else:
        raise ValueError("Reduction logarithm base can be either natural or binary")

    if reduction == "sum":
        loss = -torch.sum(losses[padded_pairs_mask & ndcg_at_k_mask])
    elif reduction == "mean":
        loss = -torch.mean(losses[padded_pairs_mask & ndcg_at_k_mask])
    else:
        raise ValueError("Reduction method can be either sum or mean")

    return loss

def neuralNDCG(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE, temperature=1., powered_relevancies=True, k=None,
               stochastic=False, n_samples=32, beta=0.1, log_scores=True):
    """
    NeuralNDCG loss introduced in "NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable
    Relaxation of Sorting" - https://arxiv.org/abs/2102.07831. Based on the NeuralSort algorithm.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param temperature: temperature for the NeuralSort algorithm
    :param powered_relevancies: whether to apply 2^x - 1 gain function, x otherwise
    :param k: rank at which the loss is truncated
    :param stochastic: whether to calculate the stochastic variant
    :param n_samples: how many stochastic samples are taken, used if stochastic == True
    :param beta: beta parameter for NeuralSort algorithm, used if stochastic == True
    :param log_scores: log_scores parameter for NeuralSort algorithm, used if stochastic == True
    :return: loss value, a torch.Tensor
    """
    dev = get_torch_device()

    if k is None:
        k = y_true.shape[1]

    mask = (y_true == padded_value_indicator)
    # Choose the deterministic/stochastic variant
    if stochastic:
        P_hat = stochastic_neural_sort(y_pred.unsqueeze(-1), n_samples=n_samples, tau=temperature, mask=mask,
                                       beta=beta, log_scores=log_scores)
    else:
        P_hat = deterministic_neural_sort(y_pred.unsqueeze(-1), tau=temperature, mask=mask).unsqueeze(0)

    # Perform sinkhorn scaling to obtain doubly stochastic permutation matrices
    P_hat = sinkhorn_scaling(P_hat.view(P_hat.shape[0] * P_hat.shape[1], P_hat.shape[2], P_hat.shape[3]),
                             mask.repeat_interleave(P_hat.shape[0], dim=0), tol=1e-6, max_iter=50)
    P_hat = P_hat.view(int(P_hat.shape[0] / y_pred.shape[0]), y_pred.shape[0], P_hat.shape[1], P_hat.shape[2])

    # Mask P_hat and apply to true labels, ie approximately sort them
    P_hat = P_hat.masked_fill(mask[None, :, :, None] | mask[None, :, None, :], 0.)
    y_true_masked = y_true.masked_fill(mask, 0.).unsqueeze(-1).unsqueeze(0)
    if powered_relevancies:
        y_true_masked = torch.pow(2., y_true_masked) - 1.

    ground_truth = torch.matmul(P_hat, y_true_masked).squeeze(-1)
    discounts = (torch.tensor(1.) / torch.log2(torch.arange(y_true.shape[-1], dtype=torch.float) + 2.)).to(dev)
    discounted_gains = ground_truth * discounts

    if powered_relevancies:
        idcg = dcg(y_true, y_true, ats=[k]).permute(1, 0)
    else:
        idcg = dcg(y_true, y_true, ats=[k], gain_function=lambda x: x).permute(1, 0)

    discounted_gains = discounted_gains[:, :, :k]
    ndcg = discounted_gains.sum(dim=-1) / (idcg + DEFAULT_EPS)
    idcg_mask = idcg == 0.
    ndcg = ndcg.masked_fill(idcg_mask.repeat(ndcg.shape[0], 1), 0.)

    assert (ndcg < 0.).sum() >= 0, "every ndcg should be non-negative"
    if idcg_mask.all():
        return torch.tensor(0.)

    mean_ndcg = ndcg.sum() / ((~idcg_mask).sum() * ndcg.shape[0])  # type: ignore
    return -1. * mean_ndcg  # -1 cause we want to maximize NDCG


def neuralNDCG_transposed(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE, temperature=1.,
                          powered_relevancies=True, k=None, stochastic=False, n_samples=32, beta=0.1, log_scores=True,
                          max_iter=50, tol=1e-6):
    """
    NeuralNDCG Transposed loss introduced in "NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable
    Relaxation of Sorting" - https://arxiv.org/abs/2102.07831. Based on the NeuralSort algorithm.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param temperature: temperature for the NeuralSort algorithm
    :param powered_relevancies: whether to apply 2^x - 1 gain function, x otherwise
    :param k: rank at which the loss is truncated
    :param stochastic: whether to calculate the stochastic variant
    :param n_samples: how many stochastic samples are taken, used if stochastic == True
    :param beta: beta parameter for NeuralSort algorithm, used if stochastic == True
    :param log_scores: log_scores parameter for NeuralSort algorithm, used if stochastic == True
    :param max_iter: maximum iteration count for Sinkhorn scaling
    :param tol: tolerance for Sinkhorn scaling
    :return: loss value, a torch.Tensor
    """
    dev = get_torch_device()

    if k is None:
        k = y_true.shape[1]

    mask = (y_true == padded_value_indicator)

    if stochastic:
        P_hat = stochastic_neural_sort(y_pred.unsqueeze(-1), n_samples=n_samples, tau=temperature, mask=mask,
                                       beta=beta, log_scores=log_scores)
    else:
        P_hat = deterministic_neural_sort(y_pred.unsqueeze(-1), tau=temperature, mask=mask).unsqueeze(0)

    # Perform sinkhorn scaling to obtain doubly stochastic permutation matrices
    P_hat_masked = sinkhorn_scaling(P_hat.view(P_hat.shape[0] * y_pred.shape[0], y_pred.shape[1], y_pred.shape[1]),
                                    mask.repeat_interleave(P_hat.shape[0], dim=0), tol=tol, max_iter=max_iter)
    P_hat_masked = P_hat_masked.view(P_hat.shape[0], y_pred.shape[0], y_pred.shape[1], y_pred.shape[1])
    discounts = (torch.tensor(1) / torch.log2(torch.arange(y_true.shape[-1], dtype=torch.float) + 2.)).to(dev)

    # This takes care of the @k metric truncation - if something is @>k, it is useless and gets 0.0 discount
    discounts[k:] = 0.
    discounts = discounts[None, None, :, None]

    # Here the discounts become expected discounts
    discounts = torch.matmul(P_hat_masked.permute(0, 1, 3, 2), discounts).squeeze(-1)
    if powered_relevancies:
        gains = torch.pow(2., y_true) - 1
        discounted_gains = gains.unsqueeze(0) * discounts
        idcg = dcg(y_true, y_true, ats=[k]).squeeze()
    else:
        gains = y_true
        discounted_gains = gains.unsqueeze(0) * discounts
        idcg = dcg(y_true, y_true, ats=[k]).squeeze()

    ndcg = discounted_gains.sum(dim=2) / (idcg + DEFAULT_EPS)
    idcg_mask = idcg == 0.
    ndcg = ndcg.masked_fill(idcg_mask, 0.)

    assert (ndcg < 0.).sum() >= 0, "every ndcg should be non-negative"
    if idcg_mask.all():
        return torch.tensor(0.)

    mean_ndcg = ndcg.sum() / ((~idcg_mask).sum() * ndcg.shape[0])  # type: ignore
    return -1. * mean_ndcg  # -1 cause we want to maximize NDCG

