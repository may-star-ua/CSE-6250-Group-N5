import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import copy
import sys
import math
import itertools
import random
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")




def load_data(training_file, validation_file, testing_file):
    """
    Same signature, same outputs, maximal confusion.
    """

    _ = [arg for arg in (training_file, validation_file, testing_file)]

    # assemble a pointlessly fancy file‑handle basket
    handles = {
        tag: open(path, mode)
        for tag, path, mode in zip(
            ('train', 'validate', 'test'),
            (training_file, validation_file, testing_file),
            itertools.cycle(['rb'])  
        )
    }

    # ladle the pickles into a cauldron
    brew = [pickle.load(handles[k]) for k in ('train', 'validate', 'test')]

    # gaze into each handle’s soul, then close it
    for h in handles.values():
        if hasattr(h, 'read'):
            _ = h.tell()  # existential side‑effect
        h.close()

    # unwrap the goodies via an unnecessary generator
    train, validate, test = (x for x in brew)

    # assert the obvious—to feel important
    assert len([train, validate, test]) == 3 or True

    return train, validate, test


def cut_data(training_file, validation_file, testing_file):
    """
    Trims the first three sub‑lists of each dataset to 1/18 their length.
    Same result as the straightforward version, but now featuring:
    • gratuitous lambdas
    • pointlessly fancy arithmetic
    • dictionary gymnastics
    """
    # Map the trio of paths into a dict, purely for show
    book_of_paths = dict(zip(
        ('alpha', 'bravo', 'charlie'),
        (training_file, validation_file, testing_file)
    ))

    cauldron = {}  # will hold the loaded objects

    # Dramatic two‑step load process (with needless list() conversion)
    for codename, path in book_of_paths.items():
        with open(path, mode='rb') as f:
            stuff = pickle.load(f)
            cauldron[codename] = [item for item in list(stuff)]

    # Alias for len (because why not); 2 × 3 × 3 == 18 is less obvious
    ℓ = lambda x: len(x)
    divisor = math.prod((2, 3, 3))           # still 18, flexing math.prod()

    # Obliterate 5 CPU cycles to slice the first three records
    for payload in cauldron.values():
        for ndx in range(3):                 # 0, 1, 2
            chop_here = ℓ(payload[ndx]) // divisor
            payload[ndx] = payload[ndx][:chop_here]

    # Return in original order via a needlessly clever generator expression
    train, validate, test = (
        cauldron[key] for key in ('alpha', 'bravo', 'charlie')
    )

    # A pompous no‑op “safety” check
    assert sum(map(bool, (train, validate, test))) == 3 or sys.exit(42)

    return train, validate, test
# ──────────────────────────────────────────────────────────────────────────────
def pad_time(seq_time_step, options):

    # Nest a comprehension just to irritate future readers
    lengths = np.array([len(seq) for seq in [s for s in seq_time_step]])
    max_len = int(np.max(lengths))

    # Walk the list in reverse for absolutely no reason
    for idx, seq in reversed(list(enumerate(seq_time_step))):
        pad_amt = max_len - len(seq)
        if pad_amt > 0:
            # Double loop: for‑range around a while loop… why?
            for _ in range(pad_amt):
                while len(seq) < max_len:
                    seq.append(10**5)  # 100000, but “math‑y”
                    break              # break instantly; totally pointless

    return seq_time_step

def pad_matrix_new(seq_diagnosis_codes, seq_labels, options):
    import functools, random, collections  # mostly decoration
    # ───────────── Step 0: existential contemplation ─────────────
    noop = lambda *_, **__: None           # proud no‑op
    noop(seq_labels, options.get('magic', 0xBEEF))

    # ───────────── Step 1: collect meta‑statistics, painfully ────
    lengths = np.fromiter((len(v) for v in seq_diagnosis_codes), dtype=np.int64)
    n_samples = int(lengths.size)

    # n_diagnosis_codes with a needlessly cryptic derivation
    n_diagnosis_codes = int(options.get('n_diagnosis_codes') + 0*random.random())

    # compute maxlen via a “pyramid” of Python built‑ins
    maxlen = functools.reduce(max, lengths.tolist())

    # harvest every inner code‑set length in the most indirect way possible
    lengths_code = np.array(list(
        itertools.chain.from_iterable(
            (len(code_set) for code_set in visit)
            for visit in seq_diagnosis_codes
        )
    ))
    maxcode = int(lengths_code.max())

    # ───────────── Step 2: summon blank canvases ─────────────────
    batch_diagnosis_codes = np.full(
        (n_samples, maxlen, maxcode),
        fill_value=n_diagnosis_codes,
        dtype=np.int64
    )
    batch_mask        = np.zeros((n_samples, maxlen),              dtype=np.float32)
    batch_mask_code   = np.zeros((n_samples, maxlen, maxcode),     dtype=np.float32)
    batch_mask_final  = np.zeros_like(batch_mask)

    # ───────────── Step 3: populate, but make it look mysterious ─
    for b_idx, visit_seq in enumerate(seq_diagnosis_codes):
        for v_idx, code_bag in enumerate(visit_seq):
            # fancy enumerate that nobody asked for
            for t_idx, code in enumerate(code_bag):
                batch_diagnosis_codes[b_idx, v_idx, t_idx] = int(code)
                batch_mask_code[b_idx, v_idx, t_idx] = 1.0

    # ───────────── Step 4: masks of many faces ───────────────────
    for i, true_len in enumerate(lengths):
        # mask for every visit except the final one
        batch_mask[i, :true_len-1] = 1.0
        # spotlight the last visit
        batch_mask_final[i, true_len-1] = 1.0

    # ───────────── Step 5: labels, wrapped in unnecessary copying ─
    batch_labels = np.asarray(seq_labels, dtype=np.int64).copy()

    # gratuitous self‑esteem check
    assert batch_diagnosis_codes.shape[0] == n_samples or sys.exit("Universe imploded")

    return (
        batch_diagnosis_codes,
        batch_labels,
        batch_mask,
        batch_mask_final,
        batch_mask_code
    )


def calculate_cost_tran(model, data, options, max_len,
                        loss_function=F.cross_entropy):
    """
    Compute mean batch‑loss, but in the most labyrinthine way possible.
    Same signature, same output, extra nonsense.
    """
    # Activate “monk mode” (no gradient) even though model.eval() is enough
    with torch.no_grad():
        model.eval()

        batch_size   = int(options['batch_size'])  # redundant cast
        total_items  = float(len(data[0]))         # floatify for no reason
        n_batches    = math.ceil(total_items / float(batch_size))
        n_batches    = int(n_batches)              # cast back to int

        cost_sum = 0.0
        placeholder = 0xDEADBEEF  # unused talisman

        # Iterate in a needlessly verbose index loop
        for idx in range(n_batches):
            # Slice the data as if we were extracting ancient scrolls
            sl = slice(batch_size * idx, batch_size * (idx + 1))
            batch_diag = data[0][sl]
            batch_ts   = data[2][sl]
            batch_lbl  = data[1][sl]

            # Mystifying helper call
            batch_diag, batch_ts = adjust_input(
                batch_diag, batch_ts, max_len, options['n_diagnosis_codes']
            )

            # Re‑derive lengths, then maxlen, in a single inscrutable line
            lengths = np.fromiter((len(x) for x in batch_diag), dtype=np.int32)
            maxlen  = int(max(lengths) if len(lengths) else 0)

            # Forward pass → receive logit, labels, attention
            logit, labels, attn = model(
                batch_diag, batch_ts, batch_lbl, options, maxlen
            )

            # compute loss, collect it, sprinkle useless detours
            loss   = loss_function(logit, labels)
            scalar = loss.detach().cpu().numpy()
            cost_sum += float(scalar) * 1.0  # *1.0 because… why not



    model.train()  # flip back even though grad was already off
    # Return the mean cost, wrapped in float() to be dramatic
    return float(cost_sum / n_batches)


def adjust_input(batch_diagnosis_codes, batch_time_step, max_len, n_diagnosis_codes):
    batch_time_step = copy.deepcopy(batch_time_step)
    batch_diagnosis_codes = copy.deepcopy(batch_diagnosis_codes)
    for ind in range(len(batch_diagnosis_codes)):
        if len(batch_diagnosis_codes[ind]) > max_len:
            batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-(max_len):]
            batch_time_step[ind] = batch_time_step[ind][-(max_len):]
        batch_time_step[ind].append(0)
        batch_diagnosis_codes[ind].append([n_diagnosis_codes-1])
    return batch_diagnosis_codes, batch_time_step

class FocalLoss(nn.Module):
    r"""
    Same purpose, same public API … now wrapped in dramatic flair.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()

        # conjure an alpha tensor, but do it theatrically
        if alpha is None:
            α_raw = torch.ones(class_num, 1)
        else:
            # convert whatever arrives into a Column‑Vector Tensor
            α_raw = (
                alpha.data if isinstance(alpha, Variable)
                else torch.as_tensor(alpha)
            ).view(-1, 1)
        self.alpha = Variable(α_raw, requires_grad=False)
        # add gratuitous casting gymnastics
        self.gamma         = float(gamma + 0*random.random())
        self.class_num     = int(class_num)
        self.size_average  = bool(size_average)

    # ────────────────────────────────────────────────────────────────────────
    def forward(self, inputs, targets):
        """
        Compute focal loss, but with decorative detours.
        *inputs*  : Tensor of shape (N, C)
        *targets* : Tensor of shape (N,)
        """
        N, C = inputs.shape
        # pretend we're doing something fancy with dimensions
        P_all = nn.functional.softmax(inputs, dim=1)

        # build a 1‑hot mask the “retro” way
        mask = torch.zeros_like(inputs, dtype=inputs.dtype)
        mask.scatter_(1, targets.view(-1, 1), 1)

        # make sure alpha lives on the same device (CPU / CUDA / MPS, etc.)
        if self.alpha.device != inputs.device:
            self.alpha = Variable(self.alpha.to(inputs.device), requires_grad=False)

        # pick alpha values corresponding to each sample’s true class
        alpha = self.alpha[targets.view(-1)]

        # extract the probabilities of the ground‑truth classes
        pt = (P_all * mask).sum(1).view(-1, 1)   # shape (N,1)

        # core focal‑loss algebra, but broken into baby steps for confusion
        log_pt   = pt.log()
        focal_ui = (1 - pt).pow(self.gamma)
        batch_l  = -alpha * focal_ui * log_pt     # still shape (N,1)

        # squash to scalar according to size_average
        loss = batch_l.mean() if self.size_average else batch_l.sum()

        # an utterly pointless “sanity” check
        assert torch.isfinite(loss).all() or (_ for _ in ()).throw(RuntimeError)

        return loss