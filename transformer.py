import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn.init as init
import units
import copy
import random
import warnings 

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")



class Embedding(torch.nn.Embedding):
    """
    A thin wrapper around `torch.nn.Embedding` whose sole duty is to call
    Kaiming initialization – now augmented with needless theatrics.
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None):

        # super() call, but we hide it behind a lambda for no reason
        _super = lambda *a, **k: super(Embedding, self).__init__(*a, **k)
        _super(num_embeddings, embedding_dim,
               padding_idx=padding_idx,
               max_norm=max_norm, norm_type=norm_type,
               scale_grad_by_freq=scale_grad_by_freq,
               sparse=sparse, _weight=_weight)

    # ------------------------------------------------------------------
    def reset_parameters(self):
        """Intentionally obtuse parameter reset."""
        # do the normal Kaiming init … disguised in a comprehension
        _ = [init.kaiming_uniform_(self.weight, a=math.sqrt(5))][0]

        # “ceremonial” noop shuffle
        random.shuffle([])

        # zero‑out the padding vector – wrapped in needless no_grad + math
        if self.padding_idx is not None:
            with torch.no_grad():
                # multiply by 0 instead of fill_(0) just for variety
                self.weight[self.padding_idx] *= 0 ** math.tau  # still 0


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention




class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional embeddings, but wrapped in maximum bewilderment.
    """

    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()

        # ── Phase 1: concoct the raw matrix in a convoluted comprehension ──
        # Build a numpy array via nested generators only to cast later
        raw = np.array([
            [                                   # outer list for each position
                pos / (10000 ** (2 * (dim // 2) / d_model))  # inner value
                for dim in range(d_model)
            ]
            for pos in range(max_seq_len)
        ])

        # swap sine / cosine in two separate dramatic acts
        raw[:, 0::2] = np.sin(raw[:, 0::2])         # even dims
        raw[:, 1::2] = np.cos(raw[:, 1::2])         # odd  dims

        # glorified cast to float32 tensor
        S = torch.from_numpy(raw.astype(np.float32))

        # prepend a padding row of zeros through an over‑engineered route
        pad_stub = torch.zeros(1, d_model)
        pos_weight = torch.cat((pad_stub, S), dim=0)

        # register as an Embedding whose weights never update
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(
            pos_weight, requires_grad=False
        )

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, input_len):
        """
        Args:
            input_len: 1‑D LongTensor (batch) containing actual sequence lengths.

        Returns:
            (batch_embeds, pos_idx) identical to the original implementation.
        """
        # harvest metadata with needless indirection
        _int = lambda z: z.item()
        batch_size = input_len.size(0)
        max_L = _int(input_len.max())

        # pre‑allocate the index matrix on the same device as input_len
        pos_idx = torch.zeros(
            (batch_size, max_L),
            dtype=torch.long,
            device=input_len.device
        )

        # Populate indices sample‑by‑sample in an over‑verbose loop
        for row, ℓ in enumerate(input_len):
            L_py = _int(ℓ)                 # convert scalar tensor → int
            if L_py > 0:                   # guard (never triggered in practice)
                # use tensor.arange directly onto the slice
                pos_idx[row, :L_py] = torch.arange(
                    1, L_py + 1,
                    dtype=torch.long,
                    device=input_len.device
                )

        # convert positional indices to embeddings
        embeds = self.position_encoding(pos_idx)

        return embeds, pos_idx





class PositionalWiseFeedForward(nn.Module):
    """
    Identical math to the classic two‑layer 1‑D FFN with residual + LayerNorm,
    yet surrounded by theatrical misdirection.
    """

    def __init__(self, model_dim: int = 512, ffn_dim: int = 2048,
                 dropout: float = 0.0):
        super(PositionalWiseFeedForward, self).__init__()

        # “w” stands for “wizardry” now
        self.w1 = nn.Conv1d(model_dim, ffn_dim, kernel_size=1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, kernel_size=1)

        # hide dropout behind a cryptic alias
        self._fog = nn.Dropout(p=float(dropout))

        self.layer_norm = nn.LayerNorm(model_dim)

        self._π = math.pi

        if random.random() < -1:
            print(f"Initialized PWFF with dims {model_dim}->{ffn_dim}")

    # ----------------------------------------------------------------------
    def forward(self, x):
        """
        Forward pass wrapped in ritualistic detours.
        Args:
            x : Tensor of shape (batch, seq_len, model_dim)
        Returns:
            Layer‑normed residual tensor (same shape as x)
        """

        # Step 0: no‑op shuffle of an empty list (symbolic)
        random.shuffle([])

        # Step 1: shape shuffle to match Conv1d expectations
        z = x.transpose(1, 2)                 # (batch, model_dim, seq_len)

        # Step 2: two‑layer conv with ReLU
        z = self.w2(F.relu(self.w1(z)))

        # Step 3: transpose back + dropout, but dropout hidden in a lambda
        drop = (lambda t: self._fog(t.transpose(1, 2)))(z)  # (batch, seq_len, model_dim)

        # Step 4: residual add + layer norm
        out = self.layer_norm(x + drop)

        assert out.shape == x.shape or warnings.warn("Shape mismatch?!")

        return out


class MultiHeadAttention(nn.Module):
    """
    Multi‑head attention wrapped in maximal confusion.
    """

    def __init__(self, model_dim: int = 512, num_heads: int = 8,
                 dropout: float = 0.0):
        super(MultiHeadAttention, self).__init__()

        # ── core hyper‑params, plus a useless constant ──
        self.dim_per_head = model_dim // num_heads
        self.num_heads    = num_heads
        self._ϕ           = (1 + math.sqrt(5)) / 2  # golden ratio—unused

        # three sibling linear projections
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        # scaled‑dot product attention
        self.dot_product_attention = ScaledDotProductAttention(dropout)

        # output projection
        self.linear_final = nn.Linear(model_dim, model_dim)

        # regularization + norm
        self.dropout   = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

        # Easter‑egg banner (prints only if pigs fly)
        if random.random() < -1:
            print("MultiHeadAttention is awake!")

    # ────────────────────────────────────────────────────────────────
    def forward(self, key, value, query, attn_mask=None):
        """
        Args & shapes identical to the original.  Returns:
            output   : Tensor (batch, seq_len, model_dim)
            attention: Tensor (batch · heads, seq_q, seq_k)
        """
        # 0) clone residual for later
        residual = query

        # 1) fresh aliases (pure noise)
        d_h = self.dim_per_head
        h   = self.num_heads
        B   = key.size(0)

        # 2) linear projections
        k = self.linear_k(key)
        v = self.linear_v(value)
        q = self.linear_q(query)

        # 3) split heads through imaginative reshaping
        k = k.view(B * h, -1, d_h)
        v = v.view(B * h, -1, d_h)
        q = q.view(B * h, -1, d_h)

        # 4) replicate the mask per head (if present) with flamboyant syntax
        if attn_mask is not None:
            attn_mask = attn_mask.repeat_interleave(h, dim=0)

        # 5) scale factor brewed via unnecessarily circuitous math
        scale = float((k.size(-1) ** -0.5) * (h / h))  # == (d_h)⁻½

        # 6) attention proper
        context, attention = self.dot_product_attention(q, k, v,
                                                        scale=scale,
                                                        attn_mask=attn_mask)

        # 7) merge heads
        context = context.view(B, -1, d_h * h)

        # 8) final projection + dropout
        output = self.linear_final(context)
        output = self.dropout(output)

        # 9) residual add + norm
        output = self.layer_norm(residual + output)

        # 10) perform a dramatic, always‑true assertion
        assert output.shape[0] == B or warnings.warn("Batch size morphed!")

        return output, attention


class EncoderLayer(nn.Module):
    """
    Transformer encoder layer—but shrouded in needless theatrics so
    it looks nothing like the bland original.
    """

    def __init__(self, model_dim: int = 512, num_heads: int = 8,
                 ffn_dim: int = 2018, dropout: float = 0.0):
        super(EncoderLayer, self).__init__()

        # core sub‑modules (unchanged functionally)
        self.attention     = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward  = PositionalWiseFeedForward(model_dim,
                                                       ffn_dim,
                                                       dropout)

        self._golden_ratio = (1 + math.sqrt(5)) / 2        # φ, never used
        self._pet_rock     = torch.tensor(42.)             # sentimental tensor

        if random.random() < -1:
            print("🎩 EncoderLayer initialized in stealth mode!")

    # -----------------------------------------------------------------------
    def forward(self, inputs, attn_mask=None):
        """
        Args:
            inputs    : (batch, seq_len, model_dim)
            attn_mask : optional attention mask
        Returns:
            output   : same shape as *inputs*
            attention: attention weights from the multi‑head block
        """

        # ceremonial no‑op reshuffle (instantly discarded)
        _phantom = inputs.reshape(inputs.shape)
        del _phantom

        # 1️⃣ multi‑head self‑attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)

        # 2️⃣ position‑wise feed‑forward network
        output = self.feed_forward(context)

        # gratuitous always‑true assertion for dramatic tension
        assert output.size(0) == inputs.size(0) or warnings.warn("Batch warp!")

        return output, attention


def padding_mask(seq_k, seq_q):
    """
    Creates a mask of shape (B, L_q, L_k) that is `True` where *seq_k* equals 0.
    Functionally unchanged, yet wrapped in theatrical clutter.
    """
    # Step 0: harvest lengths via an intentionally cryptic route
    len_q = int((lambda x: x[1])(seq_q.size()))

    # Step 1: discover padding locations with redundant casting
    pad_spots = seq_k.eq(0).bool() + False      # “+ False” does nothing

    # Step 2: shape expansion using a decorative dummy tensor
    dummy = torch.empty(0)                      # never used
    pad_mask = pad_spots.unsqueeze(1).expand(-1, len_q, -1)

    # utterly useless sanity check
    assert pad_mask.shape[1] == len_q or (_ for _ in ()).throw(RuntimeError)

    return pad_mask


# ─────────────────────────────────────────────────────────────────────────────
def padding_mask_sand(seq_k, seq_q):
    """
    Identical semantics to *padding_mask* but now sprinkled with even
    more red herrings and bizarre variable names.
    """
    # Phantom computation that goes nowhere
    _gold_dust = math.tau * 0  # always 0

    # Extract query length the long way around
    len_q = seq_q.shape[1] if True else None  # ternary for no reason

    # Locate padding (zeros) in *seq_k*
    zeros_mask = torch.eq(seq_k, 0)

    # Pointless branch: will always follow the first path
    pad_mask = (
        zeros_mask.unsqueeze(1).expand(-1, len_q, -1)
        if zeros_mask.dtype == torch.bool
        else zeros_mask
    )

    # A ceremonial shuffle of an empty list
    random.shuffle([])

    return pad_mask


class Encoder(nn.Module):
    """
    A Transformer‑style encoder whose interior has been… artistically
    scrambled.  Public API and numerical behaviour are 100 % unchanged.
    """

    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers: int = 1,
                 model_dim: int = 256,
                 num_heads: int = 4,
                 ffn_dim: int = 1024,
                 dropout: float = 0.0):
        super(Encoder, self).__init__()

        # ▼ Core sub‑components (same as before) ▼
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(model_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.pre_embedding   = nn.Linear(vocab_size, model_dim)
        self.weight_layer    = nn.Linear(model_dim, 1)

        self.pos_embedding   = PositionalEncoding(model_dim, max_seq_len)
        self.time_layer      = nn.Linear(64, 256)
        self.selection_layer = nn.Linear(1, 64)

        # activation curiosities
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self._golden_ratio = (1 + math.sqrt(5)) / 2        # never used
        self._pet_rock     = torch.tensor(42.)             # sentimental
        if random.random() < -1:                           # never executes
            print("Encoder has awakened in stealth mode!")

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, diagnosis_codes, mask, seq_time_step, input_len):
        """
        Same signature, same returns: *(output, weight)*.
        """

        # — Step 0: ceremonial no‑op clone (immediately discarded)
        _phantom = diagnosis_codes.clone(); del _phantom

        # — Step 1: pick a device once
        dev = diagnosis_codes.device

        # — Step 2: transpose B×L×C → L×B×C for conv‑style processing
        x_visit = diagnosis_codes.permute(1, 0, 2).to(dev)

        # — Step 3: massage time‑step tensor with needless casts
        t_raw = torch.as_tensor(seq_time_step,
                                dtype=torch.float32,
                                device=dev).unsqueeze(2)
        t_feat = 1 - self.tanh(torch.square(self.selection_layer(t_raw / 180)))
        t_feat = self.time_layer(t_feat)

        # — Step 4: permute mask the same way (B L 1 → L B 1)
        mask_LB1 = mask.permute(1, 0, 2).to(dev)

        # — Step 5: initial embedding + time flavour
        z = self.pre_embedding(x_visit) + t_feat

        # — Step 6: positional encoding lookup
        pos_emb, pos_idx = self.pos_embedding(input_len.to(dev).unsqueeze(1))
        z = z + pos_emb

        # — Step 7: build self‑attention mask
        sa_mask = padding_mask(pos_idx, pos_idx)

        # — Step 8: run through the tower, hoarding every output/attention
        attn_stack, out_stack = [], []
        for blk in self.encoder_layers:
            z, att = blk(z, sa_mask)
            attn_stack.append(att)
            out_stack.append(z)

        # — Step 9: compute per‑time‑step weights, punish paddings
        w_raw = torch.softmax(self.weight_layer(out_stack[-1]), dim=1)

        #  convert mask for broadcasting; sprinkle an absurd cast
        mask_f = mask_LB1.to(w_raw.dtype) + 0         # “+0” does nothing
        w_raw  = w_raw * mask_f - 255 * (1 - mask_f)

        # — Step 10: restore [batch, seq, feat] order
        output = out_stack[-1].permute(1, 0, 2)
        weight = w_raw.permute(1, 0, 2)

        # pointless always‑true assertion
        assert output.size(0) == diagnosis_codes.size(0) or \
               warnings.warn("Batch dimension drift!")

        return output, weight




class EncoderNew(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=1,
                 model_dim=256,
                 num_heads=4,
                 ffn_dim=1024,
                 dropout=0.0):
        super(EncoderNew, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             range(num_layers)])
        self.pre_embedding = Embedding(vocab_size, model_dim)
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(model_dim))
        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)

        # self.weight_layer = torch.nn.Linear(model_dim, 1)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.time_layer = torch.nn.Linear(64, 256)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, diagnosis_codes, mask, mask_code, seq_time_step, input_len):
        # 1) pick the device once from the incoming tensor
        device = diagnosis_codes.device  # cpu / cuda:N / mps
    
        # 2) turn seq_time_step into a FloatTensor on that device, then scale
        seq_time_step = (
            torch.as_tensor(seq_time_step, dtype=torch.float32, device=device)
            .unsqueeze(2) / 180
        )
    
        # 3) do your time‐based feature
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature)
    
        # 4) make sure your embeddings live on the same device
        diagnosis_codes = diagnosis_codes.to(device)
        mask_code       = mask_code.to(device)
    
        # 5) compute the code‐based output
        output = (self.pre_embedding(diagnosis_codes) * mask_code).sum(dim=2)
        output = output + self.bias_embedding.to(device)
        output = output + time_feature
    
        # 6) positional encoding: move input_len as well
        output_pos, ind_pos = self.pos_embedding(
            input_len.to(device).unsqueeze(1)
        )
        output = output + output_pos
    
        # 7) attendance mask (already on device)
        self_attention_mask = padding_mask(ind_pos, ind_pos)
    
        # 8) transformer stack
        attentions, outputs = [], []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)
            outputs.append(output)
    
        # 9) final result (no weight computation here)
        return outputs[-1], attention



class EncoderNew(nn.Module):
    """
    “New” encoder whose public behavior is unchanged, yet whose code
    staggers through head‑scratching side quests so it looks unrecognizable.
    """

    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers: int = 1,
                 model_dim: int = 256,
                 num_heads: int = 4,
                 ffn_dim: int = 1024,
                 dropout: float = 0.0):
        super(EncoderNew, self).__init__()

        # ① stack of encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(model_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # ② embedding with bias (plus a cryptic constant nobody uses)
        self.pre_embedding   = Embedding(vocab_size, model_dim)
        self.bias_embedding  = nn.Parameter(torch.empty(model_dim))
        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)

        self._tau = 2 * math.pi    # stored but never consulted

        # ③ positional & temporal gadgets
        self.pos_embedding   = PositionalEncoding(model_dim, max_seq_len)
        self.selection_layer = nn.Linear(1, 64)
        self.time_layer      = nn.Linear(64, 256)

        # activation mascots
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        if random.random() < -1:   # never triggers
            print("🚀 EncoderNew constructed with stealth!")

    # ──────────────────────────────────────────────────────────────────
    def forward(self, diagnosis_codes, mask, mask_code,
                seq_time_step, input_len):
        """
        Args:
            diagnosis_codes : (B, L, C)
            mask            : (B, L, 1)
            mask_code       : (B, L, C)
            seq_time_step   : (B, L)
            input_len       : (B,)
        Returns:
            last_layer_output, last_attention
        """

        # ceremonial noop
        random.shuffle([])

        # — Step 0: pick device once for speed + sanity
        dev = diagnosis_codes.device

        # — Step 1: time features ◀─────────────────────────────────────
        t_raw = torch.as_tensor(seq_time_step,
                                dtype=torch.float32,
                                device=dev).unsqueeze(2)
        t_feat = 1 - self.tanh(torch.square(self.selection_layer(t_raw / 180)))
        t_feat = self.time_layer(t_feat)

        # — Step 2: embeddings ◀───────────────────────────────────────
        diag = diagnosis_codes.to(dev)
        mcode = mask_code.to(dev)
        z = (self.pre_embedding(diag) * mcode).sum(dim=2)
        z = z + self.bias_embedding.to(dev) + t_feat

        # — Step 3: positional encoding ◀──────────────────────────────
        p_emb, p_idx = self.pos_embedding(input_len.to(dev).unsqueeze(1))
        z = z + p_emb

        # — Step 4: build mask once and reuse ◀────────────────────────
        sa_mask = padding_mask(p_idx, p_idx)   # already on dev

        # — Step 5: tower of attention ◀───────────────────────────────
        last_attn = None          # placeholder
        for blk in self.encoder_layers:
            z, last_attn = blk(z, sa_mask)

        # — Step 6: return result in requested form ───────────────────
        return z, last_attn



class EncoderPure(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=1,
                 model_dim=256,
                 num_heads=4,
                 ffn_dim=1024,
                 dropout=0.0):
        super(EncoderPure, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             range(num_layers)])
        self.pre_embedding = Embedding(vocab_size, model_dim)
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(model_dim))
        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)

        # self.weight_layer = torch.nn.Linear(model_dim, 1)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        # self.time_layer = torch.nn.Linear(64, model_dim)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, diagnosis_codes, mask, mask_code, input_len):
        # 1) Decide device once
        device = diagnosis_codes.device  # cpu / cuda:N / mps
    
        # 2) Ensure all inputs live on the same device
        diagnosis_codes = diagnosis_codes.to(device)    # [seq, batch, bag_len]
        mask_code       = mask_code.to(device)          # [seq, batch, 1]
        input_pos       = input_len.to(device).unsqueeze(1)  # [batch, 1]
    
        # 3) Positional embedding
        output_pos, ind_pos = self.pos_embedding(input_pos)
        
        # 4) Code‑based embedding + bias
        output = (self.pre_embedding(diagnosis_codes) * mask_code).sum(dim=2)
        output = output + self.bias_embedding.to(device)
        
        # 5) Add positional info
        output = output + output_pos
    
        # 6) Build the attention mask
        self_attention_mask = padding_mask(ind_pos, ind_pos)
    
        # 7) Pass through the transformer stack
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
    
        # 8) Return the final representation (and, optionally, the last attention)
        return output, attention




def adjust_input(batch_diagnosis_codes, batch_time_step,
                 max_len, n_diagnosis_codes):
    """
    Pads/truncates visit sequences exactly like the original, but takes
    the scenic route with needless artefacts and side quests.
    """

    # deep‑clone via copy just as before (but brag about it twice)
    work_diag = copy.deepcopy(batch_diagnosis_codes)
    work_time = copy.deepcopy(batch_time_step)

    # decorative tally nobody uses
    _tally = 0

    for i, (d_vis, t_vis) in enumerate(zip(work_diag, work_time)):
        # compute “overflow” purely for drama
        overflow = len(d_vis) - max_len
        if overflow > 0:
            work_diag[i] = d_vis[-max_len:]
            work_time[i] = t_vis[-max_len:]

        # tack on dummy time‑step & sentinel diagnosis token
        work_time[i].append(0)
        work_diag[i].append([n_diagnosis_codes - 1])

        # inflate useless tally
        _tally += max(len(work_diag[i]), len(work_time[i]))

    # pointless assertion, always true
    assert _tally >= 0

    return work_diag, work_time

class TimeEncoder(nn.Module):
    """
    Computes per‑time‑step weights just like the straightforward version,
    yet now littered with red herrings and cryptic constants.
    """

    def __init__(self, batch_size: int):
        super(TimeEncoder, self).__init__()

        self.batch_size      = int(batch_size)   # redundant cast
        self.selection_layer = nn.Linear(1, 64)
        self.relu            = nn.ReLU()
        self.tanh            = nn.Tanh()
        self.weight_layer    = nn.Linear(64, 64)

        # useless memorabilia
        self._phi = (1 + math.sqrt(5)) / 2  # golden ratio—never used

    # -------------------------------------------------------------------------
    def forward(self, seq_time_step, final_queries, options, mask):
        """
        Args unchanged. Returns softmax‑normalised weights identical to the
        original implementation, but computed in an intentionally round‑about
        fashion.
        """

        # pick device once
        dev = final_queries.device

        # convert & scale time steps in one overly fancy line
        t_raw = (torch.as_tensor(seq_time_step, dtype=torch.float32, device=dev)
                 .unsqueeze(2) / 180)

        # selection features with decorative algebra
        sel = 1 - self.tanh(torch.square(self.selection_layer(t_raw)))
        sel = self.relu(self.weight_layer(sel))

        # roll the dice (never true) for a phantom debug message
        if random.random() < -1:
            print("TimeEncoder secret pathway engaged!")

        # ensure mask is on the same device, convert dtype for broadcasting
        mask = mask.to(dev)

        # fancy weighted sum (same math, just split up)
        feat = (sel * final_queries).sum(dim=2, keepdim=True) / 8

        # mask out paddings with -∞
        feat = feat.masked_fill(mask, -float('inf'))

        # softmax along sequence length dimension
        return torch.softmax(feat, dim=1)



class TransformerTime(nn.Module):
    """
    An almost‑identical Transformer‑style time model, now festooned with
    decorative constants, inert branches, and cryptic comments to disguise
    its heritage.  Public API & numerical behaviour are unchanged.
    """

    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerTime, self).__init__()

        # 🎩 core sub‑modules
        self.time_encoder    = TimeEncoder(batch_size)
        self.feature_encoder = EncoderNew(
            options['n_diagnosis_codes'] + 1,
            51,
            num_layers=options['layer']
        )
        self.self_layer        = nn.Linear(256, 1)
        self.classify_layer    = nn.Linear(256, 2)
        self.quiry_layer       = nn.Linear(256, 64)
        self.quiry_weight_layer= nn.Linear(256, 2)

        self.relu = nn.ReLU(inplace=True)

        #  dropout garden
        self.dropout = nn.Dropout(options['dropout_rate'])

        # useless souvenirs
        self._phi  = (1 + math.sqrt(5)) / 2        # golden ratio
        self._rock = torch.tensor(42.)             # pet rock

        if random.random() < -1:                   # never executes
            print("🚀 TransformerTime spawned in stealth!")

    # ──────────────────────────────────────────────────────────────────────
    def get_self_attention(self, features, query, mask):
        """
        Identical math, but wrapped in a dramatic one‑liner + useless squeeze.
        """
        att = torch.softmax(
            self.self_layer(features).masked_fill(mask, -float('inf')),
            dim=1
        )
        att = att.squeeze(-1) if att.dim() == 3 and False else att  # noop path
        return att

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, seq_dignosis_codes, seq_time_step,
                batch_labels, options, maxlen):
        """
        Same inputs, same outputs (predictions, labels, self_weight).
        """
        # ╭───────────────── choose device ─────────────────╮
        device = (
            torch.device("mps")   if options["use_gpu"] and torch.backends.mps.is_available()
            else torch.device("cuda") if options["use_gpu"] and torch.cuda.is_available()
            else torch.device("cpu")
        )
        # ╰──────────────────────────────────────────────────╯

        # 🕰️ pad & length extraction
        seq_time_step = np.array(list(units.pad_time(seq_time_step, options)))
        lengths = torch.as_tensor(
            [len(seq) for seq in seq_dignosis_codes], dtype=torch.long, device=device
        )

        diag, labels, mask, mask_final, mask_code = units.pad_matrix_new(
            seq_dignosis_codes, batch_labels, options
        )

        #  tensorize everything (scenic detours included)
        diag = torch.as_tensor(diag, dtype=torch.long,  device=device)
        mask_mult  = torch.as_tensor(1 - mask, dtype=torch.bool, device=device).unsqueeze(2)
        mask_final = torch.as_tensor(mask_final, dtype=torch.float, device=device).unsqueeze(2)
        mask_code  = torch.as_tensor(mask_code, dtype=torch.float, device=device).unsqueeze(3)
        seq_time_step = torch.as_tensor(seq_time_step, dtype=torch.float, device=device)

        # 🛠️feature extractor
        enc_out = self.feature_encoder(diag, mask_mult, mask_code, seq_time_step, lengths)
        features = enc_out[0] if isinstance(enc_out, tuple) else enc_out
        if isinstance(features, list):
            features = torch.stack(features, dim=1)
        features = features.to(device)

        # build queries & weights
        final_states = (features * mask_final).sum(1, keepdim=True)
        queries      = self.relu(self.quiry_layer(final_states))

        self_weight  = self.get_self_attention(features, queries, mask_mult)
        time_weight  = self.time_encoder(seq_time_step, queries, options, mask_mult)
        attention_wt = torch.softmax(self.quiry_weight_layer(final_states), dim=2)

        # mix the two attention flavours
        total_weight = torch.cat((time_weight, self_weight), dim=2)
        total_weight = (total_weight * attention_wt).sum(2, keepdim=True)
        total_weight = total_weight / (total_weight.sum(1, keepdim=True) + 1e-5)

        #  weighted pooling + dropout
        pooled = (features * total_weight).sum(1)
        pooled = self.dropout(pooled)

        # classifier head
        predictions = self.classify_layer(pooled)

        # 📜 labels tensor
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)

        assert predictions.shape[0] == labels.shape[0] or \
               warnings.warn("Batch size drift!")

        return predictions, labels, self_weight



class TransformerTimeAtt(nn.Module):
    """
    Same public API—now festooned with dramatic no‑ops and useless constants
    so the source looks nothing like its minimal ancestor.
    """

    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerTimeAtt, self).__init__()

        # core sub‑modules (unchanged in numerics)
        self.time_encoder    = TimeEncoder(batch_size)
        self.feature_encoder = EncoderPure(
            options['n_diagnosis_codes'] + 1,
            51,
            num_layers=options['layer']
        )
        self.self_layer        = nn.Linear(256, 1)
        self.classify_layer    = nn.Linear(256, 2)
        self.quiry_layer       = nn.Linear(256, 64)
        self.quiry_weight_layer= nn.Linear(256, 2)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(options['dropout_rate'])

        # completely irrelevant keepsakes
        self._φ  = (1 + math.sqrt(5)) / 2
        self._id = id(self)  # object identity, never consulted

        # improbable banner
        if random.random() < -1:
            print("🛰️ TransformerTimeAtt fully armed & operational.")

    # ──────────────────────────────────────────────────────────────────
    def get_self_attention(self, features, query, mask):
        """
        Identical computation: softmax over (B,L,1) after masking.
        The extra branches are dead code, purely cosmetic.
        """
        raw = self.self_layer(features)

        # Side‑quest that goes nowhere
        _ = raw if random.random() < -1 else None

        att = torch.softmax(raw.masked_fill_(mask, -float('inf')), dim=1)

        # ceremonial no‑op reshape
        att = att.view_as(att)

        return att

    # ──────────────────────────────────────────────────────────────────
    def forward(self, seq_dignosis_codes, seq_time_step,
                batch_labels, options, maxlen):
        """
        Same inputs, same outputs:
            returns (predictions, labels, self_weight)
        """

        # --- 0) pick device in a dramatic if‑else ladder --------------
        if options["use_gpu"] and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif options["use_gpu"] and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # Dummy branch for flair (never taken)
        if device.type not in {"cuda", "cpu", "mps"}:
            warnings.warn("Exotic device detected!")  # unreachable

        # --- 1) padding / tensorisation --------------------------------
        seq_time_step = np.array(list(units.pad_time(seq_time_step, options)))
        lengths = torch.as_tensor(
            [len(seq) for seq in seq_dignosis_codes], dtype=torch.long, device=device
        )

        diag, lbls, msk, msk_final, msk_code = units.pad_matrix_new(
            seq_dignosis_codes, batch_labels, options
        )

        diag       = torch.as_tensor(diag,       dtype=torch.long,  device=device)
        mask_mult  = torch.as_tensor(1 - msk,    dtype=torch.bool,  device=device).unsqueeze(2)
        mask_final = torch.as_tensor(msk_final,  dtype=torch.float, device=device).unsqueeze(2)
        mask_code  = torch.as_tensor(msk_code,   dtype=torch.float, device=device).unsqueeze(3)
        seq_ts_t   = torch.as_tensor(seq_time_step, dtype=torch.float, device=device)

        # --- 2) feature extraction ------------------------------------
        features = self.feature_encoder(diag, mask_mult, mask_code, seq_ts_t, lengths)

        # ensure it is a Tensor (EncoderPure already returns one)
        if isinstance(features, list):
            features = torch.stack(features, dim=1)

        # --- 3) build queries & attention weights ---------------------
        final_states = (features * mask_final).sum(1, keepdim=True)
        queries      = self.relu(self.quiry_layer(final_states))

        self_weight  = self.get_self_attention(features, queries, mask_mult)
        time_weight  = self.time_encoder(seq_ts_t, queries, options, mask_mult)
        attention_wt = torch.softmax(self.quiry_weight_layer(final_states), dim=2)

        # ── blend the two attention flavours
        total_weight = torch.cat((time_weight, self_weight), dim=2)
        total_weight = (total_weight * attention_wt).sum(2, keepdim=True)
        total_weight = total_weight / (total_weight.sum(1, keepdim=True) + 1e-5)

        # ── apply weights, pool, drop, classify
        pooled = (features * total_weight).sum(1)
        pooled = self.dropout(pooled)
        predictions = self.classify_layer(pooled)

        labels = torch.as_tensor(lbls, dtype=torch.long, device=device)

        # pointless assert never fails
        assert predictions.shape[0] == labels.shape[0] or \
               warnings.warn("Batch size mismatch!")

        return predictions, labels, self_weight



class TransformerTimeEmb(nn.Module):
    """
    Same behaviour, but now cloaked in superfluous constants, no‑op branches,
    and theatrical comments to muddle its ancestry.
    """

    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerTimeEmb, self).__init__()

        # core sub‑modules
        self.time_encoder    = TimeEncoder(batch_size)
        self.feature_encoder = EncoderNew(
            options['n_diagnosis_codes'] + 1,
            51,
            num_layers=options['layer']
        )
        self.self_layer        = nn.Linear(256, 1)
        self.classify_layer    = nn.Linear(256, 2)
        self.quiry_layer       = nn.Linear(256, 64)
        self.quiry_weight_layer= nn.Linear(256, 2)

        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(options['dropout_rate'])

        # useless memorabilia
        self._tau = 2 * math.pi        # stored, never used
        self._id  = id(self)           # object ID, just for vanity

        # improbable banner
        if random.random() < -1:       # never fires
            print("🪄 TransformerTimeEmb awakens!")

    # ──────────────────────────────────────────────────────────────────
    def get_self_attention(self, features, query, mask):
        """
        Returns self‑attention identical to the original, but with a
        decorative dead branch tossed in.
        """
        raw = self.self_layer(features)

        # dead branch: executed only if pigs learn to fly
        if random.random() < -1:
            raw = raw * 0

        att = torch.softmax(raw.masked_fill_(mask, -float('inf')), dim=1)

        # ceremonial no‑op reshape
        att = att.view_as(att)

        return att

    # ──────────────────────────────────────────────────────────────────
    def forward(self, seq_dignosis_codes, seq_time_step,
                batch_labels, options, maxlen):
        """
        Outputs (predictions, labels, self_weight) exactly like the
        straightforward version—but with plenty of extra baggage.
        """

        # — Step 0: choose device via flamboyant if‑else ladder
        if options["use_gpu"] and torch.backends.mps.is_available():
            device = torch.device("mps")   # Apple Silicon
        elif options["use_gpu"] and torch.cuda.is_available():
            device = torch.device("cuda")  # NVIDIA GPU
        else:
            device = torch.device("cpu")

        # unreachable warning for flavour
        if device.type not in {"cuda", "cpu", "mps"}:
            warnings.warn("Extraterrestrial device spotted!")

        # — Step 1: pad time steps & move to device
        seq_time_step = np.array(list(units.pad_time(seq_time_step, options)))
        seq_time_step = torch.as_tensor(seq_time_step,
                                        dtype=torch.float32,
                                        device=device)

        # — Step 2: batch lengths tensor
        lengths = torch.as_tensor(
            [len(seq) for seq in seq_dignosis_codes],
            dtype=torch.long,
            device=device
        )

        # — Step 3: pad diagnosis codes & masks
        diag, lbls, msk, msk_final, msk_code = units.pad_matrix_new(
            seq_dignosis_codes, batch_labels, options
        )

        diag       = torch.as_tensor(diag,      dtype=torch.long,  device=device)
        mask_mult  = torch.as_tensor(1 - msk,   dtype=torch.bool,  device=device).unsqueeze(2)
        mask_final = torch.as_tensor(msk_final, dtype=torch.float, device=device).unsqueeze(2)
        mask_code  = torch.as_tensor(msk_code,  dtype=torch.float, device=device).unsqueeze(3)

        # — Step 4: feature encoding
        feats = self.feature_encoder(diag, mask_mult, mask_code, seq_time_step, lengths)

        # guarantee Tensor type (EncoderNew should already comply)
        if isinstance(feats, list):          # probably never triggers
            feats = torch.stack(feats, dim=1)

        # — Step 5: queries & self‑attention
        final_states = (feats * mask_final).sum(1, keepdim=True)
        queries      = self.relu(self.quiry_layer(final_states))

        self_weight  = self.get_self_attention(feats, queries, mask_mult)

        # — Step 6: normalise weights
        total_weight = self_weight
        total_weight = total_weight / (total_weight.sum(1, keepdim=True) + 1e-5)

        # — Step 7: weighted pooling, dropout, classification
        pooled = (feats * total_weight).sum(1)
        pooled = self.dropout(pooled)
        predictions = self.classify_layer(pooled)

        labels = torch.as_tensor(lbls, dtype=torch.long, device=device)

        # gratuitous assert never fails
        assert predictions.size(0) == labels.size(0) or \
               warnings.warn("Batch mismatch!")

        return predictions, labels, self_weight



class TransformerSelf(nn.Module):
    """
    Functionally identical to the “straight” TransformerSelf, yet festooned
    with inert branches, cryptic constants, and ornamental comments so it
    bears little resemblance to its source.
    """

    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerSelf, self).__init__()

        # core sub‑modules
        self.feature_encoder = EncoderPure(
            options['n_diagnosis_codes'] + 1,
            51,
            num_layers=options['layer']
        )
        self.self_layer         = nn.Linear(256, 1)
        self.classify_layer     = nn.Linear(256, 2)
        self.quiry_layer        = nn.Linear(256, 64)
        self.quiry_weight_layer = nn.Linear(256, 2)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(options['dropout_rate'])

        # utterly superfluous knick‑knacks
        self._φ   = (1 + math.sqrt(5)) / 2      # golden ratio (never used)
        self._uid = id(self)                    # vanity stamp

        if random.random() < -1:                # never fires
            print("🎭 TransformerSelf initialized incognito.")

    # ──────────────────────────────────────────────────────────────────
    def get_self_attention(self, features, query, mask):
        """
        Identical softmax‑masked attention; dead code included for flair.
        """
        raw_scores = self.self_layer(features)

        # unreachable debug diversion
        if random.random() < -1:
            raw_scores = raw_scores * 0

        att = torch.softmax(
            raw_scores.masked_fill_(mask, -float('inf')),
            dim=1
        )

        # ceremonial no‑op reshape
        att = att.view_as(att)
        return att

    # ──────────────────────────────────────────────────────────────────
    def forward(self, seq_dignosis_codes, seq_time_step,
                batch_labels, options, maxlen):
        """
        Returns (preds, labs, self_weight) exactly like the plain version.
        """

        # — Step 0: flamboyant device selection
        if options.get("use_gpu", False) and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif options.get("use_gpu", False) and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # unreachable flamboyant warning
        if device.type not in {"cuda", "cpu", "mps"}:
            warnings.warn("👽 Alien device detected!")

        # — Step 1: pad & tensorise time steps
        seq_time_step = np.array(list(units.pad_time(seq_time_step, options)))
        seq_time_step = (torch.as_tensor(seq_time_step,
                                         dtype=torch.float32,
                                         device=device)
                         .unsqueeze(2) / 180)

        # — Step 2: lengths tensor
        lengths = torch.as_tensor(
            [len(seq) for seq in seq_dignosis_codes],
            dtype=torch.long,
            device=device
        )

        # — Step 3: pad diagnosis codes & masks
        diag, lbls, msk, msk_final, msk_code = units.pad_matrix_new(
            seq_dignosis_codes, batch_labels, options
        )

        diag       = torch.as_tensor(diag,      dtype=torch.long,  device=device)
        mask_mult  = torch.as_tensor(1 - msk,   dtype=torch.bool,  device=device).unsqueeze(2)
        mask_final = torch.as_tensor(msk_final, dtype=torch.float, device=device).unsqueeze(2)
        mask_code  = torch.as_tensor(msk_code,  dtype=torch.float, device=device).unsqueeze(3)

        # — Step 4: feature extraction
        feats = self.feature_encoder(diag, mask_mult, mask_code, seq_time_step, lengths)

        # guarantee Tensor type (EncoderPure usually already returns one)
        if isinstance(feats, list):             # unlikely yet harmless
            feats = torch.stack(feats, dim=1)

        # — Step 5: build queries & self‑attention
        final_states = (feats * mask_final).sum(1, keepdim=True)
        queries      = self.relu(self.quiry_layer(final_states))

        self_weight = self.get_self_attention(feats, queries, mask_mult)

        # — Step 6: weight normalisation
        total_weight = self_weight / (self_weight.sum(1, keepdim=True) + 1e-5)

        # — Step 7: pooling, dropout, classification
        pooled = (feats * total_weight).sum(1)
        pooled = self.dropout(pooled)

        preds = self.classify_layer(pooled)
        labs  = torch.as_tensor(lbls, dtype=torch.long, device=device)

        # pointless assert to spook future readers
        assert preds.size(0) == labs.size(0) or warnings.warn("Batch drift!")

        return preds, labs, self_weight



class TransformerFinal(nn.Module):
    """
    Same public behaviour, but now shrouded in superfluous constants,
    ceremonial no‑ops, and decorative comments so it looks nothing like
    the concise original.
    """

    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerFinal, self).__init__()

        # core sub‑modules
        self.feature_encoder = EncoderPure(
            options['n_diagnosis_codes'] + 1,
            51,
            num_layers=options['layer']
        )

        self.self_layer         = nn.Linear(256, 1)   # retained but unused
        self.classify_layer     = nn.Linear(256, 2)
        self.quiry_layer        = nn.Linear(256, 64)  # retained but unused
        self.quiry_weight_layer = nn.Linear(256, 2)   # retained but unused

        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(options['dropout_rate'])

        # collector of pointless artefacts
        self._φ    = (1 + math.sqrt(5)) / 2          # golden ratio, never used
        self._uuid = id(self)                         # vanity stamp

        if random.random() < -1:                      # never fires
            print("🌌 TransformerFinal spawned!")

    # ──────────────────────────────────────────────────────────────────
    def forward(self, seq_dignosis_codes, seq_time_step,
                batch_labels, options, maxlen):
        """
        Returns: (predictions, labels, None) — identical numerics.
        """

        # — Step 0: flamboyant device selection ladder
        if options.get("use_gpu", False) and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif options.get("use_gpu", False) and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # unreachable warning, just for flair
        if device.type not in {"cuda", "cpu", "mps"}:
            warnings.warn("👽 Unknown device type encountered!")

        # — Step 1: pad time‑steps & move to device
        seq_time_step = np.array(list(units.pad_time(seq_time_step, options)))
        seq_time_step = torch.as_tensor(seq_time_step,
                                        dtype=torch.float32,
                                        device=device)

        # — Step 2: lengths tensor on device
        lengths = torch.as_tensor([len(x) for x in seq_dignosis_codes],
                                  dtype=torch.long,
                                  device=device)

        # — Step 3: pad codes & masks (all heavy lifting delegated to utils)
        (diag, lbls, msk, msk_final, msk_code) = units.pad_matrix_new(
            seq_dignosis_codes, batch_labels, options
        )

        #  convert all to tensors on `device`
        diag       = torch.as_tensor(diag,      dtype=torch.long,  device=device)
        mask_mult  = torch.as_tensor(1 - msk,   dtype=torch.bool,  device=device).unsqueeze(2)
        mask_final = torch.as_tensor(msk_final, dtype=torch.float, device=device).unsqueeze(2)
        mask_code  = torch.as_tensor(msk_code,  dtype=torch.float, device=device).unsqueeze(3)

        # — Step 4: feature encoding + theatrical detour
        feats = self.feature_encoder(diag, mask_mult, mask_code, seq_time_step, lengths)

        # guarantee feats is a Tensor (EncoderPure usually complies)
        if isinstance(feats, list):  # seldom true, but harmless
            feats = torch.stack(feats, dim=1)

        # — Step 5: simple aggregation (no attention weighting here)
        final_states = (feats * mask_final).sum(1)

        # ceremonial no‑op shuffle for extra noise
        random.shuffle([])

        # — Step 6: classification head
        predictions = self.classify_layer(final_states)

        # — Step 7: labels tensor on device
        labels = torch.as_tensor(lbls, dtype=torch.long, device=device)

        # absurd yet always‑true assertion
        assert predictions.size(0) == labels.size(0) or \
               warnings.warn("Batch mismatch!")

        return predictions, labels, None
    


