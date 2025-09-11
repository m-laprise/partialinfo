import torch


def _memory_attention(self, h: torch.Tensor) -> torch.Tensor:
    """Fused q/k/v for causal memory attention.
    Input/Output shape: [B, T, A, H]
    """
    B, T, A, H = h.shape
    # Prepare stacked weights: [A, 3, H, H]
    if self.sharedV:
        # W_v is shared linear weight [H, H], expand to per-agent
        W_v_exp = self.W_v_mem_shared.weight.unsqueeze(0).expand(A, -1, -1)
        W_stack = torch.stack([self.W_q_mem, self.W_k_mem, W_v_exp], dim=1)
    else:
        W_stack = torch.stack([self.W_q_mem, self.W_k_mem, self.W_v_mem], dim=1)

    # Single fused matmul -> qkv: [B, T, A, 3, H]
    # einsum string: 'btah,aqhd->btaqd'  (q indexes Q=3)
    qkv = torch.einsum('btah,aqhd->btaqd', h, W_stack)

    # Split q,k,v -> each [B, T, A, H]
    q = qkv[..., 0, :]
    k = qkv[..., 1, :]
    v = qkv[..., 2, :]

    # Reshape to attention input: [B*A, heads, T, Dh]
    def to_mem_attn_input(x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, A, H] -> [B, T, A, heads, Dh]
        x = x.reshape(B, T, A, self.heads, self.head_dim)
        # permute to [B, A, heads, T, Dh] then merge BA -> [B*A, heads, T, Dh]
        x = x.permute(0, 2, 3, 1, 4).reshape(B * A, self.heads, T, self.head_dim)
        return x

    q_att = to_mem_attn_input(q)
    k_att = to_mem_attn_input(k)
    v_att = to_mem_attn_input(v)

    dropout_p = self.dropout if self.training else 0.0

    # scaled_dot_product_attention expects q,k,v in [B*, heads, L, Dh] shape
    if q_att.is_cuda:
        backends_ok = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        with sdpa_kernel(backends_ok, set_priority=True):
            out = F.scaled_dot_product_attention(q_att, k_att, v_att, dropout_p=dropout_p, is_causal=True)
    else:
        out = F.scaled_dot_product_attention(q_att, k_att, v_att, dropout_p=dropout_p, is_causal=True)

    # out: [B*A, heads, T, Dh] -> un-flatten to [B, T, A, H]
    out = out.reshape(B, A, self.heads, T, self.head_dim).permute(0, 3, 1, 2, 4).reshape(B, T, A, H)
    return out


def _social_attention(self, h: torch.Tensor) -> torch.Tensor:
    """Fused q/k/v for social attention across agents.
    Input/Output shape: [B, T, A, H]
    """
    B, T, A, H = h.shape

    # Prepare stacked weights: [A, 3, H, H]
    if self.sharedV:
        W_v_exp = self.W_v_soc_shared.weight.unsqueeze(0).expand(A, -1, -1)
        W_stack = torch.stack([self.W_q_soc, self.W_k_soc, W_v_exp], dim=1)
    else:
        W_stack = torch.stack([self.W_q_soc, self.W_k_soc, self.W_v_soc], dim=1)

    # Single fused matmul -> qkv: [B, T, A, 3, H]
    qkv = torch.einsum('btah,aqhd->btaqd', h, W_stack)

    # Split q,k,v -> each [B, T, A, H]
    q = qkv[..., 0, :]
    k = qkv[..., 1, :]
    v = qkv[..., 2, :]

    # Reshape to attention input: [B*T, heads, A, Dh]
    def to_soc_attn_input(x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, A, H] -> [B, T, A, heads, Dh]
        x = x.reshape(B, T, A, self.heads, self.head_dim)
        # permute to [B, T, heads, A, Dh] then merge BT -> [B*T, heads, A, Dh]
        x = x.permute(0, 1, 3, 2, 4).reshape(B * T, self.heads, A, self.head_dim)
        return x

    q_att = to_soc_attn_input(q)
    k_att = to_soc_attn_input(k)
    v_att = to_soc_attn_input(v)

    attn_bias = None
    if hasattr(self, 'connect') and self.adjacency_mode == 'socnet':
        attn_bias = self.connect()  # [1, A, A]
        attn_bias = attn_bias.to(device=q_att.device, dtype=q_att.dtype).unsqueeze(1)  # [1,1,A,A]

    dropout_p = self.dropout if self.training else 0.0

    if q_att.is_cuda:
        backends_ok = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        with sdpa_kernel(backends_ok, set_priority=True):
            out = F.scaled_dot_product_attention(q_att, k_att, v_att, attn_mask=attn_bias, dropout_p=dropout_p, is_causal=False)
    else:
        out = F.scaled_dot_product_attention(q_att, k_att, v_att, attn_mask=attn_bias, dropout_p=dropout_p, is_causal=False)

    # out: [B*T, heads, A, Dh] -> un-flatten to [B, T, A, H]
    out = out.reshape(B, T, self.heads, A, self.head_dim).permute(0, 1, 3, 2, 4).reshape(B, T, A, H)
    return out