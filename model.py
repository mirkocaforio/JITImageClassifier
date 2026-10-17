import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input tensor to help the model understand the order of the sequence.
    The Transformer architecture does not have an inherent notion of position, so this class encodes
    positional information using sine and cosine functions of different frequencies.
    Args:
        d_model (int): Dimension of the model (the size of the input embedding).
        device (str): Device on which the positional encodings are calculated (e.g., 'cpu', 'cuda').
        max_len (int): Maximum length of the input sequence to encode positions for.
    """

    def __init__(self, d_model, device="cpu", max_len=500):
        super().__init__()

        # Create a positional encoding matrix of shape (max_len, d_model)
        self.pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).type(torch.float32)
        tmp = torch.arange(0, d_model, 2)
        den = 1 / torch.pow(torch.ones(int(d_model / 2)) * max_len, 2 * tmp / d_model)
        den = den.unsqueeze(0)

        # Populate the positional encoding matrix with sine and cosine functions
        self.pe[:, 0::2] = torch.sin(torch.matmul(pos, den))
        self.pe[:, 1::2] = torch.cos(torch.matmul(pos, den))
        self.pe = self.pe.to(device)

    def forward(self, x):
        """
        Adds the positional encoding to the input tensor.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            Tensor: Input tensor with positional encoding added.
        """
        return x + self.pe[: x.shape[1], :]


class PreLayer(nn.Module):
    """
    A pre-processing layer that applies a linear transformation to input data, mapping it to the model's
    dimensionality.
    Args:
        hid (int): Hidden layer size.
        d_model (int): Dimension of the model.
        drop_out (float): Dropout rate.
        in_dim (int): Dimension of the input data.
    """

    def __init__(self, hid, d_model, drop_out=0.0, in_dim=1):
        super().__init__()
        self.linear = nn.Linear(in_dim, d_model)

    def forward(self, x):
        """
        Transforms the input using a linear layer.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, in_dim).
        Returns:
            Tensor: Transformed tensor of shape (batch_size, seq_len, d_model).
        """

        out = self.linear(x)
        return out


class PostLayer(nn.Module):
    """
    A post-processing layer that applies a linear transformation to reduce the dimensionality to the desired
    output size.
    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hid (int): Hidden layer size.
        drop_out (float): Dropout rate.
    """

    def __init__(self, in_dim, out_dim, hid, drop_out=0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        """
        Applies a linear transformation to the input data.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, in_dim).
        Returns:
            Tensor: Transformed tensor of shape (batch_size, seq_len, out_dim).
        """

        out = self.linear(x)
        return out


class SelfAttention(nn.Module):
    """
    Self-Attention mechanism to compute attention scores and apply them to the input sequence. This helps
    the model focus on different parts of the input sequence when making predictions.
    Args:
        d_model (int): Dimension of the model.
        n_head (int): Number of attention heads (for multi-head attention).
        attn_type (str): Type of attention mechanism (e.g., 'full' or 'causal').
    """

    def __init__(self, d_model, n_head, attn_type):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.n_head = n_head
        self.attn_type = attn_type
        self.softmax = nn.Softmax(dim=-1)
        scale = torch.sqrt(torch.FloatTensor([d_model // n_head]))
        self.register_buffer("scale", scale)

    def forward(self, x, mask=None, kv=None):
        """
        Applies the (self- or cross-) attention mechanism to the input sequence.
        Args:
            x (Tensor): Query input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor, optional): Mask to avoid attending to certain positions (e.g., padding or future tokens).
            kv (Tensor, optional): Key/value source of shape (batch_size, kv_len, d_model). When None the
                module performs standard self-attention (kv = x); when provided it performs cross-attention,
                with queries from ``x`` and keys/values from ``kv`` (e.g. the encoder memory).
        Returns:
            Tensor: Output of the attention mechanism with the same shape as ``x``.
        """

        if kv is None:
            kv = x
        B, N, D = x.shape
        M = kv.shape[1]
        q = self.query(x).view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        k = self.key(kv).view(B, M, self.n_head, D // self.n_head).transpose(1, 2)
        v = self.value(kv).view(B, M, self.n_head, D // self.n_head).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.softmax(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        return out


class FeedForward(nn.Module):
    """
    A two-layer feed-forward neural network applied independently to each position in the sequence.
    Args:
        d_model (int): Dimension of the model.
        ff_hidnum (int): Size of the hidden layer in the feed-forward network.
    """

    def __init__(self, d_model, ff_hidnum):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ff_hidnum)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ff_hidnum, d_model)

    def forward(self, x):
        """
        Applies the feed-forward network to the input.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            Tensor: Output tensor with the same shape as the input.
        """
        return self.linear2(self.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    """
    A single layer of the Transformer encoder. It consists of a self-attention mechanism followed by
    a feed-forward network, both with residual connections and layer normalization.
    Args:
        d_model (int): Dimension of the model.
        n_head (int): Number of attention heads.
        attn_type (str): Type of attention.
        ff_hidnum (int): Hidden layer size of the feed-forward network.
        drop_out (float): Dropout rate.
    """

    def __init__(self, d_model, n_head, attn_type, ff_hidnum, drop_out=0.1):
        super().__init__()
        self.attn = SelfAttention(d_model, n_head, attn_type)
        self.ff = FeedForward(d_model, ff_hidnum)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)

    def forward(self, x, mask=None):
        """
        Forward pass through a single encoder layer.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor, optional): Optional attention mask.
        Returns:
            Tensor: Output of the encoder layer with the same shape as the input.
        """

        x = self.norm1(x + self.dropout1(self.attn(x, mask)))
        x = self.norm2(x + self.dropout2(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    """
    A single layer of the Transformer decoder. It consists of two attention mechanisms (self-attention
    and encoder-decoder attention) followed by a feed-forward network. Residual connections and
    layer normalization are applied after each sub-layer.
    Args:
        d_model (int): Dimension of the model.
        n_head (int): Number of attention heads.
        attn_type (str): Type of attention.
        ff_hidnum (int): Hidden layer size of the feed-forward network.
        drop_out (float): Dropout rate.
    """

    def __init__(self, d_model, n_head, attn_type, ff_hidnum, drop_out=0.1):
        super().__init__()
        self.attn1 = SelfAttention(d_model, n_head, attn_type)
        self.attn2 = SelfAttention(d_model, n_head, attn_type)
        self.ff = FeedForward(d_model, ff_hidnum)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)
        self.dropout3 = nn.Dropout(drop_out)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        """
        Forward pass through a single decoder layer.
        Args:
            x (Tensor): Target sequence input.
            memory (Tensor): Output from the encoder.
            src_mask (Tensor): Mask for the source sequence.
            tgt_mask (Tensor): Mask for the target sequence.
        Returns:
            Tensor: Output of the decoder layer with the same shape as the input.
        """

        x = self.norm1(x + self.dropout1(self.attn1(x, mask=tgt_mask)))
        x = self.norm2(x + self.dropout2(self.attn2(x, mask=src_mask, kv=memory)))
        x = self.norm3(x + self.dropout3(self.ff(x)))
        return x


class Encoder(nn.Module):
    """
    The Transformer encoder, which is composed of multiple stacked encoder layers.
    Args:
        N (int): Number of encoder layers.
        d_model (int): Dimension of the model.
        n_head (int): Number of attention heads.
        attn_type (str): Type of attention.
        ff_hidnum (int): Hidden layer size of the feed-forward network.
        drop_out (float): Dropout rate.
    """

    def __init__(self, N, d_model, n_head, attn_type, ff_hidnum, drop_out=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, n_head, attn_type, ff_hidnum, drop_out)
                for _ in range(N)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Forward pass through the encoder.
        Args:
            x (Tensor): Input sequence tensor.
        Returns:
            Tensor: Final output of the encoder.
        """

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """
    The Transformer decoder, which is composed of multiple stacked decoder layers.
    Args:
        N (int): Number of decoder layers.
        d_model (int): Dimension of the model.
        n_head (int): Number of attention heads.
        attn_type (str): Type of attention.
        ff_hidnum (int): Hidden layer size of the feed-forward network.
        drop_out (float): Dropout rate.
    """

    def __init__(self, N, d_model, n_head, attn_type, ff_hidnum, drop_out=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, n_head, attn_type, ff_hidnum, drop_out)
                for _ in range(N)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        """
        Forward pass through the decoder.
        Args:
            x (Tensor): Target sequence tensor.
            memory (Tensor): Encoder output (source sequence representation).
            src_mask (Tensor): Mask for the source sequence.
            tgt_mask (Tensor): Mask for the target sequence.
        Returns:
            Tensor: Final output of the decoder.
        """

        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """
    Full Transformer model combining the encoder and decoder, adapted for image classification while
    preserving the original autoregressive (feedback) encoder-decoder form.

    The input image is split into a sequence of flattened patches that the encoder processes into a
    memory. The decoder then produces the class prediction autoregressively: starting from a single
    learnable prediction token it runs ``feedback_steps`` refinement steps, appending the softmax of
    each step's logits back as the next decoder token (the "retroazione"/feedback). A causal mask keeps
    the decoder self-attention autoregressive, while the decoder cross-attends to the encoder memory.

    Args:
        device (str): Device on which the model will be run ('cpu' or 'cuda').
        d_model (int): Dimension of the model.
        in_dim_enc (int): Input dimension for the encoder (= channels * patch_size**2).
        in_dim_dec (int): Input dimension for the decoder (= num_classes, the class distribution fed back).
        attn_type (str): Type of attention mechanism.
        N_enc (int): Number of encoder layers.
        N_dec (int): Number of decoder layers.
        h_enc (int): Number of attention heads in the encoder.
        h_dec (int): Number of attention heads in the decoder.
        ff_hidnum (int): Hidden layer size of the feed-forward network.
        hid_pre (int): Dimension for the pre-processing layers.
        hid_post (int): Dimension for the post-processing layers.
        dropout_pre (float): Dropout rate for the pre-processing layer.
        dropout_post (float): Dropout rate for the post-processing layer.
        dropout_model (float): Dropout rate for the encoder and decoder.
        num_classes (int): Number of output classes.
        patch_size (int): Side of the square patches the image is split into.
        img_size (int): Input image size (H = W). Used only for sanity checks.
        in_channels (int): Number of input image channels.
        feedback_steps (int): Number of autoregressive feedback/refinement steps performed by the decoder.
        use_checkpoint (bool): Whether to use gradient checkpointing to save memory.
    """

    def __init__(
        self,
        device,
        d_model,
        in_dim_enc,
        in_dim_dec,
        attn_type,
        N_enc,
        N_dec,
        h_enc,
        h_dec,
        ff_hidnum,
        hid_pre,
        hid_post,
        dropout_pre,
        dropout_post,
        dropout_model,
        num_classes,
        patch_size,
        img_size=224,
        in_channels=3,
        feedback_steps=4,
        use_checkpoint=False,
    ):
        super().__init__()
        self.device = device
        self.use_checkpoint = use_checkpoint
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_channels = in_channels
        self.feedback_steps = feedback_steps
        self.x_pre = PreLayer(hid_pre, d_model, dropout_pre, in_dim_enc)
        self.y_pre = PreLayer(hid_pre, d_model, dropout_pre, in_dim_dec)
        self.pos = PositionalEncoding(d_model, device=device)
        self.enc = Encoder(N_enc, d_model, h_enc, attn_type, ff_hidnum, dropout_model)
        self.dec = Decoder(N_dec, d_model, h_dec, attn_type, ff_hidnum, dropout_model)
        self.post = PostLayer(d_model, num_classes, hid_post, dropout_post)
        # Learnable start token of the decoder, expressed in class-distribution space (in_dim_dec).
        self.y_start = nn.Parameter(torch.zeros(1, 1, num_classes))

    def patchify(self, x):
        """
        Splits a batch of images into a sequence of flattened patches.
        Args:
            x (Tensor): Image batch of shape (B, C, H, W).
        Returns:
            Tensor: Patch sequence of shape (B, num_patches, C * patch_size**2).
        """

        B, C, H, W = x.shape
        P = self.patch_size
        # (B, C, H/P, P, W/P, P)
        x = x.reshape(B, C, H // P, P, W // P, P)
        # -> (B, H/P, W/P, C, P, P) -> (B, num_patches, C*P*P)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, (H // P) * (W // P), C * P * P)
        return x

    def _causal_mask(self, n, device):
        """Lower-triangular causal mask broadcastable to (1, 1, n, n)."""
        mask = torch.tril(torch.ones(n, n, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def embed_input(self, x):
        """
        Turns the input batch into the encoder token sequence.
        Args:
            x (Tensor): Image batch of shape (B, C, H, W).
        Returns:
            Tensor: Embedded patch tokens with positional encoding, shape (B, num_patches, d_model).
        """

        return self.pos(self.x_pre(self.patchify(x)))

    def forward(self, x, return_all_steps=False):
        """
        Forward pass: encode the input tokens, then autoregressively refine the class prediction.
        Args:
            x (Tensor): Input batch (images of shape (B, C, H, W) for this class).
            return_all_steps (bool): If True, also return the per-step logits stack (B, feedback_steps, num_classes).
        Returns:
            Tensor: Final-step class logits of shape (B, num_classes). If ``return_all_steps`` is True,
                returns a tuple ``(final_logits, all_step_logits)``.
        """

        # --- Encoder: input tokens -> memory ---
        x_emb_pos = self.embed_input(x)
        if self.use_checkpoint and self.training:
            memory = checkpoint(self.enc, x_emb_pos, use_reentrant=True)
        else:
            memory = self.enc(x_emb_pos)

        # --- Decoder: autoregressive feedback refinement ---
        B = x.shape[0]
        y = self.y_start.expand(B, -1, -1)  # (B, 1, num_classes)
        step_logits_list = []
        for _ in range(self.feedback_steps):
            n = y.shape[1]
            y_emb_pos = self.pos(self.y_pre(y))
            tgt_mask = self._causal_mask(n, x.device)
            out = self.dec(y_emb_pos, memory, src_mask=None, tgt_mask=tgt_mask)
            step_logits = self.post(out)[:, -1]  # (B, num_classes) for the latest token
            step_logits_list.append(step_logits)
            # Feedback: append the predicted class distribution as the next decoder token.
            next_token = torch.softmax(step_logits, dim=-1).unsqueeze(
                1
            )  # (B, 1, num_classes)
            y = torch.cat([y, next_token], dim=1)

        final_logits = step_logits_list[-1]
        if return_all_steps:
            return final_logits, torch.stack(step_logits_list, dim=1)
        return final_logits


def define_model(
    device,
    num_classes,
    patch_size=16,
    img_size=224,
    in_channels=3,
    feedback_steps=4,
    use_checkpoint=False,
):
    """
    Utility function to create and define the Transformer image classifier with specified parameters.
    Args:
        device (str): Device on which the model will be run ('cpu' or 'cuda').
        num_classes (int): Number of output classes.
        patch_size (int): Side of the square patches the image is split into.
        img_size (int): Input image size (H = W).
        in_channels (int): Number of input image channels.
        feedback_steps (int): Number of autoregressive feedback/refinement steps.
        use_checkpoint (bool): Whether to use gradient checkpointing to save memory during training.
    Returns:
        Transformer: The instantiated Transformer model.
    """

    model = Transformer(
        device=device,
        in_dim_enc=in_channels * patch_size * patch_size,
        in_dim_dec=num_classes,
        d_model=512,
        N_enc=4,
        N_dec=4,
        h_enc=8,
        h_dec=8,
        ff_hidnum=1024,
        hid_pre=16,
        hid_post=8,
        dropout_pre=0.0,
        dropout_post=0.0,
        dropout_model=0.0,
        attn_type="full",
        num_classes=num_classes,
        patch_size=patch_size,
        img_size=img_size,
        in_channels=in_channels,
        feedback_steps=feedback_steps,
        use_checkpoint=use_checkpoint,
    ).to(device)

    return model
