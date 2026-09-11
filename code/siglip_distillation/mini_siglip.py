"""Small dual Transformer encoders; alignment heads included in parameter counts."""
import torch
from torch import nn
from torch.nn import functional as F

class MiniSiglip(nn.Module):
    def __init__(self, width, depth=2, vocab_size=4096, text_length=64, output_dim=768):
        super().__init__()
        self.config = dict(width=width, depth=depth, vocab_size=vocab_size,
                           text_length=text_length, output_dim=output_dim)
        self.patch = nn.Conv2d(3, width, 16, 16)
        self.image_pos = nn.Parameter(torch.zeros(1, 196, width))
        self.word = nn.Embedding(vocab_size, 32, padding_idx=0)
        self.word_up = nn.Linear(32, width)
        self.text_pos = nn.Parameter(torch.zeros(1, text_length, width))
        self.vision = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            width, 4, 4*width, dropout=0, activation='gelu', batch_first=True,
            norm_first=True), depth, enable_nested_tensor=False)
        self.text = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            width, 4, 4*width, dropout=0, activation='gelu', batch_first=True,
            norm_first=True), depth, enable_nested_tensor=False)
        self.image_head = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,output_dim))
        self.text_head = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,output_dim))
        # TransformerEncoder clones its input layer; independently initialize each block.
        for stack in (self.vision, self.text):
            for layer in stack.layers:
                nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
                for module in layer.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None: nn.init.zeros_(module.bias)
        nn.init.normal_(self.image_pos,std=.02)
        nn.init.normal_(self.text_pos,std=.02)

    def encode_image(self,x):
        x=self.patch(x).flatten(2).transpose(1,2)+self.image_pos
        return F.normalize(self.image_head(self.vision(x).mean(1)).float(),dim=-1)

    def forward(self,images,ids):
        return self.encode_image(images),self.encode_text(ids)

    def encode_text(self,ids):
        mask=ids.ne(0)
        x=self.word_up(self.word(ids))+self.text_pos[:,:ids.shape[1]]
        x=self.text(x,src_key_padding_mask=~mask)
        x=(x*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
        return F.normalize(self.text_head(x).float(),dim=-1)

def make_model(millions):
    # Actual architecture search only; no unused parameters to fill the budget.
    with torch.device('meta'):
        widths=range(32,513,4)
        counts={w:sum(p.numel() for p in MiniSiglip(w).parameters()) for w in widths}
    width=min(counts,key=lambda w:abs(counts[w]-millions*1_000_000))
    return MiniSiglip(width)
