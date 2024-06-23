import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
import torch.nn.functional as F

# torch.set_float32_matmul_precision('medium')

torch.manual_seed(424)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
# device = 'cpu'
class GPTConfig:
    VOCAB_SIZE = 50257
    DIM = 768
    MAX_TOKENS = 1024
    NUM_LAYERS = 12
    NUM_HEADS = 12

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.DIM, 4 * config.DIM, dtype=torch.bfloat16) ## up projection
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.DIM, config.DIM, dtype=torch.bfloat16)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.DIM % config.NUM_HEADS == 0, \
            f'DIM/NUM_HEADS = {config.DIM} / {config.NUM_HEADS} = {config.DIM / config.NUM_HEADS} is not an integer.'

        self.dim, self.num_heads = config.DIM, config.NUM_HEADS
        # key, query, value projections combined.
        self.c_attn = nn.Linear(config.DIM, 3 * config.DIM, dtype=torch.bfloat16)
        self.c_proj = nn.Linear(config.DIM, config.DIM, dtype=torch.bfloat16)
        
    def forward(self, x):
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.dim, dim=-1)
        B, T, C = q.shape
        
        k = k.view(B, T, self.num_heads, self.dim // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, self.dim // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, self.dim // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, self.dim) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.DIM, dtype=torch.bfloat16)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.DIM, dtype=torch.bfloat16)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wpe = nn.Embedding(config.MAX_TOKENS, config.DIM, dtype=torch.bfloat16),
            wte = nn.Embedding(config.VOCAB_SIZE, config.DIM, dtype=torch.bfloat16),
            h = nn.ModuleList([Block(config) for _ in range(config.NUM_LAYERS)]),
            ln_f = nn.LayerNorm(config.DIM, dtype=torch.bfloat16)
        ))
        self.lm_head =  nn.Linear(config.DIM, config.VOCAB_SIZE, bias=False, dtype=torch.bfloat16)
        self.lm_head.weight = self.transformer.wte.weight

    @classmethod
    def from_pretrained(cls, model_name):
        print('Reading pretrained gpt2-weights.')
        gpt2_hf = GPT2LMHeadModel.from_pretrained('gpt2')

        print('Copying pretrained weights!')
        model = GPT(GPTConfig())
        for k in model.state_dict().keys():
            with torch.no_grad():
                ## copy linear layers transposed.
                if any(transposed in k for transposed in ['c_proj.weight', 'c_fc.weight', 'c_attn.weight']):
                    assert len(gpt2_hf.state_dict()[k].shape) == 2, f'{k} : {gpt2_hf.state_dict()[k].shape}'
                    model.state_dict()[k].copy_(gpt2_hf.state_dict()[k].T)
                else:
                    model.state_dict()[k].copy_(gpt2_hf.state_dict()[k])
        return model
    
    def forward(self, inputs_BT, targets_BT = None):
        B, T = inputs_BT.shape
        assert T <= self.config.MAX_TOKENS, f'{T} tokens exceed the limit of {self.config.MAX_TOKENS}'
        
        hidden_states_BTD = self.transformer.wte(inputs_BT)
        position_embeddings_TD = self.transformer.wpe(torch.arange(0, T, device=inputs_BT.device))

        hidden_states_BTD = hidden_states_BTD + position_embeddings_TD
        
        for block in self.transformer.h:
            hidden_states_BTD = block(hidden_states_BTD)
        hidden_states_BTD = self.transformer.ln_f(hidden_states_BTD)
        logits_BTV = self.lm_head(hidden_states_BTD)
        loss = None
        if targets_BT != None:
            loss = F.cross_entropy(logits_BTV.view(-1, logits_BTV.size(-1)), targets_BT.view(-1))
        return logits_BTV, loss
    
    @torch.no_grad
    def generate_(self, inputs_BT, max_tokens = 20):

        while inputs_BT.shape[1] < max_tokens:
            logits_BTV, _ = self(inputs_BT)
            logits_BV = logits_BTV[:,-1,:] ## Take the logits only for last element.

            probs_BV = logits_BV.softmax(dim=-1)
            ## Top-k = 50
            probs_BK, indics_BK = probs_BV.topk(50, dim=-1)

            ixs = probs_BK.multinomial(1)
            next_token_B1 = indics_BK.gather(dim=1, index=ixs)
            inputs_BT = torch.cat((inputs_BT, next_token_B1), dim=1)
        return inputs_BT

    @torch.no_grad
    def generate(self, prompt="", max_tokens=20, num_samples=5):
        input_tokens_BT = torch.tensor(tokenizer.encode(prompt)).repeat(num_samples, 1).to(device)
        generated_BT = self.generate_(input_tokens_BT, max_tokens=max_tokens)
        return [tokenizer.decode(x) for x in generated_BT.tolist()]
            
    
torch.manual_seed(1337)

print('Encoding...')
## Tokenizer
import tiktoken
tokenizer = tiktoken.get_encoding('gpt2')
full_text = open('tinyshakespeare.txt').read()
full_text_encodings = tokenizer.encode(full_text)
print('Text encoded...')

import random 
def get_batch(batch_size=32, max_tokens = 1024):
    start_pos = [random.randint(0, len(full_text_encodings) - max_tokens - 1) for _ in range(batch_size)]
    xs = torch.tensor([full_text_encodings[i:i+max_tokens] for i in start_pos])
    ys = [full_text_encodings[i+1:i+max_tokens+1] for i in start_pos]
    # print(xs, ys)
    return torch.tensor(xs, dtype=torch.long, device=device), torch.tensor(ys, dtype=torch.long, device=device)
    
# model = GPT.from_pretrained('gpt2')
torch.manual_seed(42)
torch.manual_seed(1337)

# print('\n'.join(model.generate("Hello, I'm a language model,", max_tokens=10, num_samples=1)))

model = GPT(GPTConfig()).to(device)
# model = torch.compile(model)
# print('Compiled.')
import time

BATCH_SIZE = 4
MAX_TOKENS = 1024
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
for epoch in range(501):
    if epoch == 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.005
    if epoch == 200:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    start_time = time.time()
    
    xs, ys = get_batch(BATCH_SIZE, MAX_TOKENS)
    _, loss = model(xs, ys)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    end_time = time.time()
    
    print(f'Epoch : {epoch} |Loss : {loss.item()} | Tokens/sec : {BATCH_SIZE * MAX_TOKENS / (end_time - start_time)}')
    if epoch % 50 == 0:
        print('\n'.join(model.generate("\n", max_tokens=20, num_samples=5)))

        
# import code
# code.interact(local=locals())



