import mlx
import mlx.nn as nn
import mlx.core as mx
# from transformers import GPT2LMHeadModel
from mlx.nn.losses import cross_entropy
import mlx.optimizers as optim
from functools import partial


# mlx.set_float32_matmul_precision('medium')

mlx.core.random.seed(424)
class GPTConfig:
    VOCAB_SIZE = 50257
    DIM = 768
    MAX_TOKENS = 1024
    NUM_LAYERS = 12
    NUM_HEADS = 12

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.DIM, 4 * config.DIM) ## up projection
        self.c_fc.weight = self.c_fc.weight.astype(mx.bfloat16)
        self.c_fc.bias = self.c_fc.bias.astype(mx.bfloat16)
        
        self.gelu = nn.GELU(approx='fast')
        self.c_proj = nn.Linear(4 * config.DIM, config.DIM)
        self.c_proj.weight = self.c_proj.weight.astype(mx.bfloat16)
        self.c_proj.bias = self.c_proj.bias.astype(mx.bfloat16)

    def __call__(self, x):
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
        self.c_attn = nn.Linear(config.DIM, 3 * config.DIM)
        self.c_proj = nn.Linear(config.DIM, config.DIM)
        
    def __call__(self, x):
        qkv = self.c_attn(x)
        q, k, v = qkv.split(3, axis=-1)
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
        self.ln_1 = nn.LayerNorm(config.DIM)
        self.ln_1.weight = self.ln_1.weight.astype(mx.bfloat16)
        self.ln_1.bias = self.ln_1.bias.astype(mx.bfloat16)

        self.attn = nn.MultiHeadAttention(dims=config.DIM, num_heads=config.NUM_HEADS)
        self.attn.query_proj.weight = self.attn.query_proj.weight.astype(mx.bfloat16)

        
        self.attn.key_proj.weight = self.attn.key_proj.weight.astype(mx.bfloat16)

        
        self.attn.value_proj.weight = self.attn.value_proj.weight.astype(mx.bfloat16)
        
        self.attn.out_proj.weight = self.attn.out_proj.weight.astype(mx.bfloat16)
        
        ## fix attention
        # self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.DIM)
        self.ln_2.weight = self.ln_2.weight.astype(mx.bfloat16)
        self.ln_2.bias = self.ln_2.bias.astype(mx.bfloat16)
        self.mlp = MLP(config)

    def __call__(self, x):
        x = self.ln_1(x)
        # print('After layer norm dtype: ', x.dtype)
        x = x + self.attn(x, x, x)
        # print('After attention dtype : ', x.dtype)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer_wpe = nn.Embedding(config.MAX_TOKENS, config.DIM)
        self.transformer_wpe.weight = self.transformer_wpe.weight.astype(mx.bfloat16)
        
        self.transformer_wte = nn.Embedding(config.VOCAB_SIZE, config.DIM)
        self.transformer_wte.weight = self.transformer_wte.weight.astype(mx.bfloat16)
        
        self.transformer_h = [Block(config) for _ in range(config.NUM_LAYERS)]
        self.transformer_ln_f = nn.LayerNorm(config.DIM)
        self.transformer_ln_f.weight = self.transformer_ln_f.weight.astype(mx.bfloat16)
        self.transformer_ln_f.bias = self.transformer_ln_f.bias.astype(mx.bfloat16)
        
        # self.lm_head =  nn.Linear(config.DIM, config.VOCAB_SIZE, bias=False)
        # self.lm_head.weight = self.transformer_wte.weight

    # @classmethod
    # def from_pretrained(cls, model_name):
    #     print('Reading pretrained gpt2-weights.')
    #     gpt2_hf = GPT2LMHeadModel.from_pretrained('gpt2')

    #     print('Copying pretrained weights!')
    #     model = GPT(GPTConfig())
    #     for k in model.state_dict().keys():
    #         with mlx.no_grad():
    #             ## copy linear layers transposed.
    #             if any(transposed in k for transposed in ['c_proj.weight', 'c_fc.weight', 'c_attn.weight']):
    #                 assert len(gpt2_hf.state_dict()[k].shape) == 2, f'{k} : {gpt2_hf.state_dict()[k].shape}'
    #                 model.state_dict()[k].copy_(gpt2_hf.state_dict()[k].T)
    #             else:
    #                 model.state_dict()[k].copy_(gpt2_hf.state_dict()[k])
    #     return model
    
    # @mx.compile
    def __call__(self, inputs_BT):
        B, T = inputs_BT.shape
        assert T <= self.config.MAX_TOKENS, f'{T} tokens exceed the limit of {self.config.MAX_TOKENS}'
        
        hidden_states_BTD = self.transformer_wte(inputs_BT)
        position_embeddings_TD = self.transformer_wpe(mx.arange(0, T))
        
        # print('WTE+Pos dtype: ', hidden_states_BTD.dtype)

        hidden_states_BTD = hidden_states_BTD + position_embeddings_TD
        
        for block in self.transformer_h:
            hidden_states_BTD = block(hidden_states_BTD)
        hidden_states_BTD = self.transformer_ln_f(hidden_states_BTD)
        logits_BTV = hidden_states_BTD @ self.transformer_wte.weight.T
        # loss = None
        # if targets_BT != None:
        #     loss = cross_entropy(logits_BTV.view(-1, logits_BTV.size(-1)), targets_BT.view(-1))
        return logits_BTV
    
    # @mlx.no_grad
    def generate_(self, inputs_BT, max_tokens = 20):

        while inputs_BT.shape[1] < max_tokens:
            logits_BTV, _ = self(inputs_BT)
            logits_BV = logits_BTV[:,-1,:] ## Take the logits only for last element.

            probs_BV = logits_BV.softmax(dim=-1)
            ## Top-k = 50
            probs_BK, indics_BK = probs_BV.topk(50, dim=-1)

            ixs = probs_BK.multinomial(1)
            next_token_B1 = indics_BK.gather(dim=1, index=ixs)
            inputs_BT = mlx.cat((inputs_BT, next_token_B1), dim=1)
        return inputs_BT

    # @mlx.no_grad
    def generate(self, prompt="", max_tokens=20, num_samples=5):
        input_tokens_BT = mx.array(tokenizer.encode(prompt)).repeat(num_samples, 1)
        generated_BT = self.generate_(input_tokens_BT, max_tokens=max_tokens)
        return [tokenizer.decode(x) for x in generated_BT.tolist()]
            
    
mlx.core.random.seed(1337)

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
    xs = mx.array([full_text_encodings[i:i+max_tokens] for i in start_pos])
    ys = [full_text_encodings[i+1:i+max_tokens+1] for i in start_pos]
    # print(xs, ys)
    return mx.array(xs, dtype=mx.uint32), mx.array(ys, dtype=mx.uint32)


# model = GPT.from_pretrained('gpt2')
mlx.core.random.seed(42)
mlx.core.random.seed(1337)

# print('\n'.join(model.generate("Hello, I'm a language model,", max_tokens=10, num_samples=1)))

model = GPT(GPTConfig())
mx.eval(model.parameters())
# model = mlx.compile(model)
# print('Compiled.')
import time

BATCH_SIZE = 4
MAX_TOKENS = 1024

optimizer = optim.AdamW(learning_rate=0.01)
state = [model.state, optimizer.state]

def loss_fn(model, x, y, reduce=True):
    logits = model(x)
    # print(logits.dtype)
    losses = nn.losses.cross_entropy(logits, y)
    return mx.mean(losses) if reduce else mx.mean(losses, axis=(-1, -2))

@partial(mx.compile, inputs=state, outputs=state)
def step(inputs, targets):
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model, inputs, targets)
    optimizer.update(model, grads)
    return loss

optimizer.set_dtype(mx.bfloat16)
for epoch in range(100):


    start_time = time.time()
    
    inputs, targets = get_batch(BATCH_SIZE, MAX_TOKENS)
    loss = step(inputs, targets)
    mx.eval(state)
    
    end_time = time.time()
    
    print(f'Epoch : {epoch} |Loss : {loss.item()} | Tokens/sec : {BATCH_SIZE * MAX_TOKENS / (end_time - start_time)}')
    # if epoch % 50 == 0:
    #     print('\n'.join(model.generate("\n", max_tokens=20, num_samples=5)))

        
# import code
# code.interact(local=locals())



