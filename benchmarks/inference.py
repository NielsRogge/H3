import argparse

import torch

from transformers import GPT2Tokenizer

from src.models.ssm_seq import SSMLMHeadModel


parser = argparse.ArgumentParser(description='H3 generation benchmarking')
parser.add_argument('--dmodel', type=int, default=2048)
parser.add_argument('--nlayer', type=int, default=24)
parser.add_argument('--attn-layer-idx', type=list, default=[8, 16])
parser.add_argument('--nheads', type=int, default=16)
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--promptlen', type=int, default=1024)
parser.add_argument('--genlen', type=int, default=128)
args = parser.parse_args()

repeats = 3
device = 'cuda'
dtype = torch.float16
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# set seed
torch.random.manual_seed(0)
d_model = args.dmodel
n_layer = args.nlayer
ssm_cfg = dict(mode='diag', measure='diag-lin')
attn_layer_idx = args.attn_layer_idx
attn_cfg = dict(num_heads=args.nheads)
model = SSMLMHeadModel(d_model, n_layer=n_layer, d_inner=4 * d_model, vocab_size=len(tokenizer),
                       ssm_cfg=ssm_cfg, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg,
                       pad_vocab_size_multiple=8).to(device=device)
print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
if args.ckpt is not None:
    state_dict = torch.load(args.ckpt, map_location=device)
    if 'pytorch-lightning_version' in state_dict:
        state_dict = {k[len('model.'):]: v for k, v in state_dict['state_dict'].items()
                      if k.startswith('model.')}
    model.load_state_dict(state_dict)
model.eval()
# Only cast the nn.Linear parameters to dtype, the SSM params stay in fp32
# Pytorch lacks support for complex32 (i.e. complex<float16>) and complex<bfloat16>.
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)):
        module.to(dtype=dtype)

print("Generating text...")
input_ids = tokenizer('I enjoy walking with my cute dog', return_tensors='pt').cuda()
max_length = input_ids.shape[1] + args.genlen

outputs = model.generate(input_ids=input_ids, max_length=max_length)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

