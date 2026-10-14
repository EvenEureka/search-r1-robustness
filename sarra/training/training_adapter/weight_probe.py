"""Numerical samples from actual live training and inference weights."""
import torch

def probe(state):
 result={}
 for key in sorted(state):
  if key in ['model.embed_tokens.weight','lm_head.weight','model.norm.weight'] or key.endswith(('.input_layernorm.weight','.post_attention_layernorm.weight')):
   tensor=state[key].detach();flat=tensor.reshape(-1);n=min(4096,flat.numel())
   idx=torch.linspace(0,flat.numel()-1,n,device=flat.device,dtype=torch.float64).long()
   values=flat[idx].to(torch.bfloat16).float().cpu();assert torch.isfinite(values).all()
   result[key]={'shape':list(tensor.shape),'values':values.tolist()}
 assert 'model.embed_tokens.weight' in result and 'lm_head.weight' in result and 'model.norm.weight' in result
 return result

