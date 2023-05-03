import torch
import math

from hijack_utils import CondFunc

# MPS workaround for https://github.com/pytorch/pytorch/issues/96153
CondFunc('torch.narrow', lambda orig_func, *args, **kwargs: orig_func(*args, **kwargs).clone(), None)


# Run randn and randn_like on CPU
def cpu_randn(randn, *args, **kwargs):
    device = kwargs.get('device', 'cpu')
    kwargs.update({'device':'cpu'})
    return randn(*args, **kwargs).to(device if device is not None else 'cpu')

CondFunc('torch.randn', cpu_randn, None)

def cpu_randn_like(randn_like, input, **kwargs):
    device = kwargs.get('device', input.device)
    kwargs.update({'device':'cpu'})
    return randn_like(input, **kwargs).to(device)

CondFunc('torch.randn_like', cpu_randn_like, None)


# Apply MPS fix for clamp/clip
def clamp_fix(clamp, input, min=None, max=None, *args, **kwargs):
    kwargs.update({'min':min.contiguous() if isinstance(min, torch.Tensor) else min, 'max':max.contiguous() if isinstance(max, torch.Tensor) else max})
    return clamp(input.contiguous(), *args, **kwargs)

for funcName in ['torch.Tensor.clip', 'torch.Tensor.clamp', 'torch.clip', 'torch.clamp']:
    CondFunc(funcName, clamp_fix, lambda _, input, min=None, max=None, *args, **kwargs: input.device.type == 'mps')

# MPS workaround for https://github.com/pytorch/pytorch/issues/96113
CondFunc('torch.nn.functional.layer_norm', lambda orig_func, x, normalized_shape, weight, bias, eps, **kwargs: orig_func(x.float(), normalized_shape, weight.float() if weight is not None else None, bias.float() if bias is not None else bias, eps).to(x.dtype), lambda *args, **kwargs: len(args) == 6)


# Cast to float32 on MPS to work around unsupported dtype error
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device, dtype=torch.float32)[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

CondFunc('deepfloyd_if.model.gaussian_diffusion._extract_into_tensor', lambda _, *args, **kwargs: _extract_into_tensor(*args, **kwargs), lambda _, arr, timesteps, broadcast_shape: timesteps.device.type == 'mps')


# Use sub-quadratic attention if xformers is not being used
from sub_quadratic_attention import efficient_dot_product_attention
import deepfloyd_if.model.unet

upcast_attn = False

def QKVAttention_forward(self, qkv, encoder_kv=None):
    bs, width, length = qkv.shape
    if self.disable_self_attention:
        ch = width // (1 * self.n_heads)
        q, = qkv.reshape(bs * self.n_heads, ch * 1, length).split(ch, dim=1)
    else:
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
    if encoder_kv is not None:
        assert encoder_kv.shape[1] == self.n_heads * ch * 2
        if self.disable_self_attention:
            k, v = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
        else:
            ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = torch.cat([ek, k], dim=-1)
            v = torch.cat([ev, v], dim=-1)
    scale = 1 / math.sqrt(math.sqrt(ch))
    q, k, v = map(lambda t: t.permute(0, 2, 1).contiguous(), (q, k, v))
    if deepfloyd_if.model.unet._FORCE_MEM_EFFICIENT_ATTN:
        a = memory_efficient_attention(q, k, v)
        a = a.permute(0, 2, 1)
    else:
        dtype = q.dtype
        if upcast_attn:
            q, k = q.float().contiguous(), k.float().contiguous()
        bytes_per_token = torch.finfo(q.dtype).bits//8
        batch_x_heads, q_tokens, _ = q.shape
        _, k_tokens, _ = k.shape
        q_chunk_size = 1024
        qk_matmul_size_bytes = batch_x_heads * bytes_per_token * q_tokens * k_tokens
    
        chunk_threshold_bytes = 268435456 * bytes_per_token
        kv_chunk_size_min = chunk_threshold_bytes // (batch_x_heads * bytes_per_token * (k.shape[2] + v.shape[2]))
        if chunk_threshold_bytes is not None and qk_matmul_size_bytes <= chunk_threshold_bytes:
            # the big matmul fits into our memory limit; do everything in 1 chunk,
            # i.e. send it down the unchunked fast-path
            query_chunk_size = q_tokens
            kv_chunk_size = k_tokens
        a = efficient_dot_product_attention(
                q,
                k,
                v,
                query_chunk_size=q_chunk_size,
                kv_chunk_size=None,
                kv_chunk_size_min=kv_chunk_size_min,
                use_checkpoint=False,
            )
        a = a.to(dtype)
        a = a.permute(0, 2, 1)
    return a.reshape(bs, -1, length)

CondFunc('deepfloyd_if.model.unet.QKVAttention.forward', lambda _, *args, **kwargs: QKVAttention_forward(*args, **kwargs), None)


from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder

device = 'mps'
if_I = IFStageI('IF-I-XL-v1.0', device=device)
if_II = IFStageII('IF-II-L-v1.0', device=device)
if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device)
t5 = T5Embedder(device=device, torch_dtype=torch.float32) # Use float32 for T5 model because MPS does not support bfloat16

from deepfloyd_if.pipelines import dream

prompt = 'ultra close-up color photo portrait of rainbow owl with deer horns in the woods'
count = 1

result = dream(
    t5=t5, if_I=if_I, if_II=if_II, if_III=if_III,
    prompt=[prompt]*count,
    seed=41,
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "smart100",
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": "smart50",
    },
    if_III_kwargs={
        "guidance_scale": 9.0,
        "noise_level": 20,
        "sample_timestep_respacing": "75",
    },
)

#if_I.show(result['I'], size=3)
#if_I.show(result['II'], size=6)
if_III.show(result['III'])