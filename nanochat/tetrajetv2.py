from random import randint

from scipy.linalg import hadamard

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def rtn_fp4(x):
    x_abs = tl.abs(x)
    x_sign = tl.where(
        x > 0,
        1,
        -1,
    )
    x_fp4_abs = tl.where(
        x_abs >= 5,
        6,
        tl.where(
            x_abs >= 3.5,
            4,
            tl.where(
                x_abs >= 2.5,
                3,
                tl.where(
                    x_abs >= 1.75,
                    2,
                    tl.where(
                        x_abs >= 1.25,
                        1.5,
                        tl.where(
                            x_abs >= 0.75,
                            1,
                            tl.where(
                                x_abs >= 0.25,
                                0.5,
                                0.0,
                            )
                        )
                    )
                )
            )
        )
    )
    return x_fp4_abs * x_sign


@triton.jit
def sr_fp4(x, sampled_prob):
    x_abs = tl.abs(x)
    x_sign = tl.where(
        x > 0,
        1,
        -1,
    )
    x_fp4_high = tl.where(
        x_abs >= 4,
        6,
        tl.where(
            x_abs >= 3,
            4,
            tl.where(
                x_abs >= 2,
                3,
                tl.where(
                    x_abs >= 1.5,
                    2,
                    tl.where(
                        x_abs >= 1.0,
                        1.5,
                        tl.where(
                            x_abs >= 0.5,
                            1,
                            0.5,
                        )
                    )
                )
            )
        )
    )
    
    x_fp4_low = tl.where(
        x_abs > 4,
        4,
        tl.where(
            x_abs > 3,
            3,
            tl.where(
                x_abs > 2,
                2,
                tl.where(
                    x_abs > 1.5,
                    1.5,
                    tl.where(
                        x_abs > 1.0,
                        1.0,
                        tl.where(
                            x_abs > 0.5,
                            0.5,
                            0.0,
                        )
                    )
                )
            )
        )
    )
    
    prob_up = (x_abs - x_fp4_low) / (x_fp4_high - x_fp4_low)
    return tl.where(
        sampled_prob < prob_up,
        x_fp4_high,
        x_fp4_low,
    ) * x_sign


@triton.jit
def get_scales(x, amax, val_max, scales_max):
    s_dec = tl.where(
        amax == 0.0,
        1.0,
        amax / scales_max / val_max,
    )
    
    s_dec_b = tl.max(tl.abs(x), axis=-1, keep_dims=True) / val_max
    s_dec_b_e4m3 = (s_dec_b / s_dec).to(tl.float8e4nv).to(tl.float32)
    s_dec_b_e4m3 = tl.where(
        s_dec_b_e4m3 == 0,
        1.0,
        s_dec_b_e4m3,
    )
    return s_dec_b_e4m3, s_dec


@triton.jit
def get_alt_scales(x, val_max, s_dec):    
    s_dec_b = tl.max(tl.abs(x), axis=-1, keep_dims=True) / val_max
    s_dec_b_e4m3 = (s_dec_b * (6/4) / s_dec).to(tl.float8e4nv).to(tl.float32)
    s_dec_b_e4m3 = tl.where(
        s_dec_b_e4m3 == 0,
        1.0,
        s_dec_b_e4m3,
    )
    return s_dec_b_e4m3


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64 * 32}),
        triton.Config({"BLOCK_SIZE": 128 * 32}),
        triton.Config({"BLOCK_SIZE": 256 * 32}),
        triton.Config({"BLOCK_SIZE": 512 * 32}),
    ],
    key=[],
)
@triton.jit
def rtn_1x16s_fp4_kernel(
    x_ptr,
    amax_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    scale_override: tl.constexpr,
    group_size: tl.constexpr,
    four_over_six: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):        
    # load x
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # group
    x_grouped = tl.reshape(x_flat, (BLOCK_SIZE // group_size, group_size))
    
    # amax
    scales_max = 256.00 if four_over_six else 448.00
    val_max = 6.0 / scale_override
    amax = tl.load(amax_ptr)
    
    s_dec_b_e4m3, s_dec = get_scales(x_grouped, amax, val_max, scales_max)
    x_scaled = x_grouped / (s_dec_b_e4m3 * s_dec)
    
    # quantize
    x_fp4 = rtn_fp4(x_scaled)
    x_dequantized = x_fp4 * (s_dec_b_e4m3 * s_dec)
    
    if not four_over_six:
        best_x_dequantized = x_dequantized
    else:
        alt_s_dec_b_e4m3 = get_alt_scales(x_grouped, val_max, s_dec)
        alt_x_scaled = x_grouped / (alt_s_dec_b_e4m3 * s_dec)
        
        alt_x_fp4 = rtn_fp4(alt_x_scaled)
        alt_x_dequantized = alt_x_fp4 * (alt_s_dec_b_e4m3 * s_dec)
        
        error_six = tl.sum((x_grouped - x_dequantized) * (x_grouped - x_dequantized), axis=-1, keep_dims=True)
        error_four = tl.sum((x_grouped - alt_x_dequantized) * (x_grouped - alt_x_dequantized), axis=-1, keep_dims=True)
        
        best_x_dequantized = tl.where(
            error_six <= error_four,
            x_dequantized,
            alt_x_dequantized,
        )
    
    x_dequantized_flat = tl.reshape(best_x_dequantized, (BLOCK_SIZE,))
    tl.store(output_ptr + offsets, x_dequantized_flat, mask=mask)


@torch.compiler.disable()
def rtn_1x16s_fp4_kernel_wrapper(
    x,
    scale_override: float,
    group_size: int,
    four_over_six: bool,
):
    x = x.contiguous()
    output = torch.empty_like(x)
    
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    rtn_1x16s_fp4_kernel[grid](
        x_ptr=x,
        amax_ptr=x.abs().max(),
        output_ptr=output,
        n_elements=n_elements,
        scale_override=scale_override,
        group_size=group_size,
        four_over_six=four_over_six,
    )
    return output


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64 * 32}),
        triton.Config({"BLOCK_SIZE": 128 * 32}),
        triton.Config({"BLOCK_SIZE": 256 * 32}),
        triton.Config({"BLOCK_SIZE": 512 * 32}),
    ],
    key=[],
)
@triton.jit
def sr_1x16s_fp4_kernel(
    x_ptr,
    amax_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    scale_override: tl.constexpr,
    group_size: tl.constexpr,
    four_over_six: tl.constexpr,
    seed: int,
    BLOCK_SIZE: tl.constexpr,
):
    # load x
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # group
    x_grouped = tl.reshape(x_flat, (BLOCK_SIZE // group_size, group_size))
    
    # amax
    scales_max = 256.00 if four_over_six else 448.00
    val_max = 6.0 / scale_override
    amax = tl.load(amax_ptr)
    
    s_dec_b_e4m3, s_dec = get_scales(x_grouped, amax, val_max, scales_max)
    x_scaled = x_grouped / (s_dec_b_e4m3 * s_dec)
    
    # quantize
    sampled_prob = tl.rand(seed, offsets).reshape(BLOCK_SIZE // group_size, group_size)
    x_fp4 = sr_fp4(x_scaled, sampled_prob)
    x_dequantized = x_fp4 * (s_dec_b_e4m3 * s_dec)
    
    if not four_over_six:
        best_x_dequantized = x_dequantized
    else:
        alt_s_dec_b_e4m3 = get_alt_scales(x_grouped, val_max, s_dec)
        alt_x_scaled = x_grouped / (alt_s_dec_b_e4m3 * s_dec)
        
        alt_sampled_prob = tl.rand(seed + 1, offsets).reshape(BLOCK_SIZE // group_size, group_size)
        alt_x_fp4 = sr_fp4(alt_x_scaled, alt_sampled_prob)
        alt_x_dequantized = alt_x_fp4 * (alt_s_dec_b_e4m3 * s_dec)
        
        error_six = tl.sum((x_grouped - x_dequantized) * (x_grouped - x_dequantized), axis=-1, keep_dims=True)
        error_four = tl.sum((x_grouped - alt_x_dequantized) * (x_grouped - alt_x_dequantized), axis=-1, keep_dims=True)
        
        best_x_dequantized = tl.where(
            error_six <= error_four,
            x_dequantized,
            alt_x_dequantized,
        )

    x_dequantized_flat = tl.reshape(best_x_dequantized, (BLOCK_SIZE,))
    tl.store(output_ptr + offsets, x_dequantized_flat, mask=mask)


@torch.compiler.disable()
def sr_1x16s_fp4_kernel_wrapper(
    x,
    scale_override: float,
    group_size: int,
    four_over_six: bool,
):
    x = x.contiguous()
    output = torch.empty_like(x)
    seed = randint(0, 1000000)
    
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    sr_1x16s_fp4_kernel[grid](
        x_ptr=x,
        amax_ptr=x.abs().max(),
        output_ptr=output,
        n_elements=n_elements,
        scale_override=scale_override,
        group_size=group_size,
        four_over_six=four_over_six,
        seed=seed,
    )
    return output


def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    return torch.tensor(
        hadamard(group_size) * group_size**-0.5,
        dtype=dtype,
        device=device,
        requires_grad=False,
    )


def rerotate_hadamard(hadamard_matrix):
    signs = torch.diag(
        torch.randint(
            0, 2, (hadamard_matrix.size(0),),
            device=hadamard_matrix.device,
            dtype=hadamard_matrix.dtype
        ) * 2 - 1
    )
    return hadamard_matrix @ signs # NOTE: rerotate along last dim, inner dim for TN GEMM


class TetraJetV2_fn(torch.autograd.Function):
    group_size = 16
    hadamard_matrix = None

    @torch.compile(dynamic=False)
    @staticmethod
    def forward(ctx, input, weight, disable_forward_quant: bool, disable_backward_quant: bool):
        ctx.batch = input.shape[0]
        ctx.seq = input.shape[1]
        ctx.in_dim = weight.shape[1]
        ctx.out_dim = weight.shape[0]
        ctx.disable_backward_quant = disable_backward_quant
        
        if disable_forward_quant:
            input_fp4 = input
            weight_fp4 = weight
        else:
            input_fp4 = rtn_1x16s_fp4_kernel_wrapper(input, scale_override=1.0, group_size=TetraJetV2_fn.group_size, four_over_six=False)
            weight_fp4 = rtn_1x16s_fp4_kernel_wrapper(weight, scale_override=1.0, group_size=TetraJetV2_fn.group_size, four_over_six=False)

        ctx.save_for_backward(input_fp4, weight_fp4)
        return F.linear(input_fp4, weight_fp4)

    @torch.compile(dynamic=False)
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    @staticmethod
    def backward(ctx, grad_output):
        # Load ctx and reshape
        input_fp4, weight_fp4 = ctx.saved_tensors
        weight_fp4 = weight_fp4.to(grad_output.dtype)
        
        input_fp4 = input_fp4.reshape(ctx.batch * ctx.seq, ctx.in_dim)
        grad_output = grad_output.reshape(ctx.batch * ctx.seq, ctx.out_dim)
        
        # Re-randomize the rotation
        TetraJetV2_fn.hadamard_matrix = TetraJetV2_fn.hadamard_matrix.to(grad_output.dtype)
        TetraJetV2_fn.hadamard_matrix = rerotate_hadamard(TetraJetV2_fn.hadamard_matrix)
        
        # No backward quant if flag
        if ctx.disable_backward_quant:
            grad_input = F.linear(
                grad_output,
                weight_fp4.T,
                None,
            ).view(ctx.batch, ctx.seq, ctx.in_dim)
            
            grad_weight = F.linear(
                grad_output.T,
                input_fp4.T,
                None,
            )
            return grad_input, grad_weight, None, None
        
        # EW
        e_ht = (grad_output.reshape(-1, TetraJetV2_fn.hadamard_matrix.size(0)) @ TetraJetV2_fn.hadamard_matrix.T).reshape(ctx.batch * ctx.seq, ctx.out_dim)
        e_ht_fp4 = sr_1x16s_fp4_kernel_wrapper(e_ht, (17 / 16), TetraJetV2_fn.group_size, False)
        
        weight_tht = (weight_fp4.T.reshape(-1, TetraJetV2_fn.hadamard_matrix.size(0)) @ TetraJetV2_fn.hadamard_matrix.T).reshape(ctx.in_dim, ctx.out_dim)
        weight_tht_fp4 = sr_1x16s_fp4_kernel_wrapper(weight_tht, (17 / 16), TetraJetV2_fn.group_size, False)
        
        grad_input = F.linear(
            e_ht_fp4,
            weight_tht_fp4,
            None,
        ).view(ctx.batch, ctx.seq, ctx.in_dim)

        # EtX
        e_tht = (grad_output.T.reshape(-1, TetraJetV2_fn.hadamard_matrix.size(0)) @ TetraJetV2_fn.hadamard_matrix.T).reshape(ctx.out_dim, ctx.batch * ctx.seq)
        e_tht_fp4 = sr_1x16s_fp4_kernel_wrapper(e_tht, (17 / 16), TetraJetV2_fn.group_size, False)
        
        input_tht = (input_fp4.T.reshape(-1, TetraJetV2_fn.hadamard_matrix.size(0)) @ TetraJetV2_fn.hadamard_matrix.T).reshape(ctx.in_dim, ctx.batch * ctx.seq)
        input_tht_fp4 = sr_1x16s_fp4_kernel_wrapper(input_tht, (17 / 16), TetraJetV2_fn.group_size, False)
        
        grad_weight = F.linear(
            e_tht_fp4,
            input_tht_fp4,
            None,
        )
        
        return grad_input, grad_weight, None, None


class TetraJetV2Linear(torch.nn.Linear):
    def __init__(self, *args, hadamard_dim=32, disable_forward_quant=False, disable_backward_quant=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.hadamard_dim = hadamard_dim
        self.disable_forward_quant = disable_forward_quant
        self.disable_backward_quant = disable_backward_quant
        
        if TetraJetV2_fn.hadamard_matrix is None:
            TetraJetV2_fn.hadamard_matrix = get_hadamard_matrix(self.hadamard_dim, device="cuda", dtype=torch.float32)
        
    
    def forward(self, x):
        return TetraJetV2_fn.apply(x, self.weight, self.disable_forward_quant, self.disable_backward_quant)
