import math
from dataclasses import dataclass

import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

MODEL_PATH = "/media/ubuntu/D/Models/pi05_base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = PI05Policy.from_pretrained(MODEL_PATH).to(device)

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    MODEL_PATH,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)


@dataclass
class PrefixHiddenStateResult:
    """
    保存 PI05 VLM（PaliGemma 大脑）前缀前向传播的输出结果。

    字段说明:
        hidden_states:  [B, prefix_seq_len, hidden_dim]  — VLM 最后一层的隐藏状态。
        input_embeds:   [B, prefix_seq_len, hidden_dim]  — 进入 transformer 之前的 token embedding。
        pad_masks:      [B, prefix_seq_len]              — True 表示真实 token，False 表示 padding。
        image_slices:   list[slice]  — 每路相机对应的 token 区间（在 seq_len 维度上的切片）。
        language_slice: slice        — 语言 token 对应的区间。

    使用示例
    --------
    取第 0 路相机的图像隐藏状态:
        result.hidden_states[:, result.image_slices[0], :]   # [B, num_img_tokens, D]

    取语言 token 的隐藏状态:
        result.hidden_states[:, result.language_slice, :]    # [B, num_lang_tokens, D]
    """

    hidden_states: torch.Tensor
    input_embeds: torch.Tensor
    pad_masks: torch.Tensor
    image_slices: list
    language_slice: slice

# ── Tokenizer / Language Encoding Test (runs before extract_vlm_hidden_states) ──
print("=" * 70)
print("  Tokenizer & Language Encoding Test")
print("=" * 70)

TASK = "Pick up the red apple and place it in the box"

# Minimal dummy batch; images/state are zeros — only the task text matters here.
# to_batch_processor wraps the task str in a list,
# pi05_prepare_state_tokenizer_processor_step formats:
#   "Task: {task}, State: {discretized_state};\nAction: "
# tokenizer_processor then encodes with google/paligemma-3b-pt-224.
dummy_batch = {
    "observation.images.base_0_rgb":        torch.zeros(3, 224, 224),
    "observation.images.left_wrist_0_rgb":  torch.zeros(3, 224, 224),
    "observation.images.right_wrist_0_rgb": torch.zeros(3, 224, 224),
    "observation.state":                    torch.zeros(32),
    "task":                                 TASK,
}

preprocessed = preprocess(dummy_batch)

tokens    = preprocessed[OBS_LANGUAGE_TOKENS]          # [1, max_len]
attn_mask = preprocessed[OBS_LANGUAGE_ATTENTION_MASK]  # [1, max_len]

real_len = int(attn_mask[0].sum())

print(f"\nTask string  : {TASK!r}")
print(f"Token shape  : {tuple(tokens.shape)}")
print(f"Real tokens  : {real_len} / {tokens.shape[-1]}")
print(f"Token IDs    : {tokens[0, :real_len].cpu().tolist()}")

# Reload the same tokenizer to decode IDs back to text (verify round-trip)
from transformers import AutoTokenizer as _HFTok
_tok = _HFTok.from_pretrained("google/paligemma-3b-pt-224")
decoded = _tok.decode(tokens[0, :real_len].cpu(), skip_special_tokens=False)
print(f"Decoded text : {decoded!r}")
print("=" * 70)
print()



@torch.no_grad()
def extract_vlm_hidden_states(policy: "PI05Policy", batch: dict) -> PrefixHiddenStateResult:
    """
    对 PI05 VLM（PaliGemma 大脑）执行仅前缀的前向传播，
    返回 last_hidden_state 以及各段 token 的边界信息。

    前缀序列结构:
        [图像0 tokens | 图像1 tokens | ... | 语言 tokens]

    所有 token 都会经过完整的 Gemma transformer（gemma_2b 共 18 层），
    每个位置可以 attend 到所有非 padding 的前缀 token。

    参数:
        policy:  已加载的 PI05Policy 实例（建议设为 eval 模式）。
        batch:   字典，至少包含:
                   - 与 policy.config.image_features 匹配的一路或多路图像
                   - OBS_LANGUAGE_TOKENS        : LongTensor  [B, seq_len]
                   - OBS_LANGUAGE_ATTENTION_MASK: BoolTensor  [B, seq_len]

    返回:
        PrefixHiddenStateResult — 见 dataclass 说明。
    """
    policy.eval()
    model = policy.model          # PI05Pytorch
    pwe = model.paligemma_with_expert  # PaliGemmaWithExpertModel

    # ------------------------------------------------------------------ #
    # 1. 预处理原始图像 → SigLIP 所需格式                                  #
    # ------------------------------------------------------------------ #
    images, img_masks = policy._preprocess_images(batch)

    tokens = batch[OBS_LANGUAGE_TOKENS].to(device)
    lang_pad_mask = batch[OBS_LANGUAGE_ATTENTION_MASK].to(device)  # [B, L]

    # ------------------------------------------------------------------ #
    # 2. 构建前缀 embedding（与 PI05Pytorch.embed_prefix 保持一致）         #
    # ------------------------------------------------------------------ #
    emb_list = []
    pad_list = []
    att_mask_ints = []
    image_slices = []
    current_pos = 0

    for img, img_mask in zip(images, img_masks, strict=True):
        img_emb = pwe.embed_image(img)          # [B, N_img, D_vlm]
        bsize, num_img_tokens = img_emb.shape[:2]

        emb_list.append(img_emb)
        pad_list.append(img_mask[:, None].expand(bsize, num_img_tokens))  # [B, N_img]
        att_mask_ints += [0] * num_img_tokens

        image_slices.append(slice(current_pos, current_pos + num_img_tokens))
        current_pos += num_img_tokens

    # 语言 token embedding
    lang_emb = pwe.embed_language_tokens(tokens)               # [B, L, D_vlm]
    lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])        # 与 embed_prefix 一致的缩放
    num_lang_tokens = lang_emb.shape[1]

    emb_list.append(lang_emb)
    pad_list.append(lang_pad_mask.bool())
    att_mask_ints += [0] * num_lang_tokens

    language_slice = slice(current_pos, current_pos + num_lang_tokens)

    prefix_embs = torch.cat(emb_list, dim=1)       # [B, prefix_len, D_vlm]
    prefix_pad_masks = torch.cat(pad_list, dim=1)  # [B, prefix_len]

    att_masks = torch.tensor(att_mask_ints, dtype=torch.bool, device=prefix_pad_masks.device)
    att_masks = att_masks[None, :].expand(bsize, len(att_mask_ints))  # [B, prefix_len]

    # ------------------------------------------------------------------ #
    # 3. 构建 4D attention mask 和 position_ids                           #
    # ------------------------------------------------------------------ #
    att_2d = make_att_2d_masks(prefix_pad_masks, att_masks)  # [B, L, L]
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1  # [B, L]
    att_4d = model._prepare_attention_masks_4d(att_2d)        # [B, 1, L, L]

    # ------------------------------------------------------------------ #
    # 4. bfloat16 类型转换（与训练时的 forward 保持一致）                   #
    # ------------------------------------------------------------------ #
    q_proj_dtype = pwe.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
    if q_proj_dtype == torch.bfloat16:
        prefix_embs = prefix_embs.to(torch.bfloat16)

    # ------------------------------------------------------------------ #
    # 5. 仅前缀模式前向传播                                                 #
    #    inputs_embeds=[prefix_embs, None] → 只走 PaliGemma，跳过 DiT expert #
    # ------------------------------------------------------------------ #
    (prefix_hidden_states, _), _ = pwe.forward(
        attention_mask=att_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=False,
    )
    # prefix_hidden_states: [B, prefix_len, D_vlm]

    return PrefixHiddenStateResult(
        hidden_states=prefix_hidden_states,
        input_embeds=prefix_embs,
        pad_masks=prefix_pad_masks,
        image_slices=image_slices,
        language_slice=language_slice,
    )


# ------------------------------------------------------------------ #
# 快速冒烟测试 — 替换成真实数据集的 batch 即可正式使用                    #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # 构造一个最小的假 batch（1 路相机，batch_size=1）用于测试
    from lerobot.configs.types import FeatureType

    B = 1
    H, W = policy.config.image_resolution
    img_key = next(k for k, v in policy.config.image_features.items()
                   if v.type == FeatureType.VISUAL)
    lang_len = 64  # 典型的指令 token 长度

    fake_batch = {
        img_key: torch.zeros(B, 3, H, W, device=device),
        OBS_LANGUAGE_TOKENS: torch.zeros(B, lang_len, dtype=torch.long, device=device),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(B, lang_len, dtype=torch.bool, device=device),
    }

    result = extract_vlm_hidden_states(policy, fake_batch)

    print("hidden_states 形状 :", result.hidden_states.shape)
    print("input_embeds  形状 :", result.input_embeds.shape)
    print("image_slices       :", result.image_slices)
    print("language_slice     :", result.language_slice)

    # 示例：取第 0 路相机的图像区域隐藏状态
    img0_hs = result.hidden_states[:, result.image_slices[0], :]
    lang_hs = result.hidden_states[:, result.language_slice, :]
    print("img0 隐藏状态形状  :", img0_hs.shape)  # [B, N_img, D]
    print("lang 隐藏状态形状  :", lang_hs.shape)  # [B, L, D]
