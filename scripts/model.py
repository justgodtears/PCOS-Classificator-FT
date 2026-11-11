import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

MODEL_NAME = "facebook/dinov2-small"
NUM_CLASSES = 2


def find_qv_modules(model):
    # nazwy warstw q/v zależą od wersji transformers, więc wykrywamy je z modelu
    # q/v names depends on transformers version so we gonna detect it from model
    names = {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    for q, v in [("query", "value"), ("q_proj", "v_proj")]:
        if q in names and v in names:
            return [q, v]
    raise ValueError(f"Couldn't find q/v layers, available Linear: {sorted(names)}")


class PcosModel(nn.Module):
    # backbone DINOv2 wtih LoRA and small classification head
    def __init__(self, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        backbone = AutoModel.from_pretrained(MODEL_NAME)

        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=find_qv_modules(backbone),  # LoRA only on attention q and v
        )
        # peft freezes base weights, LoRA matrixes learns only
        self.backbone = get_peft_model(backbone, lora_cfg)

        hidden = backbone.config.hidden_size  # 384 for dinov2-small
        # head on [CLS + mean from patches], trained entirely
        self.head = nn.Linear(hidden * 2, NUM_CLASSES)

    def forward(self, x):
        h = self.backbone(pixel_values=x).last_hidden_state  # [B, 1+N, hidden]
        cls_tok = h[:, 0]
        patch_mean = h[:, 1:].mean(dim=1)
        return self.head(torch.cat([cls_tok, patch_mean], dim=1))


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


if __name__ == "__main__":
    # number of trainable parameters and forward on random batch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PcosModel().to(device)

    trainable, total = count_params(model)
    print(f"trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    head_trainable = all(p.requires_grad for p in model.head.parameters())
    print("head trainable:", head_trainable)

    x = torch.randn(4, 3, 224, 224, device=device)
    out = model(x)
    print("out:", out.shape)  # expected [4, 2]