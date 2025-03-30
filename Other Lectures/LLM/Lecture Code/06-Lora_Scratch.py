#%% --------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=16):
        super().__init__()
        self.r = r
        self.scaling = alpha / r
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.lora_up   = nn.Linear(r, out_features, bias=False)

        nn.init.zeros_(self.lora_down.weight)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.lora_up(self.lora_down(x)) * self.scaling

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=16, output_dim=2, r=4, alpha=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc2.lora = LoRALayer(hidden_dim, output_dim, r, alpha)
        old_forward = self.fc2.forward
        def forward_with_lora(x):
            out = old_forward(x)
            out += self.fc2.lora(x)
            return out
        self.fc2.forward = forward_with_lora
    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x

#%% --------------------------------------------------------------------------------------------------------------------
model = SimpleMLP(r=4, alpha=16)
x = torch.randn(1, 8)  # random input
out = model(x)

#%% --------------------------------------------------------------------------------------------------------------------
print("Input:", x)
print("Output:", out)
print("Trainable parameters (LoRA) in fc2.lora:", sum(p.numel() for p in model.fc2.lora.parameters() if p.requires_grad))
