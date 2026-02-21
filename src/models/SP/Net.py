import torch
from torch import nn
from torch.nn import functional as F
from torchtyping import TensorType

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

MAX_CLIP = 1


class FourierTimeEncoding(nn.Module):
    def __init__(self, dim: int, log_max_time: float = 4.0):  # 10^4 ≈ 10 000
        super().__init__()
        self.register_buffer(
            "freqs", 10 ** torch.linspace(0, log_max_time, dim // 2)
        )

    def forward(self, t: TensorType[("batch", 1)]):
        ang = t * self.freqs
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class LearnableSoftplus(nn.Module):
    def __init__(self, beta_init=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x):
        return F.softplus(x, beta=self.beta.item())


class LearnableScaleParam(nn.Module):
    def __init__(self, alpha_init=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x):
        return torch.clip(x, 0, MAX_CLIP) ** self.alpha


class ClassifierArchitecture(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 64, t_embed_dim: float = 32):
        super(ClassifierArchitecture, self).__init__()
        self.t_embed = FourierTimeEncoding(t_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + t_embed_dim, hidden_dim),
            # nn.Linear(input_dim + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            # nn.Sigmoid(),
            LearnableSoftplus(),
            # LearnableScaleParam(),
        )

        self.init_weights()

    def forward(self, x, t):
        x_ = torch.cat([x, self.t_embed(t)], dim=-1)
        # x_ = torch.cat([x, t], dim=-1)
        y = self.net(x_)
        return torch.clip(y, 0, MAX_CLIP)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
