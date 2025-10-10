import torch
import torch.nn.functional as F


class GaussianDiffusion:
    def __init__(self, timesteps=300):
        # Define beta schedule
        self.T = timesteps
        self.betas = self.linear_beta_schedule(timesteps=self.T)

        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        # Keep on CPU; move per-use in _extract
        return torch.linspace(start, end, timesteps, dtype=torch.float32)

    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def _extract(self, vec, t, x):
        # vec: [T] (CPU), t: [B] (device of x). Move vec to t.device before gather.
        vec = vec.to(t.device)
        out = vec.gather(0, t)
        shape = [t.shape[0]] + [1] * (x.dim() - 1)
        return out.reshape(*shape)  # already on t/x device

    @torch.no_grad()
    def forward_diffusion_sample(self, x, t, device=None):
        # x: [B, ...], t: [B]
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x)
        x_noisy = sqrt_alphas_cumprod * x + sqrt_one_minus * noise
        return x_noisy, noise

    @torch.no_grad()
    def sample_timestep(self, model, x, t, y=None):
        # x: [B, ...], t: [B] or int
        if not torch.is_tensor(t):
            t = torch.full((x.shape[0],), int(t), device=x.device, dtype=torch.long)

        beta_t = self._extract(self.betas, t, x)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x)
        sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, t, x)

        eps_theta = model(x, t, y)

        # Mean of q(x_{t-1} | x_t, x_0) parameterized by eps_theta
        mean = sqrt_recip_alpha * (x - (beta_t / sqrt_one_minus) * eps_theta)

        # Add noise only when t > 0 (per-sample mask)
        z = torch.randn_like(x)
        nonzero_mask = (t > 0).float().view(-1, *([1] * (x.dim() - 1)))
        x_prev = mean + nonzero_mask * torch.sqrt(beta_t) * z
        return x_prev
        

if __name__ == "__main__":
    diffusion = GaussianDiffusion(timesteps=10)
    x = torch.randn((64, 181))
    t = torch.tensor([0])
    noisy_image, noise = diffusion.forward_diffusion_sample(x, t)
    print(f"Timestep {0}:")
    print("Noisy image shape:", noisy_image.shape)
    print("Noise shape:", noise.shape)
    print()