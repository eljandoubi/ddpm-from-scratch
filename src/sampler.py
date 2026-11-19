import torch


class Sampler:
    def __init__(
        self,
        total_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device="cuda",
    ):
        self.total_timesteps = total_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        ### Define Basic Beta Scheduler ###
        self.beta_schedule = self.linear_beta_schedule()

        ### Compute Alphas for Direction 0 > t Noise Calculation ###
        self.alpha = 1 - self.beta_schedule
        self.alpha_cumulative_prod = torch.cumprod(self.alpha, dim=-1)

    def linear_beta_schedule(self):
        return torch.linspace(
            self.beta_start, self.beta_end, self.total_timesteps, device=self.device
        )

    def _repeated_unsqueeze(self, target_shape: torch.Tensor, input: torch.Tensor):
        while target_shape.dim() > input.dim():
            input = input.unsqueeze(-1)
        return input

    def add_noise(self, inputs: torch.Tensor, timesteps: torch.Tensor):
        alpha_cumulative_prod_timesteps = self.alpha_cumulative_prod[timesteps]

        ### Compute Mean Coefficient ###
        mean_coeff = alpha_cumulative_prod_timesteps**0.5

        ### Compute Variance Coefficient ###
        var_coeff = (1 - alpha_cumulative_prod_timesteps) ** 0.5

        ### Reshape mean_coeff and var_coeff to have shape (batch x 1 x 1 x 1) so we can broadcast with input (batch x c x height x width) ###
        mean_coeff = self._repeated_unsqueeze(inputs, mean_coeff)
        var_coeff = self._repeated_unsqueeze(inputs, var_coeff)

        ### Generate some Noise X ~ N(0,1) (rand_like will automatically place on same device as the inputs) ###
        noise = torch.randn_like(inputs)

        ### Compute Mean (mean_coef * x_0) ###
        mean = mean_coeff * inputs

        ### Compute Variance ###
        var = var_coeff * noise

        ### Compute Noisy Data ###
        noisy_image = mean + var

        return noisy_image, noise

    def remove_noise(
        self,
        input: torch.Tensor,
        timestep: torch.Tensor,
        predicted_noise: torch.Tensor,
        clip_x0: bool = False,
    ):
        assert input.shape == predicted_noise.shape, (
            "Shapes of noise pattern and input image must be identical!!"
        )

        ### Create a mask (if timestep == 0 sigma_z will also be 0 so we need to save this for later ###
        greater_than_0_mask = (timestep >= 1).int()

        ### Compute Sigma (b_t * (1 - cumulative_a_(t-1)) / (1 - cumulative_a)) * noise ###
        alpha_cumulative_t = self.alpha_cumulative_prod[timestep]
        # (timestep - 1) if timestep is 0 is WRONG! we will multiply by 0 later
        alpha_cumulative_prod_t_prev = self.alpha_cumulative_prod[timestep - 1]
        beta_t = self.beta_schedule[timestep]
        noise = torch.randn_like(input)
        variance = (
            beta_t * (1 - alpha_cumulative_prod_t_prev) / (1 - alpha_cumulative_t)
        )

        ### 0 out the variance for if the timestep == 0 ###
        variance = variance * greater_than_0_mask
        variance = self._repeated_unsqueeze(input, variance)
        sigma_z = noise * variance**0.5

        ### Compute Noise Coefficient (1 - a_t / sqrt(1 - cumulative_a)) where 1 - a_t = b_t ###
        beta_t = self.beta_schedule[timestep]
        alpha_cumulative_t = self.alpha_cumulative_prod[timestep]
        root_one_minus_cumulative_alpha_t = (1 - alpha_cumulative_t) ** 0.5
        noise_coefficient = beta_t / root_one_minus_cumulative_alpha_t
        noise_coefficient = self._repeated_unsqueeze(input, noise_coefficient)

        ### Compute 1 / sqrt(a_t) ###
        reciprocal_root_a_t = self.alpha[timestep] ** -0.5
        reciprocal_root_a_t = self._repeated_unsqueeze(input, reciprocal_root_a_t)

        if clip_x0:
            ### Posterior mean computed from the clean image estimate, clipped to the image range ###
            root_alpha_cumulative_t = self._repeated_unsqueeze(
                input, alpha_cumulative_t**0.5
            )
            x0_estimate = (
                input
                - self._repeated_unsqueeze(input, root_one_minus_cumulative_alpha_t)
                * predicted_noise
            ) / root_alpha_cumulative_t
            x0_estimate = x0_estimate.clamp(-1, 1)

            ### At timestep 0 the previous cumulative product is 1 (the clean image) ###
            previous = torch.where(
                timestep >= 1,
                alpha_cumulative_prod_t_prev,
                torch.ones_like(alpha_cumulative_prod_t_prev),
            )
            x0_coefficient = previous**0.5 * beta_t / (1 - alpha_cumulative_t)
            ### 1 - cumulative_a loses digits in float32 at timestep 0, where the coefficient is exactly 1 ###
            x0_coefficient = torch.where(
                timestep >= 1, x0_coefficient, torch.ones_like(x0_coefficient)
            )
            input_coefficient = (
                self.alpha[timestep] ** 0.5 * (1 - previous) / (1 - alpha_cumulative_t)
            )
            mean = (
                self._repeated_unsqueeze(input, x0_coefficient) * x0_estimate
                + self._repeated_unsqueeze(input, input_coefficient) * input
            )
            return mean + sigma_z

        ### Compute Denoised Image ###
        denoised = (
            reciprocal_root_a_t * (input - (noise_coefficient * predicted_noise))
            + sigma_z
        )

        return denoised

    def ddim_step(
        self,
        input: torch.Tensor,
        timestep: int,
        prev_timestep: int,
        predicted_noise: torch.Tensor,
        eta: float = 0.0,
        clip_x0: bool = True,
    ):
        ### One DDIM update from timestep to prev_timestep (-1 means the clean image) ###
        alpha_cumulative_t = self.alpha_cumulative_prod[timestep]
        if prev_timestep >= 0:
            alpha_cumulative_prev = self.alpha_cumulative_prod[prev_timestep]
        else:
            alpha_cumulative_prev = torch.ones_like(alpha_cumulative_t)

        ### Estimate x_0 from the predicted noise ###
        x0_estimate = (
            input - (1 - alpha_cumulative_t) ** 0.5 * predicted_noise
        ) / alpha_cumulative_t**0.5
        if clip_x0:
            ### Clip it to the image range, then recompute the noise so that the pair
            ### (x_0, noise) still reproduces the current image ###
            x0_estimate = x0_estimate.clamp(-1, 1)
            predicted_noise = (input - alpha_cumulative_t**0.5 * x0_estimate) / (
                1 - alpha_cumulative_t
            ) ** 0.5

        ### eta = 0 is deterministic, eta = 1 matches the DDPM variance ###
        sigma = (
            eta
            * ((1 - alpha_cumulative_prev) / (1 - alpha_cumulative_t)) ** 0.5
            * (1 - alpha_cumulative_t / alpha_cumulative_prev) ** 0.5
        )
        direction = (1 - alpha_cumulative_prev - sigma**2) ** 0.5 * predicted_noise
        return (
            alpha_cumulative_prev**0.5 * x0_estimate
            + direction
            + sigma * torch.randn_like(input)
        )


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sampler = Sampler(device=device)
    batch = 7
    rand_image = torch.rand(batch, 3, 64, 64, device=device)
    time_steps = torch.randint(0, 1000, (batch,), device=device)
    noise_image, noise = sampler.add_noise(rand_image, time_steps)
    assert noise_image.device == device
    assert noise_image.shape == noise.shape
    new_image = sampler.remove_noise(noise_image, time_steps, noise)
    assert new_image.device == device
