import os

import torch
import torch.nn.functional as F

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.algo.ddppo import DecentralizedDistributedMixin
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.utils.common import (
    LagrangeInequalityCoefficient,
    inference_mode,
)


EPS_PPO = 1e-5


@baseline_registry.register_updater
class HRLPPO(PPO):
    def get_advantages(self, rollouts) -> torch.Tensor:
        """Advantages with normalization statistics over VALID slots only.

        The HRL buffer writes one transition per high-level decision, so most
        slots hold no data. The base implementation normalizes over the full
        buffer, letting those junk slots dominate the mean/std (and they grow
        with the critic's value scale). Valid slots follow the same rule as
        ``data_generator``'s loss_mask: slot < cur_step_idxs[env] - 1.
        Invalid slots are zeroed either way.
        """
        advantages = (
            rollouts.buffers["returns"] - rollouts.buffers["value_preds"]
        )
        num_slots = advantages.size(0)
        valid = (
            torch.arange(num_slots, device=advantages.device).view(-1, 1)
            < (rollouts._cur_step_idxs.to(advantages.device) - 1).view(1, -1)
        ).unsqueeze(-1)
        advantages = advantages.masked_fill(~valid, 0.0)

        if not self.use_normalized_advantage:
            return advantages

        valid_adv = advantages[valid.expand_as(advantages)]
        if valid_adv.numel() < 2:
            return advantages
        # _compute_var_mean is distributed-aware under the DDPPO mixin.
        var, mean = self._compute_var_mean(valid_adv)
        advantages = (advantages - mean).mul_(torch.rsqrt(var + EPS_PPO))
        return advantages.masked_fill(~valid, 0.0)

    def _update_from_batch(self, batch, epoch, rollouts, learner_metrics):
        n_samples = max(batch["loss_mask"].sum(), 1)

        def record_min_mean_max(t: torch.Tensor, prefix: str):
            for name, op in (
                ("min", torch.min),
                ("mean", torch.mean),
                ("max", torch.max),
            ):
                learner_metrics[f"{prefix}_{name}"].append(op(t))

        def reduce_loss(loss):
            return (loss * batch["loss_mask"]).sum() / n_samples

        self._set_grads_to_none()

        (
            values,
            action_log_probs,
            dist_entropy,
            _,
            aux_loss_res,
        ) = self._evaluate_actions(
            batch["observations"],
            batch["recurrent_hidden_states"],
            batch["prev_actions"],
            batch["masks"],
            batch["actions"],
            batch["rnn_build_seq_info"],
        )

        ratio = torch.exp(action_log_probs - batch["action_log_probs"])

        surr1 = batch["advantages"] * ratio
        surr2 = batch["advantages"] * (
            torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
        )
        action_loss = -torch.min(surr1, surr2)
        action_loss = reduce_loss(action_loss)

        values = values.float()
        orig_values = values

        if self.use_clipped_value_loss:
            delta = values.detach() - batch["value_preds"]
            value_pred_clipped = batch["value_preds"] + delta.clamp(
                -self.clip_param, self.clip_param
            )

            values = torch.where(
                delta.abs() < self.clip_param,
                values,
                value_pred_clipped,
            )

        value_loss = 0.5 * F.mse_loss(
            values, batch["returns"], reduction="none"
        )
        value_loss = reduce_loss(value_loss)

        all_losses = [
            self.value_loss_coef * value_loss,
            action_loss,
        ]
        all_losses.extend(v["loss"] for v in aux_loss_res.values())

        dist_entropy = reduce_loss(dist_entropy)
        if isinstance(self.entropy_coef, float):
            all_losses.append(-self.entropy_coef * dist_entropy)
        else:
            all_losses.append(self.entropy_coef.lagrangian_loss(dist_entropy))

        total_loss = torch.stack(all_losses).sum()

        # DAPG_PROBE=1: measure how strong the demo anchor really is relative
        # to the PPO improvement gradient (a "weak" coefficient can still be a
        # strong pull once batch sizes differ), and whether the two gradients
        # agree or fight (cosine). Diagnostic only -- never gates the update.
        if os.environ.get("DAPG_PROBE") == "1" and "dapg" in aux_loss_res:
            params = [
                p
                for p in self.actor_critic.policy_parameters()
                if p.requires_grad
            ]

            def _flat_grad(loss):
                grads = torch.autograd.grad(
                    loss, params, retain_graph=True, allow_unused=True
                )
                return torch.cat(
                    [
                        (g if g is not None else torch.zeros_like(p)).flatten()
                        for g, p in zip(grads, params)
                    ]
                )

            g_ppo = _flat_grad(action_loss)
            g_dapg = _flat_grad(aux_loss_res["dapg"]["loss"])
            n_ppo = g_ppo.norm()
            n_dapg = g_dapg.norm()
            learner_metrics["dapg_grad_ratio"].append(
                (n_dapg / (n_ppo + 1e-12)).detach()
            )
            learner_metrics["dapg_grad_cos"].append(
                (g_ppo @ g_dapg / (n_ppo * n_dapg + 1e-12)).detach()
            )

        total_loss = self.before_backward(total_loss)
        total_loss.backward()
        self.after_backward(total_loss)

        grad_norm = self.before_step()
        self.optimizer.step()
        self.after_step()

        with inference_mode():
            # Stats over real (loss-masked) transitions only; the mostly-empty
            # HRL buffer slots would otherwise dominate them.
            lm = batch["loss_mask"].flatten().bool()
            if lm.any():
                record_min_mean_max(orig_values.flatten()[lm], "value_pred")
                record_min_mean_max(ratio.flatten()[lm], "prob_ratio")
            total_size = batch["loss_mask"].shape[0]
            if isinstance(n_samples, torch.Tensor):
                n_samples = n_samples.item()
            learner_metrics["batch_filled_ratio"].append(
                n_samples / total_size
            )

            learner_metrics["value_loss"].append(value_loss)
            learner_metrics["action_loss"].append(action_loss)
            learner_metrics["dist_entropy"].append(dist_entropy)
            if epoch == (self.ppo_epoch - 1) and lm.any():
                r = ratio.flatten()[lm]
                learner_metrics["ppo_fraction_clipped"].append(
                    (r > (1.0 + self.clip_param)).float().mean()
                    + (r < (1.0 - self.clip_param)).float().mean()
                )

            learner_metrics["grad_norm"].append(grad_norm)
            if isinstance(self.entropy_coef, LagrangeInequalityCoefficient):
                learner_metrics["entropy_coef"].append(
                    self.entropy_coef().detach()
                )
            for name, res in aux_loss_res.items():
                for k, v in res.items():
                    learner_metrics[f"aux_{name}_{k}"].append(v.detach())


@baseline_registry.register_updater
class HRLDDPPO(DecentralizedDistributedMixin, HRLPPO):
    pass
