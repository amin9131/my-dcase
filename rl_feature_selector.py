# rl_feature_selector.py
#
# کنترل‌کننده RL (PPO) برای انتخاب طول پنجره‌ی STFT در مرحله‌ی استخراج ویژگی.
# این فایل شامل:
#   - compute_state:      محاسبه‌ی سبک state از یک config مرجع (بدون گرادیان)
#   - jitter_penalty:     جریمه‌ی نوسان بین انتخاب‌های متوالی
#   - ActorCritic:         شبکه‌ی policy + value (کوچک، مستقل از backbone)
#   - RLAdaptiveWrapper:   backbone موجود (CST_former/SeldModel) + ActorCritic
#   - PPORolloutBuffer:    ذخیره‌ی transitionها در طول یک epoch
#   - ppo_update:          آپدیت clipped-PPO روی ActorCritic (backbone دست‌نخورده می‌مونه)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def compute_state(x_ref, prev_action, num_configs, eps=1e-6):
    """
    x_ref: (B, ch, T, F) -- برش یک config مرجع (میانی) از x_multi
    prev_action: (B,) long یا None -- انتخاب config در فراخوانی/batch قبلی
    خروجی: (B, 4) -- [log(نسبت انرژی باند پایین/بالا), log(spectral flux),
                       log(واریانس انرژی زمانی), prev_action نرمال‌شده]
    """
    with torch.no_grad():
        energy = x_ref.pow(2).mean(dim=(1, 2))          # (B, F) -- میانگین روی کانال و زمان
        F_bins = energy.shape[-1]
        low = energy[:, :F_bins // 2].mean(dim=-1)
        high = energy[:, F_bins // 2:].mean(dim=-1) + eps
        energy_ratio = low / high                        # (B,)

        flux = (x_ref[:, :, 1:, :] - x_ref[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))  # (B,)

        frame_energy = x_ref.pow(2).mean(dim=(1, 3))      # (B, T)
        temporal_var = frame_energy.var(dim=-1)            # (B,)

        B = x_ref.shape[0]
        if prev_action is None:
            prev_norm = torch.zeros(B, device=x_ref.device)
        else:
            prev_norm = prev_action.float() / max(num_configs - 1, 1)

        raw = torch.stack([energy_ratio, flux, temporal_var], dim=-1).clamp(min=0)
        state = torch.cat([torch.log1p(raw), prev_norm.unsqueeze(-1)], dim=-1)  # (B, 4)
    return state


def jitter_penalty(action, prev_action, num_configs):
    """
    فاصله‌ی نرمال‌شده‌ی بین config فعلی و قبلی -- در بازه‌ی [0, 1].
    هرچه پرش بین resolutionها بزرگ‌تر باشه، جریمه بیشتره.
    prev_action=None (اولین batch) -> جریمه صفر.
    """
    if prev_action is None:
        return torch.zeros_like(action, dtype=torch.float32)
    return (action.float() - prev_action.float()).abs() / max(num_configs - 1, 1)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, num_configs, hidden=32):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh())
        self.actor_head = nn.Linear(hidden, num_configs)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, state):
        h = self.trunk(state)
        logits = self.actor_head(h)
        value = self.critic_head(h)
        return logits, value


class RLAdaptiveWrapper(nn.Module):
    """
    backbone: نمونه‌ی موجود CST_former یا SeldModel -- بدون هیچ تغییری.
    num_configs: تعداد resolutionهای کاندید (طول params['feat_win_configs']).
    """
    def __init__(self, backbone, num_configs, hidden=32):
        super().__init__()
        self.backbone = backbone
        self.num_configs = num_configs
        self.actor_critic = ActorCritic(state_dim=4, num_configs=num_configs, hidden=hidden)

    def forward(self, x_multi, prev_action=None, greedy=False):
        """
        x_multi: (B, num_configs, ch, T, F)
        خروجی: doa, action, log_prob, value, entropy, state
        """
        ref_idx = self.num_configs // 2
        x_ref = x_multi[:, ref_idx]
        state = compute_state(x_ref, prev_action, self.num_configs)

        logits, value = self.actor_critic(state)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if greedy else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        idx = action.view(-1, 1, 1, 1, 1).expand(-1, 1, *x_multi.shape[2:])
        x_selected = torch.gather(x_multi, 1, idx).squeeze(1)   # (B, ch, T, F)

        doa = self.backbone(x_selected)
        return doa, action, log_prob, value.squeeze(-1), entropy, state

    def evaluate_actions(self, state, action):
        """
        برای فاز PPO update: فقط از state ذخیره‌شده استفاده می‌کنه، بدون اجرای دوباره‌ی backbone.
        """
        logits, value = self.actor_critic(state)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, value.squeeze(-1), entropy


class PPORolloutBuffer:
    def __init__(self):
        self.clear()

    def add(self, state, action, log_prob, value, reward):
        self.states.append(state.detach())
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.rewards.append(reward.detach())

    def clear(self):
        self.states, self.actions = [], []
        self.log_probs, self.values, self.rewards = [], [], []

    def __len__(self):
        return len(self.states)

    def compute_returns_advantages(self, gamma=0.0, lam=0.0):
        """
        پیش‌فرض gamma=lam=0 یعنی advantage تک-گامی (reward - value) -- به دلیل نبودِ
        پیوستگی lane بین batchهای shuffle‌شده (توضیح بالا). اگر بعدا rollout بدون
        shuffle/per_file پیاده کردید، می‌تونید این مقادیر رو بالا ببرید.
        """
        rewards = torch.stack(self.rewards)   # (T, B)
        values = torch.stack(self.values)     # (T, B)

        if gamma == 0 and lam == 0:
            advantages = rewards - values
            returns = rewards
        else:
            T = rewards.shape[0]
            advantages = torch.zeros_like(rewards)
            gae = torch.zeros_like(rewards[0])
            for t in reversed(range(T)):
                next_value = values[t + 1] if t + 1 < T else torch.zeros_like(values[0])
                delta = rewards[t] + gamma * next_value - values[t]
                gae = delta + gamma * lam * gae
                advantages[t] = gae
            returns = advantages + values

        return advantages.reshape(-1), returns.reshape(-1)

    def get_tensors(self, gamma=0.0, lam=0.0):
        states = torch.cat(self.states, dim=0)
        actions = torch.cat(self.actions, dim=0)
        old_log_probs = torch.cat(self.log_probs, dim=0)
        advantages, returns = self.compute_returns_advantages(gamma, lam)
        return states, actions, old_log_probs, returns, advantages


def ppo_update(actor_critic, buffer, optimizer, clip_eps=0.2, epochs=4,
                minibatch_size=64, ent_coef=0.01, vf_coef=0.5, gamma=0.0, lam=0.0):
    """
    فقط پارامترهای actor_critic آپدیت می‌شن -- optimizer باید جدا از optimizer اصلیِ
    backbone باشه (که با seld_loss عادی آپدیت می‌شه).
    """
    states, actions, old_log_probs, returns, advantages = buffer.get_tensors(gamma, lam)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = states.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, minibatch_size):
            idx = perm[start:start + minibatch_size]

            logits, values = actor_critic(states[idx])
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions[idx])
            entropy = dist.entropy()

            ratio = (new_log_probs - old_log_probs[idx]).exp()
            surr1 = ratio * advantages[idx]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values.squeeze(-1), returns[idx])
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    buffer.clear()