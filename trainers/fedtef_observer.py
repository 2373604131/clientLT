import torch


def normalize_by_mean(x, eps=1e-6):
    x = x.float()
    positive = x[x > float(eps)]
    mean = positive.mean() if positive.numel() else x.mean()
    return x / mean.clamp_min(float(eps))


def inverse_normalize(x, eps=1e-6):
    return 1.0 / (normalize_by_mean(x, eps) + float(eps))


def normalize_by_max(x, eps=1e-6):
    x = x.float()
    max_value = x.max()
    if max_value <= float(eps):
        return torch.zeros_like(x)
    return x / (max_value + float(eps))


class TopologyExposureSurvivalObserver:
    def __init__(
        self,
        num_classes,
        rho=0.8,
        eps=1e-6,
        gate_mode="hard_topk",
        temperature=1.0,
        threshold=0.0,
        tail_topk=30,
        exposure_budget=20,
        survival_budget=10,
        warmup_mode="all_low",
        warmup_rounds=5,
        min_hold=5,
        replace_margin=1.2,
        difficulty_power=1.0,
        w_exposure=1.0,
        w_age=0.5,
        w_survival=1.0,
        evidence_threshold=1e-6,
        oracle_bottom20=False,
        oracle_bottomk=0,
        seed=1,
        dataset_name=None,
        **_,
    ):
        self.C = int(num_classes)
        self.rho = float(rho)
        self.eps = float(eps)
        self.gate_mode = str(gate_mode)
        self.temperature = float(temperature)
        self.threshold = 0.0 if threshold is None else float(threshold)
        self.tail_topk = int(tail_topk)
        self.K_exposure = max(0, int(exposure_budget))
        self.K_survival = max(0, int(survival_budget))
        self.K = max(1, min(self.C, self.K_exposure + self.K_survival))
        self.warmup_mode = str(warmup_mode)
        self.warmup_rounds = int(warmup_rounds)
        self.min_hold = max(0, int(min_hold))
        self.replace_margin = float(replace_margin)
        self.difficulty_power = float(difficulty_power)
        self.w_exposure = float(w_exposure)
        self.w_age = float(w_age)
        self.w_survival = float(w_survival)
        self.evidence_threshold = float(evidence_threshold)
        self.oracle_bottom20 = bool(oracle_bottom20)
        self.oracle_bottomk = int(oracle_bottomk)
        self.seed = int(seed)
        self.dataset_name = dataset_name
        self.score_mode = "oracle_bottom20" if self.oracle_bottom20 else "topology_observer"

        self.exposure_mass = torch.zeros(self.C)
        self.difficulty = torch.zeros(self.C)
        self.survival = torch.ones(self.C)
        self.age = torch.zeros(self.C)
        self.protected_mask = torch.zeros(self.C, dtype=torch.bool)
        self.hold_counter = torch.zeros(self.C)
        self.protected_lifetime = torch.zeros(self.C)
        self.reliability = torch.ones(self.C)
        self.topology_score = torch.zeros(self.C)
        self.last_scores = torch.zeros(self.C)
        self.last_exposure_ids = []
        self.last_survival_ids = []
        self.last_jaccard = 1.0
        self.last_churn_rate = 0.0

        # Compatibility fields for existing diagnostics.
        self.exposure = self.exposure_mass
        self.opportunity_count = torch.zeros(self.C)
        self.class_prior_ema = torch.zeros(self.C)
        self.class_prior_proxy = torch.zeros(self.C)
        self.positive_proxy_ema = torch.zeros(self.C)
        self.observed_count_ema = torch.zeros(self.C)
        self.observation_ema = torch.zeros(self.C)

    def update(
        self,
        exposure_proxy,
        difficulty_proxy=None,
        survival_ratio=None,
        gate=None,
        **_,
    ):
        exposure_proxy = torch.as_tensor(exposure_proxy).cpu().float()
        if difficulty_proxy is None:
            difficulty_proxy = torch.zeros_like(exposure_proxy)
        difficulty_proxy = torch.as_tensor(difficulty_proxy).cpu().float()
        if survival_ratio is None:
            survival_ratio = torch.ones_like(exposure_proxy)
        survival_ratio = torch.as_tensor(survival_ratio).cpu().float().clamp(0.0, 1.0)

        observed = exposure_proxy > self.evidence_threshold
        self.age[observed] = 0.0
        self.age[~observed] += 1.0
        self.opportunity_count += observed.float()

        self.exposure_mass = (
            self.rho * self.exposure_mass
            + (1.0 - self.rho) * normalize_by_mean(exposure_proxy, self.eps)
        )
        self.difficulty = (
            self.rho * self.difficulty
            + (1.0 - self.rho) * normalize_by_mean(difficulty_proxy, self.eps)
        )
        self.survival[observed] = (
            self.rho * self.survival[observed]
            + (1.0 - self.rho) * survival_ratio[observed]
        )
        self.reliability = self.survival.clone().clamp(0.0, 1.0)

        self.exposure = self.exposure_mass
        self.class_prior_proxy = exposure_proxy
        self.positive_proxy_ema = (
            self.rho * self.positive_proxy_ema
            + (1.0 - self.rho) * normalize_by_mean(exposure_proxy, self.eps)
        )
        self.observed_count_ema = self.rho * self.observed_count_ema + (1.0 - self.rho) * observed.float()
        self.observation_ema = self.observed_count_ema
        return self.exposure_mass.clone()

    def compute_scores(self):
        low_exposure = inverse_normalize(self.exposure_mass, self.eps)
        age_score = normalize_by_max(self.age, self.eps)
        low_survival = 1.0 - self.survival.clamp(0.0, 1.0)
        difficulty_gate = torch.pow(
            normalize_by_mean(self.difficulty, self.eps).clamp_min(0.0) + self.eps,
            self.difficulty_power,
        )
        score_exposure = difficulty_gate * (
            self.w_exposure * low_exposure + self.w_age * age_score
        )
        score_survival = difficulty_gate * (self.w_survival * low_survival)
        total_score = score_exposure + score_survival
        self.topology_score = total_score
        return total_score, score_exposure, score_survival

    def _topk(self, scores, k):
        k = max(0, min(int(k), self.C))
        if k == 0:
            return torch.empty(0, dtype=torch.long)
        return torch.topk(scores, k=k).indices

    def _warmup_mask(self, current_round=0):
        proposal = torch.zeros(self.C, dtype=torch.bool)
        if self.warmup_mode.lower() == "all_low":
            ids = torch.arange(max(0, self.C - self.K), self.C)
        elif self.warmup_mode.lower() == "round_robin":
            start = (int(current_round) * self.K) % self.C
            ids = (torch.arange(self.K) + start) % self.C
        else:
            ids = torch.empty(0, dtype=torch.long)
        proposal[ids] = True
        return proposal

    def _oracle_mask(self):
        k = self.oracle_bottomk
        if k <= 0:
            k = max(1, int(round(0.2 * self.C)))
        k = min(k, self.C)
        proposal = torch.zeros(self.C, dtype=torch.bool)
        ids = torch.arange(self.C - k, self.C)
        proposal[ids] = True
        self.last_exposure_ids = [int(x) for x in ids.tolist()]
        self.last_survival_ids = []
        return proposal

    def propose_mask(self):
        total_score, score_exposure, score_survival = self.compute_scores()
        exposure_ids = self._topk(score_exposure, self.K_exposure)
        survival_ids = self._topk(score_survival, self.K_survival)

        proposal = torch.zeros(self.C, dtype=torch.bool)
        proposal[exposure_ids] = True
        proposal[survival_ids] = True
        if int(proposal.sum().item()) < self.K:
            for idx in torch.argsort(total_score, descending=True):
                proposal[idx] = True
                if int(proposal.sum().item()) >= self.K:
                    break

        self.last_exposure_ids = [int(x) for x in exposure_ids.tolist()]
        self.last_survival_ids = [int(x) for x in survival_ids.tolist()]
        return proposal, total_score

    def _update_hold_counters(self, previous, selected):
        new_rows = selected & ~previous
        self.hold_counter[selected] = torch.clamp(self.hold_counter[selected] - 1.0, min=0.0)
        self.hold_counter[new_rows] = float(self.min_hold)
        self.hold_counter[~selected] = 0.0
        self.protected_lifetime[selected] += 1.0
        self.protected_lifetime[~selected] = 0.0

    def _apply_hysteresis(self, proposal):
        scores = self.topology_score
        previous = self.protected_mask.clone()
        forced_keep = previous & (self.hold_counter > 0)
        selected = forced_keep | (proposal & previous)

        candidates = torch.nonzero(proposal & ~selected, as_tuple=False).view(-1)
        candidates = candidates[torch.argsort(scores[candidates], descending=True)] if candidates.numel() else candidates
        for candidate in candidates:
            if int(selected.sum().item()) < self.K:
                selected[candidate] = True
                continue
            removable = torch.nonzero(selected & ~forced_keep, as_tuple=False).view(-1)
            if removable.numel() == 0:
                break
            worst = removable[torch.argmin(scores[removable])]
            if scores[candidate] > self.replace_margin * scores[worst]:
                selected[worst] = False
                selected[candidate] = True

        while int(selected.sum().item()) > self.K:
            removable = torch.nonzero(selected & ~forced_keep, as_tuple=False).view(-1)
            if removable.numel() == 0:
                break
            worst = removable[torch.argmin(scores[removable])]
            selected[worst] = False

        prev_count = int((previous | selected).sum().item())
        if prev_count:
            self.last_jaccard = float((previous & selected).sum().item()) / float(prev_count)
        self.last_churn_rate = float((previous ^ selected).sum().item()) / float(max(1, self.K))
        self._update_hold_counters(previous, selected)
        self.protected_mask = selected
        return selected

    def preview_gate(self, current_round=0):
        if self.oracle_bottom20:
            proposal = self._oracle_mask()
            scores = proposal.float()
        elif int(current_round) < self.warmup_rounds and self.warmup_mode.lower() != "none":
            scores, _, _ = self.compute_scores()
            proposal = self._warmup_mask(current_round)
        else:
            proposal, scores = self.propose_mask()
        return proposal.float(), scores, proposal

    def commit_gate(self, current_round=0):
        if self.oracle_bottom20:
            proposal = self._oracle_mask()
            scores = proposal.float()
            self.topology_score = scores
        elif int(current_round) < self.warmup_rounds and self.warmup_mode.lower() != "none":
            scores, _, _ = self.compute_scores()
            proposal = self._warmup_mask(current_round)
        else:
            proposal, scores = self.propose_mask()
        protected = self._apply_hysteresis(proposal)
        gate = protected.float()
        self.last_scores = scores.detach().clone()
        self.reliability = self.survival.clone().clamp(0.0, 1.0)
        return gate, scores, protected

    def compute_gate(self, current_round=0):
        return self.preview_gate(current_round)

    def compute_tail_context(self, current_round=0):
        gate, scores, protected = self.commit_gate(current_round)
        return gate, scores, protected, self.reliability.clone()
