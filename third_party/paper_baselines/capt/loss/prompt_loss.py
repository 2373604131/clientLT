import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptLoss(nn.Module):
    def __init__(self, num_classes, temperature=0.1, tau=1.0, alpha=1.0, beta=1.0, gamma=0.1, delta=1.0, margin=0.5, weight=None):
        super(PromptLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.margin = margin
        self.ce_loss = nn.CrossEntropyLoss()



    def forward(self, general_prompt, class_aware_prompts, image_features, labels, class_priors, x, cls_num_list):
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = 1 * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = None
        x_m = x + self.m_list

        batch_size = image_features.shape[0]
        # Normalize prompts and features
        general_prompt = F.normalize(general_prompt, dim=-1)
        class_aware_prompts = F.normalize(class_aware_prompts, dim=-1)
        image_features = F.normalize(image_features, dim=-1)

        # General prompt loss
        general_similarity = torch.matmul(image_features, general_prompt.t()) / self.temperature

        general_loss = -torch.mean(
            torch.log(
                torch.exp(general_similarity) /
                torch.sum(torch.exp(general_similarity))
            )
        )

        # Class-aware prompt loss
        if class_aware_prompts.dim() == 3:
            class_aware_prompts = class_aware_prompts.mean(dim=1)
        class_aware_similarity = torch.matmul(image_features, class_aware_prompts.t()) / self.temperature


        class_aware_loss = -torch.mean(
            torch.log(
                class_priors[labels] * torch.exp(
                    torch.gather(class_aware_similarity, 1, labels.unsqueeze(1)).squeeze()) /
                torch.sum(class_priors.unsqueeze(0) * torch.exp(class_aware_similarity), dim=1)
            )
        )


        # Combine losses
        total_loss = (
                self.alpha * general_loss +
                self.beta * class_aware_loss + F.cross_entropy(x_m, labels, weight=self.weight)
        )


        return total_loss, general_loss, class_aware_loss


#RN 50
# class PromptLoss(nn.Module):
#     def __init__(self, num_classes, temperature=0.07, alpha=1.0, beta=1.0, gamma=0.1, delta=1.0, margin=0.5):
#         super(PromptLoss, self).__init__()
#         self.num_classes = num_classes
#         self.temperature = temperature
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.delta = delta
#         self.margin = margin
#         self.ce_loss = nn.CrossEntropyLoss()
#
#     def forward(self, general_prompt, class_aware_prompts, image_features, labels, class_priors):
#         batch_size = image_features.shape[0]
#
#
#         # Normalize prompts and features
#         general_prompt = F.normalize(general_prompt, dim=-1)
#         class_aware_prompts = F.normalize(class_aware_prompts, dim=-1)
#         image_features = F.normalize(image_features, dim=-1)
#
#         if general_prompt.dim() == 1:
#             general_prompt = general_prompt.unsqueeze(0)  # [512] -> [1, 512]
#         elif general_prompt.dim() == 2 and general_prompt.shape[1] == 1:
#             general_prompt = general_prompt.t()  # [512, 1] -> [1, 512]
#
#         # General prompt loss
#         general_similarity = torch.matmul(image_features, general_prompt.t()) / self.temperature
#         # print(f"general_similarity shape: {general_similarity.shape}")
#
#         general_loss = -torch.mean(
#             torch.log(
#                 torch.exp(general_similarity) /
#                 torch.sum(torch.exp(general_similarity))
#             )
#         )
#
#         # Class-aware prompt loss
#         if class_aware_prompts.dim() == 3:
#             class_aware_prompts = class_aware_prompts.mean(dim=1)
#         class_aware_similarity = torch.matmul(image_features, class_aware_prompts.t()) / self.temperature
#         # print(f"class_aware_similarity shape: {class_aware_similarity.shape}")
#
#         class_aware_loss = -torch.mean(
#             torch.log(
#                 class_priors[labels] * torch.exp(
#                     torch.gather(class_aware_similarity, 1, labels.unsqueeze(1)).squeeze()) /
#                 torch.sum(class_priors.unsqueeze(0) * torch.exp(class_aware_similarity), dim=1)
#             )
#         )
#
#         # Regularization loss
#         reg_loss = self.compute_regularization_loss(class_aware_prompts)
#
#         # Combine losses
#         total_loss = (
#                 self.alpha * general_loss +
#                 self.beta * class_aware_loss +
#                 self.gamma * reg_loss
#         )
#
#         return total_loss, general_loss, class_aware_loss, reg_loss
#
#     # ... rest of the class remains the same

    def compute_regularization_loss(self, class_aware_prompts):
        normalized_prompts = F.normalize(class_aware_prompts, dim=1)
        similarity_matrix = torch.matmul(normalized_prompts, normalized_prompts.t())
        mask = torch.eye(self.num_classes, device=class_aware_prompts.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, 0)
        reg_loss = torch.mean(F.relu(similarity_matrix - self.margin))
        return reg_loss




def update_class_priors(global_priors, local_priors, idxs_users, datanumber_client):
    total_samples = sum(datanumber_client)
    updated_priors = global_priors.clone()
    for i, idx in enumerate(idxs_users):
        weight = datanumber_client[idx] / total_samples
        updated_priors += weight * (local_priors[i] - global_priors)
    return updated_priors


def get_class_mask(labels, num_classes):
    """
    Create a mask for class-aware prompt gradient update.

    Parameters:
    - labels: Tensor of shape [batch_size]
    - num_classes: Integer

    Returns:
    - mask: Tensor of shape [num_classes]
    """
    mask = torch.zeros(num_classes, device=labels.device)
    mask[labels.unique()] = 1
    return mask


class PromptLearningLoss(nn.Module):
    def __init__(self, num_classes, feature_dim,
                 temperature=0.07, lambda1=1.0, lambda2=0.1, lambda3=0.1,
                 diversity_margin=0.5):
        """
        Initialize the combined loss for prompt learning.

        Args:
            num_classes (int): Number of classes in the dataset
            feature_dim (int): Dimension of the feature embeddings
            temperature (float): Temperature parameter for scaling logits
            lambda1 (float): Weight for class-aware loss
            lambda2 (float): Weight for diversity loss
            lambda3 (float): Weight for alignment loss
            diversity_margin (float): Margin parameter for diversity loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.diversity_margin = diversity_margin

        # Register class frequency buffer (to be updated externally)
        self.register_buffer('class_freq', torch.ones(num_classes))

    def update_class_frequency(self, class_freq):
        """Update the class frequency buffer."""
        self.class_freq.copy_(class_freq)

    # def compute_class_weights(self):
    #     """Compute class weights based on frequency."""
    #     class_probs = self.class_freq / self.class_freq.sum()
    #     # Match dtype of input tensors
    #     return torch.log(class_probs + 1e-12).to(dtype=torch.float16)
    def compute_class_weights(self):
        """Modified class weights with smoother scaling"""
        class_probs = self.class_freq / self.class_freq.sum()
        # Add smoothing factor to prevent extreme weights
        beta = 0.5  # Adjustable parameter
        scaled_probs = class_probs * (1 - beta) + beta / self.num_classes
        return torch.log(scaled_probs + 1e-12).to(dtype=torch.float16)

    def cosine_similarity_matrix(self, x1, x2):
        """Compute cosine similarity matrix between two sets of vectors."""
        x1_norm = F.normalize(x1, p=2, dim=-1)
        x2_norm = F.normalize(x2, p=2, dim=-1)
        return torch.matmul(x1_norm, x2_norm.transpose(-2, -1))

    def general_improved_loss(self, general_prompt, class_prompts, image_features, labels):
        """
        Compute improved general prompt loss.

        Args:
            general_prompt: Shape [C, D] where C is num_classes, D is feature_dim
            class_prompts: Shape [C, D]
            image_features: Shape [B, D] where B is batch_size
            labels: Shape [B]
        """
        batch_size = image_features.size(0)

        # Average the general prompt across classes to get a single prompt vector
        general_prompt_avg = general_prompt.mean(dim=0)  # [D]

        # Combine general prompt with class-specific prompts
        # Expand general prompt: [D] -> [1, D] -> [C, D]
        general_prompt_expanded = general_prompt_avg.unsqueeze(0).expand_as(class_prompts)
        combined_prompts = general_prompt_expanded + class_prompts  # [C, D]

        # Normalize the combined prompts and image features
        combined_prompts = F.normalize(combined_prompts, p=2, dim=-1)  # [C, D]
        image_features = F.normalize(image_features, p=2, dim=-1)  # [B, D]

        # Compute logits: [B, D] @ [D, C] -> [B, C]
        logits = torch.matmul(image_features, combined_prompts.t()) / self.temperature

        # Apply class weights - ensure same device and dtype
        # class_weights = self.compute_class_weights().to(device=logits.device, dtype=logits.dtype)
        # weighted_logits = logits + class_weights.unsqueeze(0)

        # # Compute cross entropy loss
        loss = F.cross_entropy(logits, labels)

        # # Add frequency-based scaling
        # batch_weights = torch.ones_like(labels, dtype=torch.float16)
        # for i, label in enumerate(labels):
        #     class_prob = self.class_freq[label] / self.class_freq.sum()
        #     # Give slightly more weight to head class samples
        #     batch_weights[i] = 1 + 0.2 * torch.log(class_prob + 1e-12)
        #
        # loss = F.cross_entropy(weighted_logits, labels, reduction='none')
        # loss = (loss * batch_weights).mean()

        return loss

    def class_aware_loss(self, class_prompts, image_features, labels):
        """
        Compute class-aware prompt loss.

        Args:
            class_prompts: Shape [C, D]
            image_features: Shape [B, D]
            labels: Shape [B]
        """
        # Normalize features and prompts
        class_prompts = F.normalize(class_prompts, p=2, dim=-1)  # [C, D]
        image_features = F.normalize(image_features, p=2, dim=-1)  # [B, D]

        # Compute logits: [B, D] @ [D, C] -> [B, C]
        logits = torch.matmul(image_features, class_prompts.t()) / self.temperature

        # Apply class weights - ensure same device and dtype
        class_weights = self.compute_class_weights().to(device=logits.device, dtype=logits.dtype)
        weighted_logits = logits + class_weights.unsqueeze(0)

        # Compute cross entropy loss
        loss = F.cross_entropy(weighted_logits, labels)

        return loss

    def diversity_loss(self, class_prompts):
        """
        Compute diversity loss between class prompts.

        Args:
            class_prompts: Shape [C, D]
        """
        # Normalize prompts
        class_prompts = F.normalize(class_prompts, p=2, dim=-1)  # [C, D]

        # Compute pairwise cosine similarities: [C, D] @ [D, C] -> [C, C]
        similarities = torch.matmul(class_prompts, class_prompts.t())

        # Create mask to exclude self-similarities
        mask = torch.eye(self.num_classes, device=class_prompts.device)
        similarities = similarities * (1 - mask)

        # Compute loss using hinge loss formulation
        loss = torch.clamp(similarities - self.diversity_margin, min=0).mean()

        return loss

    def alignment_loss(self, general_prompt, class_prompts):
        """
        Compute alignment loss between general and class-specific prompts.

        Args:
            general_prompt: Shape [C, D]
            class_prompts: Shape [C, D]
        """
        # Average the general prompt across classes
        general_prompt_avg = general_prompt.mean(dim=0)  # [D]

        # Compute target alignments - ensure same device and dtype
        class_weights = self.compute_class_weights().to(device=class_prompts.device, dtype=class_prompts.dtype)
        target_alignments = F.softmax(-class_weights, dim=0)

        # Normalize prompts
        general_prompt_avg = F.normalize(general_prompt_avg, p=2, dim=-1)  # [D]
        class_prompts = F.normalize(class_prompts, p=2, dim=-1)  # [C, D]

        # Compute alignments: [D] @ [D, C] -> [C]
        alignments = torch.matmul(general_prompt_avg, class_prompts.t())

        # Compute MSE loss between actual and target alignments
        loss = F.mse_loss(alignments, target_alignments)

        return loss

    def forward(self, general_prompt, class_prompts, image_features, labels):
        """
        Compute the total loss.

        Args:
            general_prompt (torch.Tensor): General prompt tensor [C, D]
            class_prompts (torch.Tensor): Class-specific prompts tensor [C, D]
            image_features (torch.Tensor): Image features [B, D]
            labels (torch.Tensor): Ground truth labels [B]
        """
        # Compute individual losses
        general_loss = self.general_improved_loss(general_prompt, class_prompts,
                                                  image_features, labels)
        class_aware_loss = self.class_aware_loss(class_prompts, image_features, labels)
        diversity_loss = self.diversity_loss(class_prompts)
        alignment_loss = self.alignment_loss(general_prompt, class_prompts)

        # Combine losses
        total_loss = (general_loss +
                      self.lambda1 * class_aware_loss +
                      self.lambda2 * diversity_loss +
                      self.lambda3 * alignment_loss)

        # Create loss dictionary for monitoring
        loss_dict = {
            'total_loss': total_loss.item(),
            'general_loss': general_loss.item(),
            'class_aware_loss': class_aware_loss.item()
            # 'diversity_loss': diversity_loss.item(),
            # 'alignment_loss': alignment_loss.item()
        }

        return total_loss, general_loss, class_aware_loss, diversity_loss, alignment_loss


# Example usage
"""
# Initialize loss function
criterion = PromptLearningLoss(
    num_classes=10,
    feature_dim=512,
    temperature=0.07,
    lambda1=1.0,
    lambda2=0.1,
    lambda3=0.1,
    diversity_margin=0.5
)

# Update class frequencies (should be done periodically during training)
class_frequencies = torch.ones(10)  # Replace with actual class frequencies
criterion.update_class_frequency(class_frequencies)

# Forward pass
general_prompt = torch.randn(10, 512)  # [C, D]
class_prompts = torch.randn(10, 512)  # [C, D]
image_features = torch.randn(32, 512)  # [B, D]
labels = torch.randint(0, 10, (32,))  # [B]

# Compute loss
total_loss, loss_dict = criterion(general_prompt, class_prompts, 
                                image_features, labels)
"""


class EnhancedPromptLearningLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, temperature=0.07):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.register_buffer('class_freq', torch.ones(num_classes))

    def general_prompt_loss(self, general_prompt, image_features, labels):
        """
        General prompt should learn stable, global features
        """
        # Move to correct device and compute class weights
        class_probs = self.class_freq.to(image_features.device) / self.class_freq.to(image_features.device).sum()
        head_weights = F.softmax(torch.log(class_probs + 1e-12), dim=0)

        # Initialize prototype features
        prototype_features = torch.zeros(self.num_classes, image_features.shape[-1],
                                         device=image_features.device,
                                         dtype=image_features.dtype)
        class_mask = torch.zeros(self.num_classes,
                                 device=image_features.device,
                                 dtype=torch.bool)

        # Compute prototypical features
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.any():
                prototype_features[c] = F.normalize(image_features[mask].mean(0), dim=0)
                class_mask[c] = True

        if not class_mask.any():
            return torch.tensor(0.0, device=image_features.device)

        # Normalize features and compute similarity
        general_features = F.normalize(general_prompt, dim=-1)
        proto_sim = torch.matmul(general_features, prototype_features.t())  # Already normalized

        # Apply weights
        head_weights = head_weights.to(image_features.device)
        valid_weights = head_weights[class_mask]
        valid_sim = proto_sim[:, class_mask]

        # Compute positive loss (now positive due to 1 - sim)
        loss = torch.sum((1 - valid_sim) * valid_weights.unsqueeze(0)) / (class_mask.sum() + 1e-12)

        # Add stability checks
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=image_features.device)

        return loss

    def class_aware_loss(self, class_prompts, image_features, labels):
        """
        Class-aware prompts should learn discriminative features
        - Higher weights for tail classes
        - Enforce inter-class separation
        """
        # Move all tensors to the same device
        class_probs = self.class_freq.to(image_features.device) / self.class_freq.to(image_features.device).sum()
        tail_weights = F.softmax(-torch.log(class_probs + 1e-12), dim=0).to(image_features.device)

        # Standard contrastive loss with tail class emphasis
        logits = torch.matmul(image_features, class_prompts.t()) / self.temperature

        # Apply tail-focused weighting (ensure same device)
        weighted_logits = logits + tail_weights.unsqueeze(0).to(logits.device)
        loss = F.cross_entropy(weighted_logits, labels)

        # Add inter-class separation term (already on correct device from class_prompts)
        class_similarities = torch.matmul(class_prompts, class_prompts.t())
        separation_loss = torch.triu(F.relu(class_similarities - 0.1), diagonal=1).mean()

        return loss + 0.1 * separation_loss

    def complementary_loss(self, general_prompt, class_prompts, image_features):
        """
        Ensure complementary relationship between prompts
        """
        # Normalize prompts
        general_norm = F.normalize(general_prompt, dim=-1)
        class_norm = F.normalize(class_prompts, dim=-1)

        # Compute feature coverage
        combined_features = torch.cat([general_norm, class_norm], dim=0)
        coverage_matrix = torch.matmul(combined_features, combined_features.t())

        # Encourage diversity while maintaining some correlation
        target_correlation = 0.3  # Adjustable parameter
        correlation_loss = F.mse_loss(
            coverage_matrix[:self.num_classes, self.num_classes:],
            torch.ones_like(coverage_matrix[:self.num_classes, self.num_classes:]) * target_correlation
        )

        return correlation_loss

    def forward(self, general_prompt, class_prompts, image_features, labels):
        """Combined loss with balanced emphasis"""
        g_loss = self.general_prompt_loss(general_prompt, image_features, labels)
        c_loss = self.class_aware_loss(class_prompts, image_features, labels)
        comp_loss = self.complementary_loss(general_prompt, class_prompts, image_features)

        # Dynamic weighting based on training progress
        total_loss = (
                0.2 * g_loss +  # General features (head-focused)
                0.6 * c_loss +  # Class-specific features (tail-focused)
                0.2 * comp_loss  # Complementary relationship
        )

        # loss_dict = {
        #     'total_loss': total_loss.item(),
        #     'general_loss': g_loss.item(),
        #     'class_aware_loss': c_loss.item(),
        #     'complementary_loss': comp_loss.item()
        # }

        return total_loss, g_loss, c_loss, comp_loss


class LogitAdjust(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)


class CAPFLLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, temperature=0.07, lambda1=0.5, lambda2=0.1):
        """
        Args:
            num_classes: Number of total classes
            feature_dim: Dimension of features
            temperature: Temperature for scaling logits
            lambda1: Weight for class-aware loss
            lambda2: Weight for alignment loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.register_buffer('class_freq', torch.ones(num_classes))

    def general_prompt_loss(self, general_prompt, image_features, labels):
        """
        Loss for general prompt to learn global features
        - Focus on common patterns across classes
        - More emphasis on head classes naturally through data distribution
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        general_prompt = F.normalize(general_prompt, dim=-1)

        # Compute similarity
        logits = torch.matmul(image_features, general_prompt.t()) / self.temperature

        # Compute loss with implicit head class emphasis (through data distribution)
        loss = F.cross_entropy(logits, labels)

        return loss

    def class_aware_loss(self, class_prompts, image_features, labels, available_classes):
        """
        Loss for class-aware prompt with tail class emphasis
        - Only compute loss for available classes
        - Apply inverse frequency weighting for tail class emphasis
        """
        # Create mask for available classes
        class_mask = torch.zeros(self.num_classes, dtype=torch.bool,
                                 device=image_features.device)
        class_mask[available_classes] = True

        # Compute weights for available classes (inverse frequency weighting)
        class_probs = self.class_freq.to(image_features.device)
        class_probs = class_probs / class_probs.sum()
        weights = 1.0 / (torch.sqrt(class_probs[available_classes]) + 1e-12)
        weights = weights / weights.sum()

        # Compute similarities only for available classes
        image_features = F.normalize(image_features, dim=-1)
        class_prompts = F.normalize(class_prompts[available_classes], dim=-1)

        logits = torch.matmul(image_features, class_prompts.t()) / self.temperature

        # Map labels to available class indices
        target_labels = torch.zeros_like(labels)
        for i, label in enumerate(labels):
            target_labels[i] = torch.where(available_classes == label)[0]

        # Weighted cross entropy loss
        loss = F.cross_entropy(logits, target_labels, weight=weights)

        return loss

    def complementary_alignment_loss(self, general_prompt, class_prompts, image_features):
        """
        Ensure complementary relationship between general and class-specific features
        - Maintain moderate correlation while avoiding redundancy
        """
        image_features = F.normalize(image_features, dim=-1)
        general_prompt = F.normalize(general_prompt, dim=-1)
        class_prompts = F.normalize(class_prompts, dim=-1)

        # Compute similarities with both prompts
        gen_sim = torch.matmul(image_features, general_prompt.t())
        cls_sim = torch.matmul(image_features, class_prompts.t())

        # Target moderate correlation (not too high, not too low)
        target_correlation = 0.3
        actual_correlation = F.cosine_similarity(gen_sim, cls_sim.mean(dim=1, keepdim=True))

        # Loss encourages correlation to be close to target
        loss = F.mse_loss(actual_correlation,
                          torch.ones_like(actual_correlation) * target_correlation)

        return loss

    def forward(self, general_prompt, class_prompts, image_features, labels, available_classes):
        """
        Combined loss computation
        Args:
            general_prompt: Shape [1, D] - global feature prompt
            class_prompts: Shape [C, K, D] - class-specific prompts
            image_features: Shape [B, D] - batch of image features
            labels: Shape [B] - ground truth labels
            available_classes: List of available class indices
        """
        # Compute individual losses
        gen_loss = self.general_prompt_loss(general_prompt, image_features, labels)
        cls_loss = self.class_aware_loss(class_prompts, image_features, labels,
                                         available_classes)
        align_loss = self.complementary_alignment_loss(general_prompt, class_prompts,
                                                       image_features)

        # Combine losses with weights
        total_loss = (
                gen_loss +
                self.lambda1 * cls_loss +
                self.lambda2 * align_loss
        )

        loss_dict = {
            'total_loss': total_loss.item(),
            'general_loss': gen_loss.item(),
            'class_aware_loss': cls_loss.item(),
            'align_loss': align_loss.item()
        }

        return total_loss, loss_dict