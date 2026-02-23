"""
R-EAFT v1.1: Robust Entropy-Adaptive Fine-Tuning
=================================================

A drop-in replacement for standard SFT loss that reduces catastrophic forgetting
by using a "shock ratio" to detect confident-but-wrong predictions.

Usage:
    from r_eaft import REAFTLoss
    
    loss_fn = REAFTLoss(shock_threshold=5.0, alpha=3.0)
    loss = loss_fn(logits, labels)
    loss.backward()

Reference:
    Paper: TBD
    GitHub: https://github.com/[repo]/r-eaft
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict


class REAFTLoss(torch.nn.Module):
    """
    R-EAFT Loss Function
    
    Computes a weighted cross-entropy loss where:
    - "Shocked" tokens (high loss, low entropy) get full gradient + amplification
    - "Protected" tokens (confident & correct) get entropy-weighted gradient
    - "Uncertain" tokens get entropy-weighted gradient (approx 1.0)
    
    Args:
        shock_threshold: Threshold for shock ratio (loss/entropy). Higher = more permissive.
                        Recommended range: 3.0 - 5.0. Default: 5.0
        alpha: Amplification factor for shocked tokens. Default: 3.0
        eps: Small constant for numerical stability. Default: 1e-6
    """
    
    def __init__(
        self,
        shock_threshold: float = 5.0,
        alpha: float = 3.0,
        eps: float = 1e-6
    ):
        super().__init__()
        self.shock_threshold = shock_threshold
        self.alpha = alpha
        self.eps = eps
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute R-EAFT loss.
        
        Args:
            logits: Model output logits [batch, seq, vocab]
            labels: Target token IDs [batch, seq]. Use -100 for ignored positions.
            attention_mask: Optional attention mask [batch, seq]
        
        Returns:
            Scalar loss tensor
        """
        return self._compute_loss(logits, labels, attention_mask, return_diagnostics=False)

    def forward_with_diagnostics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Compute R-EAFT loss and return diagnostics.
        """
        return self._compute_loss(logits, labels, attention_mask, return_diagnostics=True)

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        return_diagnostics: bool = False
    ):
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Per-token cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.size())
        
        # Normalized entropy
        probs = F.softmax(shift_logits, dim=-1)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        vocab_size = shift_logits.size(-1)
        max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float32, device=logits.device))
        norm_entropy = entropy / max_entropy  # Range [0, 1]
        
        # Shock ratio: high loss relative to low entropy = surprised
        shock_ratio = token_losses / (norm_entropy + self.eps)
        
        # Determine Shocked tokens
        shocked = shock_ratio > self.shock_threshold
        
        # Weight assignment logic (Matches Protocol A formulation)
        # 1. Base weights = Normalized Entropy (Protects confident-correct, allows uncertain)
        weights = norm_entropy.clone()
        
        # 2. Shocked tokens (Confident + Wrong): Amplification
        if self.alpha > 0:
            # W = 1 + alpha * clamp(shock/tau, max=10)
            shock_factor = torch.clamp(shock_ratio[shocked] / self.shock_threshold, max=10.0)
            weights[shocked] = 1.0 + self.alpha * shock_factor
        
        # Apply mask for ignored tokens
        valid_mask = shift_labels != -100
        if attention_mask is not None:
            valid_mask = valid_mask & attention_mask[..., 1:].bool()
        
        # Weighted mean
        weighted_loss = token_losses * weights
        loss = (weighted_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        
        if return_diagnostics:
             # Identify protected tokens for reporting (Confident correct)
            # We define "Protected" roughly as low-entropy but NOT shocked
            # (Matches old "confident" check just for analytics)
            confident_threshold = 0.3 # Just for reporting
            confident = norm_entropy < confident_threshold
            protected_mask = confident & (~shocked)
            
            return {
                "loss": loss,
                "token_losses": token_losses.detach(),
                "weights": weights.detach(),
                "shock_ratio": shock_ratio.detach(),
                "shocked_mask": shocked.detach(),
                "protected_mask": protected_mask.detach(),
                "n_shocked": (shocked & valid_mask).sum().item(),
                "n_protected": (protected_mask & valid_mask).sum().item(),
                "mean_weight": weights[valid_mask].mean().item()
            }
            
        return loss


def train_with_reaft(
    model,
    dataloader,
    optimizer,
    epochs: int = 3,
    shock_threshold: float = 5.0,
    alpha: float = 3.0,
    device: str = "cuda"
):
    """
    Convenience function to train a model using R-EAFT loss.
    
    Args:
        model: HuggingFace model with .forward(input_ids, attention_mask) -> outputs.logits
        dataloader: DataLoader yielding {"input_ids", "attention_mask", "labels"}
        optimizer: PyTorch optimizer
        epochs: Number of training epochs
        shock_threshold: R-EAFT shock threshold (3.0 - 5.0 recommended)
        alpha: Amplification factor
        device: Device to train on
    
    Returns:
        Trained model
    """
    from tqdm import tqdm
    
    loss_fn = REAFTLoss(shock_threshold=shock_threshold, alpha=alpha)
    model.train()
    
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels, attention_mask)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return model


# --- Recommended Hyperparameters ---
RECOMMENDED_CONFIG = {
    "shock_threshold": 5.0,      # Validated on Qwen3-0.6B, range 3-5 works
    "alpha": 3.0,                # Amplification for shocked tokens
    "learning_rate": 1e-5,       # Standard for small LLMs
    "epochs": 3,                 # Sufficient for domain adaptation
    "batch_size": 4,             # Adjust based on GPU memory
}


if __name__ == "__main__":
    # Example usage
    print("R-EAFT v1.1")
    print("=" * 40)
    print(f"Recommended config: {RECOMMENDED_CONFIG}")
    print("\nUsage:")
    print("  from r_eaft import REAFTLoss, train_with_reaft")
    print("  loss_fn = REAFTLoss(shock_threshold=5.0, alpha=3.0)")
