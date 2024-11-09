import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vt = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        decoder_state: torch.Tensor,  # (batch_size, hidden_size)
        encoder_outputs: torch.Tensor,  # (batch_size, max_seq_len, hidden_size)
    ) -> torch.Tensor:
        """
        Forward pass for the attention mechanism.

        shapes:
        - decoder_state: (batch_size, hidden_size)
        - encoder_outputs: (batch_size, max_seq_len, hidden_size)
        - return: (batch_size, max_seq_len)
        """

        # Transform encoder outputs for attention calculation
        encoder_transform: torch.Tensor = self.w1(encoder_outputs)
        # Transform decoder state and expand its dimension for addition with encoder_transform
        decoder_transform: torch.Tensor = self.w2(decoder_state)
        decoder_transform = decoder_transform.unsqueeze(1)

        # Compute attention logits with tanh activation and linear projection
        # The addition brings both encoder and decoder to the same space
        tanh_transform_sum = torch.tanh(encoder_transform + decoder_transform)
        attention_logits: torch.Tensor = self.vt(tanh_transform_sum)
        attention_logits = attention_logits.squeeze(-1)

        # Apply softmax to get attention weights across sequence length
        attention_weights: torch.Tensor = F.softmax(attention_logits, dim=1)

        return attention_weights


if __name__ == "__main__":
    attention = Attention(hidden_size=512)
    result = attention(decoder_state=torch.rand(64, 512), encoder_outputs=torch.rand(64, 10, 512))
    assert result.shape == (64, 10)
