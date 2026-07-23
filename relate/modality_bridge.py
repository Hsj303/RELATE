"""
Modality bridge (paper Sec. 3.1: "two fully connected (FC) layers that
bridge the discrepancy between modalities ... a distributional gap between
I3D features").

At inference time only RGB frames are available. ``ModalityBridge`` maps
RGB I3D features into the optical-flow feature distribution so the flow
branch keeps receiving flow-like input without ever running optical-flow
extraction. It is trained separately (see ``train_modality_bridge``) with an
MSE reconstruction loss against real optical-flow I3D features, then frozen
for RELATE's main training/inference.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim


class ModalityBridge(nn.Module):
    """1x1-conv MLP mapping RGB I3D features to the optical-flow feature space."""

    def __init__(self, dim: int, num_layers: int = 2, hidden_dim: int = 256, kernel_size: int = 1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = dim if i == 0 else hidden_dim
            out_ch = dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=0))
            if i != num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.transform = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


def train_modality_bridge(bridge, batch_gen, device, save_dir, num_epochs=100, batch_size=8, lr=1e-4):
    """
    Pre-trains the modality bridge with an MSE loss between RGB-derived and
    real optical-flow I3D features, following the FC-bridge described in
    Sec. 3.1. Features are assumed to be concatenated as
    ``[rgb (0:1024), flow (1024:2048)]`` along the channel dimension, as
    produced by the standard two-stream I3D extraction pipeline.
    """
    bridge.to(device)
    optimizer = optim.Adam(bridge.parameters(), lr=lr)
    criterion = nn.MSELoss()
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        epoch_loss, total = 0.0, 0
        while batch_gen.has_next():
            batch_input, _, _, _ = batch_gen.next_batch(batch_size, False)
            batch_input = batch_input.to(device)
            batch_rgb = batch_input[:, :1024, :]
            batch_flow = batch_input[:, 1024:, :]

            optimizer.zero_grad()
            pred_flow = bridge(batch_rgb)
            loss = criterion(pred_flow, batch_flow)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total += batch_input.size(0)

        print(f"[modality bridge] epoch {epoch + 1}/{num_epochs} loss={epoch_loss / max(total, 1):.4f}")
        batch_gen.reset()

    save_path = os.path.join(save_dir, f"modality_bridge_{num_epochs}.pt")
    torch.save(bridge.state_dict(), save_path)
    print(f"[modality bridge] saved to {save_path}")
    return save_path
