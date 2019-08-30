import numpy as np
import torch


class FFTLoss(torch.nn.Module):
    def __init__(self, signal_ndim, normalized=False):
        """
        Loss described in section 3.3.2 of the Report.

        Parameters
        ----------
        signal_ndim: integer
        Perform 1d or 2d FFT.

        Methods
        -------
        forward : Compute FFT of the signals, compare them with MSE, sum the results.
        """
        super().__init__()

        self.signal_ndim = signal_ndim
        self.normalized = normalized

    def forward(self, output, target):

        output = torch.rfft(output, self.signal_ndim, normalized = self.normalized)
        target = torch.rfft(target, self.signal_ndim, normalized = self.normalized)

        losses = []
        for i in range(output.shape[2]):
            losses.append(torch.nn.MSELoss()(output[:, 0, i, :], target[:, 0, i, :]))

        totalLoss = sum(losses)

        return totalLoss
