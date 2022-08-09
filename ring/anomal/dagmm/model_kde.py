import numpy as np
from ring.common.dataset import TimeSeriesDataset
import torch.nn.functional as F
import torch
from torch import nn
from typing import List, Dict, Tuple, Union
from copy import deepcopy
from ring.common.base_model import BaseAnormal

HIDDENSTATE = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


class dagmm(BaseAnormal):
    def __init__(
        self,
        name: str = "dagmm",
        cell_type: str = "LSTM",
        hidden_size: int = 5,
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        n_layers: int = 1,
        dropout: float = 0,
        targets: List[str] = [],
        encoder_cont: List[str] = [],
        encoder_cat: List[str] = [],
        target_lags: Dict = {},
        cov: torch.tensor = None,
        output_size: int = 1,
        eps=torch.tensor(1e-8),
        return_enc: bool = True,
        encoderdecodertype: str = "RNN",
        steps=1,
        k_clusters: int = None,
    ):
        self.hidden_size = min(5 + int(output_size / 20), hidden_size)
        super().__init__(
            name=name,
            embedding_sizes=embedding_sizes,
            target_lags=target_lags,
            targets=targets,
            encoder_cont=encoder_cont,
            encoder_cat=encoder_cat,
            cell_type=cell_type,
            hidden_size=self.hidden_size,
            n_layers=n_layers,
            dropout=dropout,
            return_enc=return_enc,
            encoderdecodertype=encoderdecodertype,
            steps=steps,
        )

        self.cov = cov
        self.eps = eps

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataset, **kwargs):
        desired_embedding_sizes = kwargs.pop("embedding_sizes", {})
        embedding_sizes = {}
        for k, v in dataset.embedding_sizes.items():
            if k in dataset.encoder_cat:
                embedding_sizes[k] = v
        for name, size in desired_embedding_sizes.items():
            cat_size, _ = embedding_sizes[name]
            embedding_sizes[name] = (cat_size, size)
        return cls(
            targets=dataset.targets,
            encoder_cat=dataset.encoder_cat,
            encoder_cont=dataset.encoder_cont,
            embedding_sizes=embedding_sizes,
            steps=dataset.get_parameters().get("indexer").get("params").get("steps"),
            **kwargs,
        )

    def forward(self, x: Dict[str, torch.Tensor], mode=None, **kwargs) -> Dict[str, torch.Tensor]:
        # low-projection and hidden
        enc_output, dec = self.encoderdecoder(x)
        if isinstance(dec, tuple):
            kl_loss = dec[1]
            dec = dec[0]
        batch_size = enc_output.shape[0]

        if mode != "predict":
            self.compute_kde_params(enc_output[:, -1], batch_size)
            if self.training:
                sample_energy, cov_diag = self.compute_kde_energy(enc_output[:, -1], batch_size)
                return (sample_energy, cov_diag), dec[:, :, self.encoder_positions]  # for loss
            else:
                return [
                    [self.dataset, self.cov, self.log_det],
                    dec[:, :, self.encoder_positions],
                ]  # for parameters
        else:
            return enc_output[:, -1], dec[:, :, self.encoder_positions]  # for socres and reconstruction

    def compute_aux(self, C: torch.tensor):
        # setup auxilary variables for computing the sample energy
        C_sta = C + torch.eye(C.shape[1]).to(C.device) * self.eps

        L, self.V = torch.linalg.eigh(C_sta)  # decompos0-ition
        idx = torch.isclose(L, torch.tensor(float(0)))
        L_inv = 1 / L
        L_inv[idx] = 0  # force the negative to zero
        self.L = L  # eigenvalue
        self.L_inv = L_inv  # eigenvalue of inverse matric

    def compute_kde_params(self, z: torch.tensor = None, batch_size: int = None):
        self._neff = torch.tensor(1 / batch_size).to(z.device)  # assume eaqul weight
        # assume use `scott` to estimate bandwidth
        scott_factor = torch.pow(self._neff, -1.0 / (self.hidden_size + 2 + 4))
        self.cov = torch.cov(z.T) * scott_factor ** 2  # assume eaqul weight and no bias
        self.compute_aux(self.cov * 2 * np.pi)  # calculate eigenvalue of cov and inverse cov
        self.log_det = 2 * torch.log(self.L).sum()
        self.dataset = z

    def compute_kde_energy(self, z, batch_size, dataset=None, cov=None, log_det=None, size_average=True):
        if self.cov is None or self.dataset is None or self.log_det is None:
            self.cov = torch.tensor(cov).to(z.device)
            self.dataset = torch.tensor(dataset).to(z.device)
            self.log_det = torch.tensor(log_det).to(z.device)
        # batch_size_before
        n = self.dataset.shape[0]
        if not hasattr(self, "L"):
            self.compute_aux(self.cov * 2 * np.pi)
        # (hidden+2) *(hidden+2)
        self.inv_cov = (self.V * self.L_inv) @ self.V.T
        if batch_size >= n:
            # batch_size_now * batch_size_before * (hidden+2)
            diff = torch.stack([self.dataset[i] - z for i in range(batch_size)], dim=1)
        else:
            diff = torch.stack([self.dataset - z[i] for i in range(batch_size)], dim=0)
        # batch_size_now * batch_size_before * (hidden+2)
        tdiff = torch.matmul(diff, self.inv_cov)
        # batch_size_now * batch_size_before / batch_size_before * batch_size_now
        energy = torch.stack([torch.sum(diff[i] * tdiff[i], dim=1) for i in range(batch_size)], dim=1)
        log_to_sum = -self.log_det - energy
        cov_diag = torch.sum(torch.diagonal(1 / torch.max(self.eps, self.cov * 2 * np.pi)))
        sample_energy = self.logsumexp(log_to_sum, dim=1)
        sample_energy = -1 * torch.mean(log_to_sum)
        if not size_average:
            sample_energy = sample_energy.reshape(1)
        return sample_energy, cov_diag

    def logsumexp(self, a, dim=None):
        a_max = torch.amax(a, dim=dim, keepdims=True)
        if a_max.ndim > 0:
            a_max[~torch.isfinite(a_max)] = 0
        elif not torch.isfinite(a_max):
            a_max = 0
        tmp = torch.exp(a - a_max)
        s = torch.sum(tmp, dim=dim)
        out = torch.log(s) + a_max.squeeze(dim)
        return out

    def calculate_params(self, dataset, cov, log_det, num_samples):
        """
        calculate specific post-training parameters in model
        """
        train_cov = cov / num_samples
        train_log_det = log_det / num_samples
        if not train_log_det.shape:
            train_log_det = train_log_det.reshape(1)
        return {
            "cov": train_cov.data.cpu().numpy(),
            "dataset": dataset.cpu().numpy(),
            "log_det": train_log_det.cpu().numpy(),
        }

    def predict(self, output: tuple = None, **kwargs):
        sample_energy, _ = self.compute_kde_energy(
            z=output[0],
            batch_size=output[0].shape[0],
            dataset=kwargs["dataset"],
            cov=kwargs["cov"],
            log_det=kwargs["log_det"],
            size_average=False,
        )
        return sample_energy.data.cpu().numpy(), output[1]
