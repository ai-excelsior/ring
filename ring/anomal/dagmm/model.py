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
        k_clusters: int = 2,
        x_categoricals: List[str] = [],
        targets: List[str] = [],
        encoder_cont: List[str] = [],
        encoder_cat: List[str] = [],
        target_lags: Dict = {},
        phi: torch.tensor = None,
        mu: torch.tensor = None,
        cov: torch.tensor = None,
        output_size: int = 1,
        eps=torch.tensor(1e-10),
        return_enc: bool = True,
        encoderdecodertype: str = "RNN",
        steps=1,
    ):
        self.hidden_size = min(5 + int(output_size / 20), hidden_size)
        super().__init__(
            name=name,
            embedding_sizes=embedding_sizes,
            target_lags=target_lags,
            x_categoricals=x_categoricals,
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

        self.k_clusters = k_clusters
        self.mu = mu
        self.cov = cov
        self.phi = phi

        layers = [
            nn.Linear(self.hidden_size + 2, 10),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(10, k_clusters),
            nn.Softmax(dim=1),
        ]
        self.estimate = nn.Sequential(*layers)
        self.eps = eps

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataset, **kwargs):
        desired_embedding_sizes = kwargs.pop("embedding_sizes", {})
        embedding_sizes = deepcopy(dataset.embedding_sizes)
        for name, size in desired_embedding_sizes.items():
            cat_size, _ = embedding_sizes[name]
            embedding_sizes[name] = (cat_size, size)
        return cls(
            targets=dataset.targets,
            encoder_cat=dataset.encoder_cat,
            encoder_cont=dataset.encoder_cont,
            embedding_sizes=embedding_sizes,
            x_categoricals=dataset.categoricals,
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
        # reconstruction error
        rec_cosine = F.cosine_similarity(
            x["encoder_cont"].reshape(batch_size, -1),
            dec[:, :, self.target_positions].reshape(batch_size, -1),
            dim=1,
        )
        rec_euclidean = (
            x["encoder_cont"].reshape(batch_size, -1)
            - dec[:, :, self.target_positions].reshape(batch_size, -1)
        ).norm(2, dim=1) / torch.clamp(x["encoder_cont"].reshape(batch_size, -1).norm(2, dim=1), min=1e-10)
        # concat low-projection with reconstruction error
        z = torch.cat(
            [  # enc_output[:, -1],  # last position of `output` equals to last layer of `hidden_state`
                enc_output[:, -1],
                rec_euclidean.unsqueeze(-1),
                rec_cosine.unsqueeze(-1),
            ],
            dim=1,
        )  # batch_size * (self.hidden_size + 2)
        gamma = self.estimate(z)  # batch_size * k_clusters

        if mode != "predict":
            self.compute_gmm_params(z, gamma, batch_size)
            if self.training:
                sample_energy, cov_diag = self.compute_energy(z)
                return (sample_energy, cov_diag), dec[:, :, self.target_positions], False  # for loss
            else:
                return (gamma, self.mu, self.phi, self.cov), dec[
                    :, :, self.target_positions
                ]  # for parameters
        else:
            return z, dec[:, :, self.target_positions]  # for socres and reconstruction

    def compute_gmm_params(self, z: torch.tensor = None, gamma: torch.tensor = None, batch_size: int = None):
        # shape = k_clusters
        sum_gamma = torch.sum(gamma, dim=0)
        # shape = k_clusters
        self.phi = sum_gamma / batch_size
        # k_clusters x (self.hidden + 2)
        self.mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        # batch_size * k_clusters * (self.hidden + 2)
        z_mu = z.unsqueeze(1) - self.mu.unsqueeze(0)
        # batch_size * k_clusters * (self.hidden + 2) * (self.hidden + 2)
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        # k_clusters * (self.hidden + 2) * (self.hidden + 2)
        self.cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0) / sum_gamma.unsqueeze(
            -1
        ).unsqueeze(-1)
        self.compute_aux(self.cov)

    def compute_aux(self, C: torch.tensor):
        # setup auxilary variables for computing the sample energy
        C_sta = torch.stack(
            [C[i] + torch.eye(C[i].shape[1]).to(C.device) * self.eps for i in range(self.k_clusters)], dim=0
        )
        L, self.V = torch.linalg.eigh(C_sta)  # decompos0-ition
        idx = torch.isclose(L, torch.tensor(float(0)))
        L_inv = 1 / L
        L_inv[idx] = 0  # force the negative to zero
        self.L = L  # eigenvalue
        self.L_inv = L_inv  # eigenvalue of inverse matric

    def compute_energy(self, z: torch.tensor = None, size_average=True, phi=None, cov=None, mu=None):
        if self.cov is None or self.mu is None or self.phi is None:
            self.cov = torch.tensor(cov).to(z.device)
            self.phi = torch.tensor(phi).to(z.device)
            self.mu = torch.tensor(mu).to(z.device)

        if not hasattr(self, "L"):
            self.compute_aux(self.cov)

        # batch_size * k_clusters * (self.hidden + 2)
        z_mu = z.unsqueeze(1) - self.mu.unsqueeze(0)
        z_mu_ = torch.cat(
            [torch.matmul(z_mu[:, i, :], self.V[i]).unsqueeze(1) for i in range(self.k_clusters)], dim=1
        )
        # NOTE: this is easier and faster
        # k_clusters
        cov_exe = torch.concat(
            [
                torch.prod(self.L[i][~torch.isclose(self.L[i], torch.tensor(float(0)))]).unsqueeze(0)
                for i in range(self.k_clusters)
            ]
        )
        det_cov = torch.max(cov_exe * (2 * np.pi), self.eps)
        # batch_size * k_clusters
        exp_term_tmp = -0.5 * (self.L_inv * z_mu_).unsqueeze(-2) @ z_mu_.unsqueeze(-1)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]
        exp_term = torch.exp(exp_term_tmp - max_val)[..., 0, 0]
        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(self.phi * exp_term / torch.sqrt(det_cov), dim=1) + self.eps
        )
        cov_diag = torch.sum(torch.diagonal(1 / torch.max(self.eps, self.cov), dim1=1, dim2=2))
        # penalize item in case of singularity problem

        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag

    def calculate_params(self, gamma_sum, mu_sum, cov_sum, num_samples):
        """
        calculate specific post-training parameters in model
        """

        train_phi = gamma_sum / num_samples
        train_mu = mu_sum / (gamma_sum.unsqueeze(-1))
        train_cov = cov_sum / (gamma_sum.unsqueeze(-1).unsqueeze(-1))
        return {
            "phi": train_phi.data.cpu().numpy(),
            "mu": train_mu.data.cpu().numpy(),
            "cov": train_cov.data.cpu().numpy(),
        }

    def predict(self, output: tuple = None, **kwargs):
        sample_energy, _ = self.compute_energy(
            z=output[0],
            size_average=False,
            phi=kwargs["phi"],
            mu=kwargs["mu"],
            cov=kwargs["cov"],
        )
        return sample_energy.data.cpu().numpy(), output[1]
