"""
Optimizer module
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections import defaultdict

import jax.numpy as jnp
import numpy as np
import polars as pl
from jax import Array
from tqdm.auto import tqdm

import pompon
from pompon import DTYPE
from pompon.layers.parameters import Parameter

logger = logging.getLogger("pompon").getChild("optimizer")
logger.setLevel(logging.WARNING)
logger.propagate = False


class Optimizer:
    """
    Base class for optimizers

    """

    def __init__(self) -> None:
        self.model: pompon.model.Model
        self.x_train: Array
        self.y_train: Array
        self.x_test: Array | None
        self.y_test: Array | None
        self.f_train: Array | None
        self.f_test: Array | None
        self.jobname: str | None
        self.data_loader: pompon.DataLoader
        self.epoch: int = 0
        self.trace: dict[str, list[float | list[int]]] = defaultdict(list)

    def setup(
        self,
        model: pompon.model.Model,
        x_train: Array | np.ndarray,
        y_train: Array | np.ndarray,
        *,
        batch_size: int = 100,
        shuffle: bool = True,
        x_test: Array | np.ndarray | None = None,
        y_test: Array | np.ndarray | None = None,
        f_train: Array | np.ndarray | None = None,
        f_test: Array | np.ndarray | None = None,
        jobname: str | None = None,
        outdir: str = ".",
    ) -> Optimizer:
        """
        Args:
           model (pompon.model.Model): the model to be optimized
           x_train (Array): the training data
           y_train (Array): the training target
           batch_size (int, optional): the batch size for stochastic method.
                                       Defaults to 100.
           shuffle (bool, optional): whether to shuffle the data.
                                     Defaults to True. When batch_size is large,
                                        it is recommended to set shuffle=False.
           x_test (Array, optional): the test data. Defaults to None.
           y_test (Array, optional): the test target. Defaults to None.
           f_train (Array, optional): the force data. Defaults to None.
           f_test (Array, optional): the force data for test.
                Defaults to None.
                Currently, test MSE is evaluated by only the energy term.
           jobname (str, optional): the name of the job. Defaults to None.
           outdir (str, optional): the output directory. Defaults to ".".

        Returns:
           Optimizer: the optimizer defined with the model and data.

        """
        self.model = model
        self.x_train = jnp.array(x_train, dtype=DTYPE)
        self.y_train = jnp.array(y_train, dtype=DTYPE)
        if x_test is not None and y_test is not None:
            self.x_test = jnp.array(x_test, dtype=DTYPE)
            self.y_test = jnp.array(y_test, dtype=DTYPE)
        else:
            self.x_test = None
            self.y_test = None
        if f_train is not None:
            self.f_train = jnp.array(f_train, dtype=DTYPE)
        else:
            self.f_train = None
        if f_test is not None:
            self.f_test = jnp.array(f_test, dtype=DTYPE)
        else:
            self.f_test = None
        self.jobname = jobname
        self.data_loader = pompon.DataLoader(
            arrays=(
                self.x_train,
                self.y_train,
                self.f_train,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
        )
        self.outdir = outdir
        self.model.export_h5(f"{self.outdir}/model_initial.h5")
        file_handler = logging.FileHandler(f"{self.outdir}/log.txt", mode="w")
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.info(
            f"Initialized optimizer: pompon version {pompon.__version__}"
        )

        return self

    def optimize(
        self,
        *,
        epochs: int = 1_000,
        epoch_per_trace: int = 1,
        epoch_per_log: int = 500,
        epoch_per_save: int = 100_000_000,
        fix_coord: bool = False,
        fix_basis: bool = False,
        wf: float = 1.0,
    ) -> pl.DataFrame:
        """

        Optimize the parameters of the model

        Args:
           epochs (int): the number of epochs executed in this execution.
           epoch_per_trace (int, optional): the number of epochs
                                            per recording the optimization trace.
                                            Defaults to 1.
           epoch_per_log (int, optional): the number of epochs per logging.
                                          Defaults to 500.
           epoch_per_save (int, optional): the number of epochs per saving the model.
                                           Defaults to 100_000_000.
           fix_coord (bool, optional): whether to fix the coordinator or not.
                                       Defaults to False.
           fix_basis (bool, optional): whether to fix the basis or not.
                                       Defaults to False.
           wf (float, optional): the weight factor for the force.

        Returns:
            pl.DataFrame: the optimization trace with columns
                          ``['epoch', 'mse_train', 'mse_test', 'tt_norm', 'tt_ranks']``.

        """  # noqa: E501
        lambda1: float = 0.0  # <- EXPERIMENTAL
        mu1: float = 1.0  # <- EXPERIMENTAL
        mu2: float = 1.0  # <- EXPERIMENTAL
        if epoch_per_log < epoch_per_trace:
            logger.warning(
                f"epochs_per_log={epoch_per_log} "
                + f"< epochs_per_trace={epoch_per_trace}"
            )
        n_iter = epochs + self.epoch
        progress_bar = tqdm(range(epochs))
        for _ in progress_bar:
            self.epoch += 1
            for arrays in self.data_loader:
                x_batch, y_batch = arrays[:2]
                if self.f_train is not None:
                    f_batch = arrays[2]
                else:
                    f_batch = None
                params = self.model.grad(
                    x=x_batch,
                    y=y_batch,
                    basis_grad=(not fix_basis),
                    coordinator_grad=(not fix_coord),
                    lambda1=lambda1,
                    mu1=mu1,
                    mu2=mu2,
                    f=f_batch,
                    wf=wf,
                )
                self.update(params)
            if self.epoch % epoch_per_trace == 0 or self.epoch == n_iter - 1:
                self._add_data_to_trace()
                if (
                    isinstance(mse := self.trace["mse_train"][-1], float)
                    and mse > 1.0e10
                ):
                    logger.info(
                        f"Diverged at epoch={self.epoch} "
                        + f"with mse_train={self.trace['mse_train'][-1]:.5e}"
                    )
                    logger.info("Stopped optimization")
                    break
            if self.epoch % epoch_per_log == 0:
                message = {}
                message["epoch"] = f"{self.trace['epoch'][-1]} / {n_iter}"
                message["Ener MSE (train, test)"] = f"({self.trace['mse_train'][-1]:.3e}, "+ f"{self.trace['mse_test'][-1]:.3e})"
                if "mse_train_f" in self.trace:
                    message["Force MSE:(train,)"] = f"{self.trace['mse_train_f'][-1]:.3e}"
                # if "entropy" in self.trace:
                #     message += f"entropy={self.trace['entropy'][-1]:.3e}, "
                # if "L1_norm" in self.trace:
                #     message += f"L1_norm={self.trace['L1_norm'][-1]:.3e}, "
                if "tt_ranks" in self.trace:
                    message["Ranks"] = f"{self.trace['tt_ranks'][-1]}"
                #logger.info(message)
                progress_bar.set_postfix(message)
                self.logging_mse()
            if self.epoch % epoch_per_save == 0:
                self.model.export_h5(
                    f"{self.outdir}/model_epoch_{self.epoch}.h5"
                )
        return self.get_trace()

    def update(self, params: list[Parameter]) -> None:
        """update whole parameters one step"""
        for param in params:
            if param.grad is not None:
                self.update_one(param)

    @abstractmethod
    def update_one(self, param: Parameter) -> None:
        """update one parameter"""
        raise NotImplementedError

    def get_trace(self) -> pl.DataFrame:
        """
        Get the optimization trace

        Returns:
           pl.DataFrame: the optimization trace with columns
              ``['epoch', 'mse_train', 'mse_test', 'tt_norm', 'tt_ranks']``.

        """
        df = pl.DataFrame(data=self.trace)
        if self.jobname is not None:
            df.write_parquet(f"{self.outdir}/{self.jobname}_trace.parquet")
        else:
            df.write_parquet(f"{self.outdir}/trace.parquet")
        return df

    def logging_mse(self) -> None:
        mse_train = self.model.mse(self.x_train, self.y_train)
        message = f"epochs: {self.epoch}, "
        if self.has_test_data:
            mse_test = self.model.mse(self.x_test, self.y_test)  # type: ignore[arg-type]
            message += (
                f"Ener MSE:(train, test)=({mse_train:.3e}, {mse_test:.3e})"
            )
        else:
            message = f"Ener MSE:(train,)=({mse_train:.3e},)"
        if self.f_train is not None:
            mse_train_f = self.model.mse_force(self.x_train, self.f_train)
            message += f" Force MSE:(train,)={mse_train_f:.3e}"
        if hasattr(self.model, "tt"):
            message += f" ranks={self.model.tt.ranks}"
        logger.info(message)

    def logging_entropy(self) -> None:
        entropy = self.model.basis_entropy(self.x_train)
        logger.info(f"\tEntropy={entropy}")

    def logging_L1_norm(self) -> None:
        L1_norm = self.model.basis_L1_norm(self.x_train)
        logger.info(f"\tL1_norm={L1_norm}")

    def _add_data_to_trace(self) -> None:
        self.trace["epoch"].append(self.epoch)
        self.trace["mse_train"].append(
            self.model.mse(self.x_train, self.y_train)
        )
        if self.has_test_data:
            self.trace["mse_test"].append(
                self.model.mse(self.x_test, self.y_test)  # type: ignore[arg-type]
            )
        if self.f_train is not None:
            self.trace["mse_train_f"].append(
                self.model.mse_force(self.x_train, self.f_train)
            )
        # self.trace["L1_norm"].append(self.model.basis_L1_norm(self.x_train))
        # self.trace["entropy"].append(self.model.basis_entropy(self.x_train))
        if hasattr(self.model, "tt"):
            self.trace["tt_norm"].append(float(self.model.tt.norm.data))
            self.trace["tt_ranks"].append(self.model.tt.ranks)

    @property
    def has_test_data(self) -> bool:
        return self.x_test is not None and self.y_test is not None
