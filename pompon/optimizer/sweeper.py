import logging
import math
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import polars as pl
from jax import Array
from jax.scipy.optimize import minimize as jax_minimize
from scipy.optimize import minimize as scipy_minimize  # type: ignore
from tqdm.auto import tqdm

import pompon.model
from pompon import DTYPE
from pompon._jittables import (
    _grad_block_and_basis2y_onedot,
    _grad_block_and_basis2y_twodot,
    _mse_block_and_basis2y_onedot,
    _mse_block_and_basis2y_twodot,
)
from pompon.layers.tensor import Core, TwodotCore
from pompon.layers.tt import TensorTrain
from pompon.optimizer import Optimizer
from pompon.optimizer.lin_reg import (
    conjugate_gradient_onedot,
    conjugate_gradient_twodot,
)

logger = logging.getLogger("pompon").getChild("optimizer")
logger.setLevel(logging.DEBUG)


class Sweeper:
    """Sweep optimizer for tensor-train"""

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer
        assert isinstance(optimizer.model, pompon.model.NNMPO)
        self.tt = optimizer.model.tt

    def sweep(
        self,
        *,
        nsweeps: int = 2,
        maxdim: int | list[int] | np.ndarray = 30,
        cutoff: float | list[float] | np.ndarray = 1e-2,
        optax_solver: optax.GradientTransformation | None = None,
        opt_maxiter: int = 1_000,
        opt_tol: float | list[float] | np.ndarray | None = None,
        opt_batchsize: int = 10_000,
        opt_lambda: float = 0.0,
        onedot: bool = False,
        use_CG: bool = False,
        use_scipy: bool = False,
        use_jax_scipy: bool = False,
        method: str = "L-BFGS-B",
        wf: float = 1.0,
        ord: str = "fro",
        auto_onedot: bool = True,
    ) -> pl.DataFrame:
        r"""
        TT-sweep optimization

        Args:
           nsweeps (int): The number of sweeps.
           maxdim (int, list[int]): the maximum rank of TT-sweep.
           cutoff (float, list[float]): the ratio of truncated singular values for TT-sweep.
                When one-dot core is optimized, this parameter is not used.
           optax_solver (optax.GradientTransformation): the optimizer for TT-sweep.
                Defaults to None. If None, the optimizer is not used.
           opt_maxiter (int): the maximum number of iterations for TT-sweep.
           opt_tol (float, list[float]): the convergence criterion of gradient for TT-sweep.
                Defaults to None, i.e., opt_tol = cutoff.
           opt_batchsize (int): the size of mini-batch for TT-sweep.
           opt_lambda (float): the L2 regularization parameter for TT-sweep.
                Only use_CG=True is supported.
           onedot (bool, optional): whether to optimize one-dot or two-dot core.
                Defaults to False, i.e. two-dot core optimization.
           use_CG (bool, optional): whether to use conjugate gradient method for TT-sweep.
                Defaults to False. CG is suitable for one-dot core optimization.
           use_scipy (bool, optional): whether to use scipy.optimize.minimize for TT-sweep.
                Defaults to False and use L-BFGS-B method. GPU is not supported.
           use_jax_scipy (bool, optional): whether to use jax.scipy.optimize.minimize for TT-sweep.
                Defaults to False. This optimizer is only supports BFGS method, which exhausts GPU memory.
           method (str, optional): the optimization method for scipy.optimize.minimize.
                Defaults to 'L-BFGS-B'. Note that jax.scipy.optimize.minimize only supports 'BFGS'.
           wf (float, optional): the weight factor of force $w_f$ in the loss function.
           ord (str, optional): the norm for scaling the initial core. Defaults to 'fro'.
                'max`, maximum absolute value, 'fro', Frobenius norm, are supported.
           auto_onedot (bool, optional): whether to switch to one-dot core optimization automatically once
                the maximum rank is reached. Defaults to True.
                This will cause overfitting in the beginning of the optimization.

        Returns:
            pl.DataFrame: the optimization trace with columns
                          ``['epoch', 'mse_train', 'mse_test', 'tt_norm', 'tt_ranks']``.

        :::{.callout .note}
        We recommend to use `optax_solver` for initial optimization and
        `use_CG=True` for the last fine-tuning.
        :::


        Two-dot optimization algorithm

        1. Construct original two-dot tensor $B\substack{i_p i_{p+1}\\\beta_{p-1} \beta_{p+1}}$

        $$
           B\substack{i_p i_{p+1}\\\beta_{p-1} \beta_{p+1}}
           = \sum_{\beta_p} W^{[p]}_{\beta_{p-1} i_p \beta_p} W^{[p+1]}_{\beta_p i_{p+1} \beta_{p+1}}
        $$

        2. Shift two-dot tensor $B\substack{i_p i_{p+1}\\\beta_{p-1} \beta_{p+1}}$
            by $\Delta B\substack{i_p i_{p+1}\\\beta_{p-1} \beta_{p+1}}$

        $$
           B\substack{i_p i_{p+1}\\\beta_{p-1} \beta_{p+1}}
           \leftarrow
           B\substack{i_p i_{p+1}\\\beta_{p-1} \beta_{p+1}}
           + \Delta B\substack{i_p i_{p+1}\\\beta_{p-1} \beta_{p+1}}
        $$

        3. Execute singular value decomposition (truncate small singular values as needed)

        $$
           B\substack{i_p i_{p+1}\\ \beta_{p-1} \beta_{p+1}}
           = \sum_{\beta_p,\beta_p^\prime}^{M^\prime}
           U\substack{i_p\\ \beta_{p-1}\beta_p}
           S\substack{\beta_p\enspace \\ \enspace\beta_p^\prime}
           V\substack{i_{p+1}\\ \beta_p^\prime \beta_{p+1}}
           \simeq
           \sum_{\beta_p,\beta_p^\prime}^{M}
           U\substack{i_p\\ \beta_{p-1}\beta_p}
           S\substack{\beta_p\enspace \\ \enspace\beta_p^\prime}
           V\substack{i_{p+1}\\ \beta_p^\prime \beta_{p+1}} \quad (M^\prime \le M)
        $$

        4. Update parameters

        $$
           W^{[p]}_{\beta_{p-1} i_p \beta_p} \leftarrow
           U\substack{i_p\\ \beta_{p-1}\beta_p}
        $$
        $$
           W^{[p+1]}_{\beta_p i_{p+1} \beta_{p+1}} \leftarrow
           \sum_{\beta_p^\prime}
           S\substack{\beta_p\enspace \\ \enspace\beta_p^\prime}
           V\substack{i_{p+1}\\ \beta_p^\prime \beta_{p+1}}
        $$

        5. Shift the center site to the left or right (sweeping)

        """  # noqa: E501
        dataloader = pompon.DataLoader(
            arrays=(
                self.optimizer.x_train,
                self.optimizer.y_train,
                self.optimizer.f_train,
            ),
            batch_size=opt_batchsize,
            shuffle=False,
        )
        q0 = self.optimizer.model.coordinator.forward(
            self.optimizer.model.x0.data
        )
        sqrt_wf = math.sqrt(wf)
        for i_sweep, batch in tqdm(enumerate(dataloader)):
            x_train, y_train = batch[:2]
            q = self.optimizer.model.coordinator.forward(x_train)
            basis: list[Array] = self.optimizer.model.basis.forward(q, q0)
            if len(batch) == 3:
                f_train = batch[2]
                U = self.optimizer.model.coordinator.U.data
                fU = -sqrt_wf * f_train @ U
                y_concat = y_train.copy()  # (D, 1)
                basis_concat = [phi.copy() for phi in basis]
                assert id(basis_concat[0]) != id(basis[0])
                # list[Array] with shape [(D, N) x f]
                partial_basis = self.optimizer.model.basis.partial(q, q0)
                # basis_concat =
                # [[ φ1,  φ2, ...,  φf],
                #  [dφ1,  φ2, ...,  φf],
                #  [ φ1, dφ2, ...,  φf],
                #  ...
                #  [ φ1, dφ2, ...,  dφf]]
                for i in range(len(basis)):
                    i_basis = [phi.copy() for phi in basis]
                    # list[Array] with shape [(D, N) x f]
                    i_basis[i] = sqrt_wf * partial_basis[i]
                    # Concatenate like
                    # [(D, N) x f] + [(D, N) x f] = [(2D, N) x f]
                    for j in range(len(basis)):
                        basis_concat[j] = jnp.vstack(
                            [basis_concat[j], i_basis[j]]
                        )
                    y_concat = jnp.vstack([y_concat, fU[:, i][:, jnp.newaxis]])
                basis = basis_concat
                y = y_concat
                opt_batchsize = opt_batchsize * (
                    len(basis) + 1
                )  # because y = ((f+1)D, 1)
            else:
                y = y_train

            sweep(
                tt=self.tt,
                basis=basis,
                y=y,
                nsweeps=nsweeps,
                maxdim=maxdim,
                cutoff=cutoff,
                optax_solver=optax_solver,
                opt_maxiter=opt_maxiter,
                opt_tol=opt_tol,
                opt_lambda=opt_lambda,
                onedot=onedot,
                use_CG=use_CG,
                use_scipy=use_scipy,
                use_jax_scipy=use_jax_scipy,
                method=method,
                ord=ord,
                auto_onedot=auto_onedot,
            )
            # self.optimizer.logging_mse()
        return self.optimizer.get_trace()


def fill_list(
    nsweeps: int,
    scalar_or_list: int | float | list[int] | list[float] | np.ndarray,
):
    if isinstance(scalar_or_list, np.ndarray):
        scalar_or_list = scalar_or_list.tolist()
    if isinstance(scalar_or_list, int | float):
        return [scalar_or_list] * nsweeps
    elif isinstance(scalar_or_list, list):
        if len(scalar_or_list) == nsweeps:
            return scalar_or_list
        elif len(scalar_or_list) > nsweeps:
            return scalar_or_list[:nsweeps]
        else:
            return scalar_or_list + [scalar_or_list[-1]] * (
                nsweeps - len(scalar_or_list)
            )
    else:
        raise ValueError(f"Invalid input {scalar_or_list=}")


def show_sweep_info(
    nsweeps: int,
    maxdim_list: list[int],
    cutoff_list: list[float],
    opt_maxiter_list: list[int],
    opt_tol_list: list[float],
) -> None:
    """
    Show like this

    | sweep | maxdim | cutoff | opt_maxiter | opt_tol |
    |-------|--------|--------|-------------|---------|
    | 1     | 30     | 1e-2   | 10000       | 1e-2    |
    | 2     | 30     | 1e-2   | 10000       | 1e-2    |

    """
    sweep_info = pl.DataFrame(
        {
            "sweep": range(1, nsweeps + 1),
            "maxdim": maxdim_list,
            "cutoff": cutoff_list,
            "opt_maxiter": opt_maxiter_list,
            "opt_tol": opt_tol_list,
        }
    )
    logger.info("BEGIN TT-sweep " + f"\n{sweep_info}")


def sweep(
    *,
    tt: TensorTrain,
    basis: list[Array],
    y: Array,
    nsweeps: int = 2,
    maxdim: int | list[int] | np.ndarray = 30,
    cutoff: float | list[float] | np.ndarray = 1e-2,
    optax_solver: optax.GradientTransformation | None = None,
    opt_maxiter: int = 1_000,
    opt_tol: float | list[float] | np.ndarray | None = None,
    opt_lambda: float = 0.0,
    onedot: bool = False,
    use_CG: bool = False,
    use_scipy: bool = False,
    use_jax_scipy: bool = False,
    method: str = "L-BFGS-B",
    ord: str = "fro",
    auto_onedot: bool = True,
):
    """
    Tensor-train sweep optimization

    Args:
        tt (TensorTrain): the tensor-train model.
        basis (list[Array]): the basis functions.
        y (Array): the target values.
        nsweeps (int): The number of sweeps.
        maxdim (int, list[int]): the maximum rank of TT-sweep.
        cutoff (float, list[float]): the ratio of truncated singular values for TT-sweep.
            When one-dot core is optimized, this parameter is not used.
        optax_solver (optax.GradientTransformation): the optimizer for TT-sweep.
            Defaults to None. If None, the optimizer is not used.
        opt_maxiter (int): the maximum number of iterations for TT-sweep.
        opt_tol (float, list[float]): the convergence criterion of gradient for TT-sweep.
            Defaults to None, i.e., opt_tol = cutoff.
        opt_lambda (float): the L2 regularization parameter for TT-sweep.
            Only use_CG=True is supported.
        onedot (bool, optional): whether to optimize one-dot or two-dot core.
            Defaults to False, i.e. two-dot core optimization.
        use_CG (bool, optional): whether to use conjugate gradient method for TT-sweep.
            Defaults to False. CG is suitable for one-dot core optimization.
        use_scipy (bool, optional): whether to use scipy.optimize.minimize for TT-sweep.
            Defaults to False and use L-BFGS-B method. GPU is not supported.
        use_jax_scipy (bool, optional): whether to use jax.scipy.optimize.minimize for TT-sweep.
            Defaults to False. This optimizer is only supports BFGS method, which exhausts GPU memory.
        method (str, optional): the optimization method for scipy.optimize.minimize.
            Defaults to 'L-BFGS-B'. Note that jax.scipy.optimize.minimize only supports 'BFGS'.
        ord (str, optional): the norm for scaling the initial core.
            Defaults to 'fro', Frobenuis norm.
            'max`, maximum absolute value, 'fro', Frobenius norm, are supported.
        auto_onedot (bool, optional): whether to switch to one-dot core optimization automatically once
            the maximum rank is reached. Defaults to True.
            This will cause overfitting in the beginning of the optimization.
    """  # noqa: E501
    maxdim_list = fill_list(nsweeps, maxdim)
    cutoff_list = fill_list(nsweeps, cutoff)
    opt_maxiter_list = fill_list(nsweeps, opt_maxiter)
    if opt_tol is None:
        opt_tol_list = cutoff_list
    else:
        opt_tol_list = fill_list(nsweeps, opt_tol)
    assert len(maxdim_list) == len(cutoff_list) == len(opt_tol_list) == nsweeps
    show_sweep_info(
        nsweeps=nsweeps,
        maxdim_list=maxdim_list,
        cutoff_list=cutoff_list,
        opt_maxiter_list=opt_maxiter_list,
        opt_tol_list=opt_tol_list,
    )
    tt.set_blocks_batch(basis=basis)
    progress_bar = tqdm(
        zip(
            maxdim_list,
            cutoff_list,
            opt_tol_list,
            opt_maxiter_list,
            strict=True,
        ),
        total=nsweeps
    )
    i_sweep = 0
    for maxdim, cutoff, opt_tol, opt_maxiter in progress_bar:
        assert isinstance(maxdim, int)
        assert isinstance(cutoff, float)
        assert isinstance(opt_tol, float)
        assert isinstance(opt_maxiter, int)
        terminal_site, to_right = _get_terminal_and_direction(tt, onedot)
        if auto_onedot and max(tt.ranks) >= maxdim and (not onedot):
            onedot = True
        ndot = 1 if onedot else 2
        progress_bar.set_postfix({'sweeps': f"{i_sweep + 1}/{nsweeps}", 'ndot': ndot, 'rank': max(tt.ranks)})
        i_sweep += 1
        while True:
            core = _get_scaled_initial_core(
                tt=tt, onedot=onedot, ord=ord, to_right=to_right
            )
            logger.debug(f"{tt.center=}")
            if (to_right and tt.center + ndot - 1 <= terminal_site) or (
                not to_right and tt.center - ndot + 1 >= terminal_site
            ):
                logger.debug(
                    f"{tt.center=}, {terminal_site=}, "
                    + f"{to_right=}, {onedot=}"
                )
                if onedot:
                    target_core_name = f"C({tt.center})"
                elif to_right:
                    target_core_name = f"B({tt.center},{tt.center + 1})"
                else:
                    target_core_name = f"B({tt.center-1},{tt.center})"
                logger.debug(f"\tBEGIN optimization of {target_core_name}")
                args = _get_opt_args(
                    tt=tt,
                    onedot=onedot,
                    basis=basis,
                    y=y,
                    to_right=to_right,
                )
                if use_CG:
                    x0 = core.data
                    core.data = _opt_CG(
                        param=x0,
                        args=args,
                        maxiter=opt_maxiter,
                        tol=opt_tol,
                        λ=opt_lambda,
                    )
                elif optax_solver is not None:
                    x0 = core.data
                    core.data = _opt_optax(
                        param=x0,
                        args=args,
                        solver=optax_solver,
                        opt_maxiter=opt_maxiter,
                        opt_tol=opt_tol,
                    )
                elif use_scipy | use_jax_scipy:
                    x0 = core.data.flatten()
                    if use_scipy:
                        x = _opt_scipy(
                            x0=x0,
                            args=args,
                            method=method,
                            opt_maxiter=opt_maxiter,
                            opt_tol=opt_tol,
                        )
                    elif use_jax_scipy:
                        x = _opt_jax_scipy(
                            x0=x0,
                            args=args,
                            method=method,
                            opt_maxiter=opt_maxiter,
                            opt_tol=opt_tol,
                        )
                    else:
                        raise ValueError(
                            "You need to specify either use_scipy "
                            + "or use_jax_scipy but got "
                            + f"{use_scipy=}, {use_jax_scipy=}"
                        )
                    core.data = x.reshape(core.data.shape)
                else:
                    raise NotImplementedError
                logger.debug(f"\tDONE optimization of {target_core_name}")
                if onedot:
                    tt.decompose_and_assign_center_onedot(
                        to_right=to_right, ord=ord
                    )
                else:
                    gauge = "LC" if to_right else "CR"
                    tt.decompose_and_assign_center_twodot(
                        to_right=to_right,
                        truncation=1.0 - cutoff,
                        rank=maxdim,
                        gauge=gauge,
                        ord=ord,
                    )
                logger.info(f"Site: {tt.center}\tCore shape: {core.shape}\tTT-MSE: {jnp.mean((tt(basis)-y)**2):.3e}")  # noqa: E501
                # self.optimizer.logging_mse()
            if tt.center == terminal_site:
                break
            else:
                tt.shift_center(
                    to_right=to_right, basis=basis, is_onedot_center=onedot
                )


def _opt_scipy(*, x0: Array, args, method, opt_maxiter, opt_tol) -> Array:
    x0_np = np.array(x0)
    if len(args) == 5:
        fun = _fun_scipy_onedot
        jac = _jac_scipy_onedot
    else:
        fun = _fun_scipy_twodot
        jac = _jac_scipy_twodot
    result = scipy_minimize(
        fun,
        x0_np,
        args=args,
        jac=jac,
        method=method,
        options={
            "maxiter": opt_maxiter,
            "disp": False,
            "gtol": opt_tol,
        },
    )
    logger.debug(f"\t{result.message=}, " + f"{result.fun=:.5e}, {result.nit=}")
    return jnp.array(result.x, dtype=DTYPE)


def _opt_jax_scipy(*, x0: Array, args, method, opt_maxiter, opt_tol) -> Array:
    if len(args) == 5:
        fun_jax = _fun_jax_scipy_onedot
    else:
        fun_jax = _fun_jax_scipy_twodot
    result = jax_minimize(
        fun_jax,
        x0,
        args=args,
        method=method,
        options={"maxiter": opt_maxiter, "gtol": opt_tol},
    )
    logger.debug(
        f"\t{result.fun=:.5e}, "
        + f"# of eval func={int(result.nfev)}, "
        + f"# of eval jac={int(result.njev)}, "
        + f"# of iter={int(result.nit)}"
    )
    return result.x


def _opt_CG(
    *,
    param: Array,
    args,
    maxiter: int,
    tol: float,
    λ: float,
) -> Array:
    if len(args) == 5:
        L, R, ϕ, norm, y = args
        cg_func = partial(
            conjugate_gradient_onedot,
            L_block=L * norm,
            R_block=R,
            phi=ϕ,
            y=y,
            tol=tol,
            maxiter=maxiter,
            lam=λ,
        )
    elif len(args) == 6:
        L, R, ϕ_L, ϕ_R, norm, y = args
        cg_func = partial(
            conjugate_gradient_twodot,
            L_block=L * norm,
            R_block=R,
            L_phi=ϕ_L,
            R_phi=ϕ_R,
            y=y,
            tol=tol,
            maxiter=maxiter,
            lam=λ,
        )
    else:
        raise ValueError(f"Invalid args length {len(args)=}")
    x = cg_func(param)
    return x


def _opt_optax(
    *,
    param: Array,
    args,
    solver: optax.GradientTransformation,
    opt_maxiter: int,
    opt_tol: float,
) -> Array:
    """
    Sweep for minibatch optimization with optax
    """
    opt_state = solver.init(param)
    if len(args) == 5:
        L, R, ϕ, norm, y = args
        value_and_grad_fn = partial(
            value_and_grad_onedot, L=L, R=R, ϕ=ϕ, norm=norm, y=y
        )
        value_fn = partial(value_onedot, L=L, R=R, ϕ=ϕ, norm=norm, y=y)
    else:
        L, R, ϕ_L, ϕ_R, norm, y = args
        value_and_grad_fn = partial(
            value_and_grad_twodot,
            L=L,
            R=R,
            ϕ_L=ϕ_L,
            ϕ_R=ϕ_R,
            norm=norm,
            y=y,
        )
        value_fn = partial(
            value_twodot,
            L=L,
            R=R,
            ϕ_L=ϕ_L,
            ϕ_R=ϕ_R,
            norm=norm,
            y=y,
        )

    @jax.jit
    def cond(args):
        i, grad, _, _ = args
        return jnp.logical_and(i < opt_maxiter, jnp.linalg.norm(grad) > opt_tol)

    def body(args):
        i, _, param, opt_state = args
        value, grad = value_and_grad_fn(param)  # in the case of line-search
        updates, opt_state = solver.update(
            grad,
            opt_state,
            param,
            value=value,
            grad=grad,
            value_fn=value_fn,
        )
        param = optax.apply_updates(param, updates)
        return i + 1, grad, param, opt_state

    _, _, param, _ = jax.lax.while_loop(
        cond, body, (0, jnp.ones_like(param) * jnp.inf, param, opt_state)
    )
    return param

    # for _ in range(opt_maxiter):
    #     value, grad = value_and_grad_fn(param)
    #     updates, opt_state = solver.update(
    #         grad,
    #         opt_state,
    #         param,
    #         value=value,
    #         grad=grad,
    #         value_fn=value_fn,
    #     )
    #     param = optax.apply_updates(param, updates)
    #     if jnp.linalg.norm(grad) < opt_tol:
    #         break
    # return param


def _get_terminal_and_direction(
    tt: TensorTrain, onedot: bool
) -> tuple[int, bool]:
    if (center := tt.center) == 0:
        to_right = True
        terminal_site = tt.ndim - 1
    elif center == tt.ndim - 1:
        to_right = False
        terminal_site = 0
    else:
        raise ValueError(
            "Invalid start point with " + f"center={tt.center}, onedot={onedot}"
        )
    return (terminal_site, to_right)


def _get_scaled_initial_core(
    tt: TensorTrain, onedot: bool, ord: str = "fro", to_right: bool = True
) -> TwodotCore | Core:
    if onedot:
        C = tt.C
        if C is None:
            raise ValueError("C is None")
        else:
            tt.norm.data *= C.scale_to(jnp.array(1.0, dtype=DTYPE), ord=ord)
            C.grad = None
            return C
    elif to_right:
        B = tt.B
        if B is None:
            raise ValueError("B is None")
        else:
            tt.norm.data *= B.scale_to(jnp.array(1.0, dtype=DTYPE), ord=ord)
            B.grad = None
            return B
    else:
        B = tt.B
        if B is None:
            raise ValueError("B is None")
        else:
            tt.norm.data *= B.scale_to(jnp.array(1.0, dtype=DTYPE), ord=ord)
            B.grad = None
            return B


def _get_opt_args(
    tt: TensorTrain,
    onedot: bool,
    basis: list[Array],
    y: Array,
    to_right: bool,
) -> tuple[Any, ...]:
    if onedot:
        left_block = tt.left_blocks_batch[-1].data
        right_block = tt.right_blocks_batch[-1].data
        center_basis = basis[tt.center]
        norm = tt.norm.data
        return (left_block, right_block, center_basis, norm, y)
    elif to_right:
        left_block = tt.left_blocks_batch[-1].data
        right_block = tt.right_blocks_batch[-2].data
        left_basis = basis[tt.center]
        right_basis = basis[tt.center + 1]
        norm = tt.norm.data
        return (left_block, right_block, left_basis, right_basis, norm, y)
    else:
        left_block = tt.left_blocks_batch[-2].data
        right_block = tt.right_blocks_batch[-1].data
        left_basis = basis[tt.center - 1]
        right_basis = basis[tt.center]
        norm = tt.norm.data
        return (left_block, right_block, left_basis, right_basis, norm, y)


def _fun_scipy_twodot(x: np.ndarray, *args) -> float:
    x_jnp = jnp.array(x, dtype=DTYPE)
    return float(_fun_jax_scipy_twodot(x_jnp, *args))


def _fun_scipy_onedot(x: np.ndarray, *args) -> float:
    x_jnp = jnp.array(x, dtype=DTYPE)
    return float(_fun_jax_scipy_onedot(x_jnp, *args))


@jax.jit
def _fun_jax_scipy_twodot(x: Array, *args) -> Array:
    """
    The objective function for `scipy.optimize.minimize<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_

    Args:
        x (Array): parameters, particularly B of TT-core
        *args: arguments

    Returns:
        float: objective function value
    """  # noqa: E501
    left_block, right_block, left_basis, right_basis, norm, y = args
    B = x.reshape(
        (
            left_block.shape[1],
            left_basis.shape[1],
            right_basis.shape[1],
            right_block.shape[1],
        )
    )
    return _mse_block_and_basis2y_twodot(
        center_twodot=B,
        y=y,
        left_phi_batch=left_basis,
        right_phi_batch=right_basis,
        left_block_batch=left_block,
        right_block_batch=right_block,
        norm=norm,
    )


@jax.jit
def _fun_jax_scipy_onedot(x: Array, *args) -> Array:
    left_block, right_block, center_basis, norm, y = args
    C = x.reshape(
        (left_block.shape[1], center_basis.shape[1], right_block.shape[1])
    )
    return _mse_block_and_basis2y_onedot(
        center_onedot=C,
        y=y,
        center_phi_batch=center_basis,
        left_block_batch=left_block,
        right_block_batch=right_block,
        norm=norm,
    )


def _jac_scipy_twodot(x: np.ndarray, *args) -> np.ndarray:
    """
    The Jacobian for `scipy.optimize.minimize<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_

    Args:
        x (np.ndarray): parameters
        *args: arguments

    Returns:
        np.ndarray: Jacobian (gradient)
    """  # noqa: E501
    left_block, right_block, left_basis, right_basis, norm, y = args
    x_jnp = jnp.array(x, dtype=DTYPE)
    B = x_jnp.reshape(
        (
            left_block.shape[1],
            left_basis.shape[1],
            right_basis.shape[1],
            right_block.shape[1],
        )
    )
    return np.array(
        _grad_block_and_basis2y_twodot(
            y=y,
            left_phi_batch=left_basis,
            right_phi_batch=right_basis,
            center_twodot=B,
            left_block_batch=left_block,
            right_block_batch=right_block,
            norm=norm,
        )
    ).flatten()


def _jac_scipy_onedot(x: np.ndarray, *args: Array) -> np.ndarray:
    left_block, right_block, center_basis, norm, y = args
    x_jnp: Array = jnp.array(x, dtype=DTYPE)
    C = x_jnp.reshape(
        (left_block.shape[1], center_basis.shape[1], right_block.shape[1])
    )
    return np.array(
        _grad_block_and_basis2y_onedot(
            y=y,
            center_phi_batch=center_basis,
            center_onedot=C,
            left_block_batch=left_block,
            right_block_batch=right_block,
            norm=norm,
        )
    ).flatten()


def value_twodot(B, L, R, ϕ_L, ϕ_R, norm, y):
    return _mse_block_and_basis2y_twodot(
        center_twodot=B,
        y=y,
        left_phi_batch=ϕ_L,
        right_phi_batch=ϕ_R,
        left_block_batch=L,
        right_block_batch=R,
        norm=norm,
    )


@jax.jit
def value_and_grad_twodot(B, L, R, ϕ_L, ϕ_R, norm, y):
    value = _mse_block_and_basis2y_twodot(
        center_twodot=B,
        y=y,
        left_phi_batch=ϕ_L,
        right_phi_batch=ϕ_R,
        left_block_batch=L,
        right_block_batch=R,
        norm=norm,
    )
    grad = _grad_block_and_basis2y_twodot(
        center_twodot=B,
        y=y,
        left_phi_batch=ϕ_L,
        right_phi_batch=ϕ_R,
        left_block_batch=L,
        right_block_batch=R,
        norm=norm,
    )
    return value, grad


def value_onedot(C, L, R, ϕ, norm, y):
    return _mse_block_and_basis2y_onedot(
        center_onedot=C,
        y=y,
        center_phi_batch=ϕ,
        left_block_batch=L,
        right_block_batch=R,
        norm=norm,
    )


@jax.jit
def value_and_grad_onedot(C, L, R, ϕ, norm, y):
    value = _mse_block_and_basis2y_onedot(
        center_onedot=C,
        y=y,
        center_phi_batch=ϕ,
        left_block_batch=L,
        right_block_batch=R,
        norm=norm,
    )
    grad = _grad_block_and_basis2y_onedot(
        center_onedot=C,
        y=y,
        center_phi_batch=ϕ,
        left_block_batch=L,
        right_block_batch=R,
        norm=norm,
    )
    return value, grad
