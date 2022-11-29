# BSD 3-Clause License

# Copyright (c) 2007-2022 The scikit-learn developers.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numbers
import numpy as np
import pykokkos as pk
import warnings

from joblib import Parallel, effective_n_jobs
from scipy import optimize
from sklearn._loss.loss import HalfBinomialLoss, HalfMultinomialLoss
from sklearn.linear_model._base import LinearClassifierMixin, SparseCoefMixin, BaseEstimator
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.linear_model._sag import sag_solver
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.svm._base import _fit_liblinear
from sklearn.utils import check_random_state
from sklearn.utils import compute_class_weight
from sklearn.utils.extmath import row_norms
from sklearn.utils.extmath import softmax
from sklearn.utils.fixes import delayed
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.optimize import _newton_cg, _check_optimize_result
from sklearn.utils.validation import check_is_fitted, _check_sample_weight

_LOGISTIC_SOLVER_CONVERGENCE_MSG = (
    "Please also refer to the documentation for alternative solver options:\n"
    "    https://scikit-learn.org/stable/modules/linear_model.html"
    "#logistic-regression"
)

def asarray(arr, dtype=pk.double):
    arr = np.asarray(arr)

    view = pk.View(arr.shape, dtype)
    view[:] = arr
    return view


def _check_solver(solver, penalty, dual):
    all_solvers = ["liblinear", "newton-cg", "lbfgs", "sag", "saga"]
    if solver not in all_solvers:
        raise ValueError(
            "Logistic Regression supports only solvers in %s, got %s."
            % (all_solvers, solver)
        )

    all_penalties = ["l1", "l2", "elasticnet", "none"]
    if penalty not in all_penalties:
        raise ValueError(
            "Logistic Regression supports only penalties in %s, got %s."
            % (all_penalties, penalty)
        )

    if solver not in ["liblinear", "saga"] and penalty not in ("l2", "none"):
        raise ValueError(
            "Solver %s supports only 'l2' or 'none' penalties, got %s penalty."
            % (solver, penalty)
        )
    if solver != "liblinear" and dual:
        raise ValueError(
            "Solver %s supports only dual=False, got dual=%s" % (solver, dual)
        )

    if penalty == "elasticnet" and solver != "saga":
        raise ValueError(
            "Only 'saga' solver supports elasticnet penalty, got solver={}.".format(
                solver
            )
        )

    if solver == "liblinear" and penalty == "none":
        raise ValueError("penalty='none' is not supported for the liblinear solver")

    return solver


def _check_multi_class(multi_class, solver, n_classes):
    if multi_class == "auto":
        if solver == "liblinear":
            multi_class = "ovr"
        elif n_classes > 2:
            multi_class = "multinomial"
        else:
            multi_class = "ovr"
    if multi_class not in ("multinomial", "ovr"):
        raise ValueError(
            "multi_class should be 'multinomial', 'ovr' or 'auto'. Got %s."
            % multi_class
        )
    if multi_class == "multinomial" and solver == "liblinear":
        raise ValueError("Solver %s does not support a multinomial backend." % solver)
    return multi_class

def _logistic_regression_path(
    X,
    y,
    pos_class=None,
    Cs=10,
    fit_intercept=True,
    max_iter=100,
    tol=1e-4,
    verbose=0,
    solver="lbfgs",
    coef=None,
    class_weight=None,
    dual=False,
    penalty="l2",
    intercept_scaling=1.0,
    multi_class="auto",
    random_state=None,
    check_input=True,
    max_squared_sum=None,
    sample_weight=None,
    l1_ratio=None,
    n_threads=1,
):
    """Compute a Logistic Regression model for a list of regularization
    parameters.
    This is an implementation that uses the result of the previous model
    to speed up computations along the set of solutions, making it faster
    than sequentially calling LogisticRegression for the different parameters.
    Note that there will be no speedup with liblinear solver, since it does
    not handle warm-starting.
    Read more in the :ref:`User Guide <logistic_regression>`.
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Input data, target values.
    pos_class : int, default=None
        The class with respect to which we perform a one-vs-all fit.
        If None, then it is assumed that the given problem is binary.
    Cs : int or array-like of shape (n_cs,), default=10
        List of values for the regularization parameter or integer specifying
        the number of regularization parameters that should be used. In this
        case, the parameters will be chosen in a logarithmic scale between
        1e-4 and 1e4.
    fit_intercept : bool, default=True
        Whether to fit an intercept for the model. In this case the shape of
        the returned array is (n_cs, n_features + 1).
    max_iter : int, default=100
        Maximum number of iterations for the solver.
    tol : float, default=1e-4
        Stopping criterion. For the newton-cg and lbfgs solvers, the iteration
        will stop when ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.
    verbose : int, default=0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.
    solver : {'lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'}, \
            default='lbfgs'
        Numerical solver to use.
    coef : array-like of shape (n_features,), default=None
        Initialization value for coefficients of logistic regression.
        Useless for liblinear solver.
    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
    dual : bool, default=False
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.
    penalty : {'l1', 'l2', 'elasticnet'}, default='l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties. 'elasticnet' is
        only supported by the 'saga' solver.
    intercept_scaling : float, default=1.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.
        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.
    multi_class : {'ovr', 'multinomial', 'auto'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.
        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.22
            Default changed from 'ovr' to 'auto' in 0.22.
    random_state : int, RandomState instance, default=None
        Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
        data. See :term:`Glossary <random_state>` for details.
    check_input : bool, default=True
        If False, the input arrays X and y will not be checked.
    max_squared_sum : float, default=None
        Maximum squared sum of X over samples. Used only in SAG solver.
        If None, it will be computed, going through all the samples.
        The value should be precomputed to speed up cross validation.
    sample_weight : array-like of shape(n_samples,), default=None
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.
    l1_ratio : float, default=None
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.
    n_threads : int, default=1
       Number of OpenMP threads to use.
    Returns
    -------
    coefs : ndarray of shape (n_cs, n_features) or (n_cs, n_features + 1)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept. For
        ``multiclass='multinomial'``, the shape is (n_classes, n_cs,
        n_features) or (n_classes, n_cs, n_features + 1).
    Cs : ndarray
        Grid of Cs used for cross-validation.
    n_iter : array of shape (n_cs,)
        Actual number of iteration for each Cs.
    Notes
    -----
    You might get slightly different results with the solver liblinear than
    with the others since this uses LIBLINEAR which penalizes the intercept.
    .. versionchanged:: 0.19
        The "copy" parameter was removed.
    """
    if isinstance(Cs, numbers.Integral):
        Cs = pk.logspace(-4, 4, Cs)

    solver = _check_solver(solver, penalty, dual)

    _, n_features = X.shape

    classes = pk.unique(y)

    random_state = check_random_state(random_state)

    multi_class = _check_multi_class(multi_class, solver, len(classes))
    if pos_class is None and multi_class != "multinomial":
        if classes.size > 2:
            raise ValueError("To fit OvR, use the pos_class argument")
        # pk.unique(y) gives labels in sorted order.
        pos_class = classes[1]

    # If sample weights exist, convert them to array (support for lists)
    # and check length
    # Otherwise set them to 1 for all examples
    sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype, copy=True)

    # If class_weights is a dict (provided by the user), the weights
    # are assigned to the original labels. If it is "balanced", then
    # the class_weights are assigned after masking the labels with a OvR.
    le = LabelEncoder()
    if isinstance(class_weight, dict) or multi_class == "multinomial":
        class_weight_ = compute_class_weight(class_weight, classes=classes, y=y)
        sample_weight *= class_weight_[le.fit_transform(y)]

    # For doing a ovr, we need to mask the labels first. for the
    # multinomial case this is not necessary.

    if multi_class == "ovr":
        w0 = pk.zeros(n_features + int(fit_intercept), dtype=X.dtype)
        mask = y == pos_class
        y_bin = pk.ones(y.shape, dtype=X.dtype)
        if solver in ["lbfgs", "newton-cg"]:
            # HalfBinomialLoss, used for those solvers, represents y in [0, 1] instead
            # of in [-1, 1].
            mask_classes = asarray([0, 1])
            y_bin[~mask] = 0.0
        else:
            mask_classes = asarray([-1, 1])
            y_bin[~mask] = -1.0

        # for compute_class_weight
        if class_weight == "balanced":
            class_weight_ = compute_class_weight(
                class_weight, classes=mask_classes, y=y_bin
            )
            sample_weight *= class_weight_[le.fit_transform(y_bin)]

    else:
        if solver in ["sag", "saga", "lbfgs", "newton-cg"]:
            # SAG, lbfgs and newton-cg multinomial solvers need LabelEncoder,
            # not LabelBinarizer, i.e. y as a 1d-array of integers.
            # LabelEncoder also saves memory compared to LabelBinarizer, especially
            # when n_classes is large.
            le = LabelEncoder()
            Y_multi = asarray(le.fit_transform(y))
        else:
            # For liblinear solver, apply LabelBinarizer, i.e. y is one-hot encoded.
            lbin = LabelBinarizer()
            Y_multi = asarray(lbin.fit_transform(y))
            if Y_multi.shape[1] == 1:
                Y_multi = pk.hstack(pk.negative(pk.subtract(Y_multi, asarray([1]))), Y_multi)

        w0 = pk.zeros(
            (classes.size, n_features + int(fit_intercept)), dtype=X.dtype
        )

    if coef is not None:
        # it must work both giving the bias term and not
        if multi_class == "ovr":
            if coef.size not in (n_features, w0.size):
                raise ValueError(
                    "Initialization coef is of shape %d, expected shape %d or %d"
                    % (coef.size, n_features, w0.size)
                )
            w0[: coef.size] = coef
        else:
            # For binary problems coef.shape[0] should be 1, otherwise it
            # should be classes.size.
            n_classes = classes.size
            if n_classes == 2:
                n_classes = 1

            if coef.shape[0] != n_classes or coef.shape[1] not in (
                n_features,
                n_features + 1,
            ):
                raise ValueError(
                    "Initialization coef is of shape (%d, %d), expected "
                    "shape (%d, %d) or (%d, %d)"
                    % (
                        coef.shape[0],
                        coef.shape[1],
                        classes.size,
                        n_features,
                        classes.size,
                        n_features + 1,
                    )
                )

            if n_classes == 1:
                w0[0, : coef.shape[1]] = -coef
                w0[1, : coef.shape[1]] = coef
            else:
                w0[:, : coef.shape[1]] = coef

    if multi_class == "multinomial":
        if solver in ["lbfgs", "newton-cg"]:
            # scipy.optimize.minimize and newton-cg accept only ravelled parameters,
            # i.e. 1d-arrays. LinearModelLoss expects classes to be contiguous and
            # reconstructs the 2d-array via w0.reshape((n_classes, -1), order="F").
            # As w0 is F-contiguous, ravel(order="F") also avoids a copy.
            w0 = pk.ravel(w0, order="F")

            loss = LinearModelLoss(
                base_loss=HalfMultinomialLoss(n_classes=classes.size),
                fit_intercept=fit_intercept,
            )
        target = Y_multi
        if solver in "lbfgs":
            func = loss.loss_gradient
        elif solver == "newton-cg":
            func = loss.loss
            grad = loss.gradient
            hess = loss.gradient_hessian_product  # hess = [gradient, hessp]
        warm_start_sag = {"coef": pk.transpose(w0)}
    else:
        target = y_bin
        if solver == "lbfgs":
            loss = LinearModelLoss(
                base_loss=HalfBinomialLoss(), fit_intercept=fit_intercept
            )
            func = loss.loss_gradient
        elif solver == "newton-cg":
            loss = LinearModelLoss(
                base_loss=HalfBinomialLoss(), fit_intercept=fit_intercept
            )
            func = loss.loss
            grad = loss.gradient
            hess = loss.gradient_hessian_product  # hess = [gradient, hessp]
        warm_start_sag = {"coef": pk.expand_dims(w0, axis=1)}

    coefs = list()
    
    n_iter = pk.zeros(len(Cs), dtype=pk.int32)
    for i, C in enumerate(Cs):
        if solver == "lbfgs":
            l2_reg_strength = 1.0 / C
            iprint = [-1, 50, 1, 100, 101][
                pk.searchsorted([0, 1, 2, 3], verbose)
            ]
            opt_res = optimize.minimize(
                func,
                np.asarray(w0),
                method="L-BFGS-B",
                jac=True,
                args=(np.asarray(X), np.asarray(target), sample_weight, l2_reg_strength, n_threads),
                options={"iprint": iprint, "gtol": tol, "maxiter": max_iter},
            )

            n_iter_i = _check_optimize_result(
                solver,
                opt_res,
                max_iter,
                extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,
            )
            w0, loss = asarray(opt_res.x), opt_res.fun
        elif solver == "newton-cg":
            l2_reg_strength = 1.0 / C
            args = (X, target, sample_weight, l2_reg_strength, n_threads)
            w0, n_iter_i = _newton_cg(
                hess, func, grad, w0, args=args, maxiter=max_iter, tol=tol
            )
        elif solver == "liblinear":
            coef_, intercept_, n_iter_i, = _fit_liblinear(
                X,
                target,
                C,
                fit_intercept,
                intercept_scaling,
                None,
                penalty,
                dual,
                verbose,
                max_iter,
                tol,
                random_state,
                sample_weight=sample_weight,
            )
            coef_ = asarray(coef_)
            if fit_intercept:
                w0 = pk.hstack(pk.ravel(coef_), intercept_)
            else:
                w0 = pk.ravel(coef_)

        elif solver in ["sag", "saga"]:
            if multi_class == "multinomial":
                target = target.astype(X.dtype, copy=False)
                loss = "multinomial"
            else:
                loss = "log"
            # alpha is for L2-norm, beta is for L1-norm
            if penalty == "l1":
                alpha = 0.0
                beta = 1.0 / C
            elif penalty == "l2":
                alpha = 1.0 / C
                beta = 0.0
            else:  # Elastic-Net penalty
                alpha = (1.0 / C) * (1 - l1_ratio)
                beta = (1.0 / C) * l1_ratio

            w0, n_iter_i, warm_start_sag = sag_solver(
                X,
                target,
                sample_weight,
                loss,
                alpha,
                beta,
                max_iter,
                tol,
                verbose,
                random_state,
                False,
                max_squared_sum,
                warm_start_sag,
                is_saga=(solver == "saga"),
            )

        else:
            raise ValueError(
                "solver must be one of {'liblinear', 'lbfgs', "
                "'newton-cg', 'sag'}, got '%s' instead" % solver
            )

        if multi_class == "multinomial":
            n_classes = max(2, classes.size)
            if solver in ["lbfgs", "newton-cg"]:
                multi_w0 = pk.reshape(w0, (n_classes, -1), order="F")
            else:
                multi_w0 = w0
            coefs.append(asarray(multi_w0))
        else:
            coefs.append(asarray(w0))

        n_iter[i] = n_iter_i

    return asarray(coefs), asarray(Cs), n_iter


class LogisticRegression(LinearClassifierMixin, SparseCoefMixin, BaseEstimator):
    """
    Logistic Regression (aka logit, MaxEnt) classifier.
    In the multiclass case, the training algorithm uses the one-vs-rest (OvR)
    scheme if the 'multi_class' option is set to 'ovr', and uses the
    cross-entropy loss if the 'multi_class' option is set to 'multinomial'.
    (Currently the 'multinomial' option is supported only by the 'lbfgs',
    'sag', 'saga' and 'newton-cg' solvers.)
    This class implements regularized logistic regression using the
    'liblinear' library, 'newton-cg', 'sag', 'saga' and 'lbfgs' solvers. **Note
    that regularization is applied by default**. It can handle both dense
    and sparse input. Use C-ordered arrays or CSR matrices containing 64-bit
    floats for optimal performance; any other input format will be converted
    (and copied).
    The 'newton-cg', 'sag', and 'lbfgs' solvers support only L2 regularization
    with primal formulation, or no regularization. The 'liblinear' solver
    supports both L1 and L2 regularization, with a dual formulation only for
    the L2 penalty. The Elastic-Net regularization is only supported by the
    'saga' solver.
    Read more in the :ref:`User Guide <logistic_regression>`.
    Parameters
    ----------
    penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
        Specify the norm of the penalty:
        - `'none'`: no penalty is added;
        - `'l2'`: add a L2 penalty term and it is the default choice;
        - `'l1'`: add a L1 penalty term;
        - `'elasticnet'`: both L1 and L2 penalty terms are added.
        .. warning::
           Some penalties may not work with some solvers. See the parameter
           `solver` below, to know the compatibility between the penalty and
           solver.
        .. versionadded:: 0.19
           l1 penalty with SAGA solver (allowing 'multinomial' + L1)
    dual : bool, default=False
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.
    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
    intercept_scaling : float, default=1
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.
        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.
    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
        .. versionadded:: 0.17
           *class_weight='balanced'*
    random_state : int, RandomState instance, default=None
        Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
        data. See :term:`Glossary <random_state>` for details.
    solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, \
            default='lbfgs'
        Algorithm to use in the optimization problem. Default is 'lbfgs'.
        To choose a solver, you might want to consider the following aspects:
            - For small datasets, 'liblinear' is a good choice, whereas 'sag'
              and 'saga' are faster for large ones;
            - For multiclass problems, only 'newton-cg', 'sag', 'saga' and
              'lbfgs' handle multinomial loss;
            - 'liblinear' is limited to one-versus-rest schemes.
        .. warning::
           The choice of the algorithm depends on the penalty chosen:
           Supported penalties by solver:
           - 'newton-cg'   -   ['l2', 'none']
           - 'lbfgs'       -   ['l2', 'none']
           - 'liblinear'   -   ['l1', 'l2']
           - 'sag'         -   ['l2', 'none']
           - 'saga'        -   ['elasticnet', 'l1', 'l2', 'none']
        .. note::
           'sag' and 'saga' fast convergence is only guaranteed on
           features with approximately the same scale. You can
           preprocess the data with a scaler from :mod:`sklearn.preprocessing`.
        .. seealso::
           Refer to the User Guide for more information regarding
           :class:`LogisticRegression` and more specifically the
           :ref:`Table <Logistic_regression>`
           summarizing solver/penalty supports.
        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.
        .. versionchanged:: 0.22
            The default solver changed from 'liblinear' to 'lbfgs' in 0.22.
    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.
    multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.
        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.22
            Default changed from 'ovr' to 'auto' in 0.22.
    verbose : int, default=0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.
    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Useless for liblinear solver. See :term:`the Glossary <warm_start>`.
        .. versionadded:: 0.17
           *warm_start* to support *lbfgs*, *newton-cg*, *sag*, *saga* solvers.
    n_jobs : int, default=None
        Number of CPU cores used when parallelizing over classes if
        multi_class='ovr'". This parameter is ignored when the ``solver`` is
        set to 'liblinear' regardless of whether 'multi_class' is specified or
        not. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.
        See :term:`Glossary <n_jobs>` for more details.
    l1_ratio : float, default=None
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.
    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.
        `coef_` is of shape (1, n_features) when the given problem is binary.
        In particular, when `multi_class='multinomial'`, `coef_` corresponds
        to outcome 1 (True) and `-coef_` corresponds to outcome 0 (False).
    intercept_ : ndarray of shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.
        If `fit_intercept` is set to False, the intercept is set to zero.
        `intercept_` is of shape (1,) when the given problem is binary.
        In particular, when `multi_class='multinomial'`, `intercept_`
        corresponds to outcome 1 (True) and `-intercept_` corresponds to
        outcome 0 (False).
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    n_iter_ : ndarray of shape (n_classes,) or (1, )
        Actual number of iterations for all classes. If binary or multinomial,
        it returns only 1 element. For liblinear solver, only the maximum
        number of iteration across all classes is given.
        .. versionchanged:: 0.20
            In SciPy <= 1.0.0 the number of lbfgs iterations may exceed
            ``max_iter``. ``n_iter_`` will now report at most ``max_iter``.
    See Also
    --------
    SGDClassifier : Incrementally trained logistic regression (when given
        the parameter ``loss="log"``).
    LogisticRegressionCV : Logistic regression with built-in cross validation.
    Notes
    -----
    The underlying C implementation uses a random number generator to
    select features when fitting the model. It is thus not uncommon,
    to have slightly different results for the same input data. If
    that happens, try with a smaller tol parameter.
    Predict output may not match that of standalone liblinear in certain
    cases. See :ref:`differences from liblinear <liblinear_differences>`
    in the narrative documentation.
    References
    ----------
    L-BFGS-B -- Software for Large-scale Bound-constrained Optimization
        Ciyou Zhu, Richard Byrd, Jorge Nocedal and Jose Luis Morales.
        http://users.iems.northwestern.edu/~nocedal/lbfgsb.html
    LIBLINEAR -- A Library for Large Linear Classification
        https://www.csie.ntu.edu.tw/~cjlin/liblinear/
    SAG -- Mark Schmidt, Nicolas Le Roux, and Francis Bach
        Minimizing Finite Sums with the Stochastic Average Gradient
        https://hal.inria.fr/hal-00860051/document
    SAGA -- Defazio, A., Bach F. & Lacoste-Julien S. (2014).
            :arxiv:`"SAGA: A Fast Incremental Gradient Method With Support
            for Non-Strongly Convex Composite Objectives" <1407.0202>`
    Hsiang-Fu Yu, Fang-Lan Huang, Chih-Jen Lin (2011). Dual coordinate descent
        methods for logistic regression and maximum entropy models.
        Machine Learning 85(1-2):41-75.
        https://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = LogisticRegression(random_state=0).fit(X, y)
    >>> clf.predict(X[:2, :])
    array([0, 0])
    >>> clf.predict_proba(X[:2, :])
    array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
           [9.7...e-01, 2.8...e-02, ...e-08]])
    >>> clf.score(X, y)
    0.97...
    """

    def __init__(
        self,
        penalty="l2",
        *,
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):

        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        sample_weight : array-like of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
            .. versionadded:: 0.17
               *sample_weight* support to LogisticRegression.
        Returns
        -------
        self
            Fitted estimator.
        Notes
        -----
        The SAGA solver supports both float64 and float32 bit arrays.
        """
        solver = _check_solver(self.solver, self.penalty, self.dual)

        if not isinstance(self.C, numbers.Number) or self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)" % self.C)
        if self.penalty == "elasticnet":
            if (
                not isinstance(self.l1_ratio, numbers.Number)
                or self.l1_ratio < 0
                or self.l1_ratio > 1
            ):
                raise ValueError(
                    "l1_ratio must be between 0 and 1; got (l1_ratio=%r)"
                    % self.l1_ratio
                )
        elif self.l1_ratio is not None:
            warnings.warn(
                "l1_ratio parameter is only used when penalty is "
                "'elasticnet'. Got "
                "(penalty={})".format(self.penalty)
            )
        if self.penalty == "none":
            if self.C != 1.0:  # default values
                warnings.warn(
                    "Setting penalty='none' will ignore the C and l1_ratio parameters"
                )
                # Note that check for l1_ratio is done right above
            C_ = pk.inf
            penalty = "l2"
        else:
            C_ = self.C
            penalty = self.penalty
        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError(
                "Maximum number of iteration must be positive; got (max_iter=%r)"
                % self.max_iter
            )
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError(
                "Tolerance for stopping criteria must be positive; got (tol=%r)"
                % self.tol
            )

        if solver == "lbfgs":
            _dtype = np.float64
        else:
            _dtype = [np.float64, np.float32]

        X, y = self._validate_data(
            np.asarray(X),
            np.asarray(y),
            accept_sparse="csr",
            dtype=_dtype,
            order="C",
            accept_large_sparse=solver not in ["liblinear", "sag", "saga"],
        )
        check_classification_targets(y)
        
        X = asarray(X)
        y = asarray(y)
        self.classes_ = pk.unique(y)

        multi_class = _check_multi_class(self.multi_class, solver, len(self.classes_))

        if solver == "liblinear":
            if effective_n_jobs(self.n_jobs) != 1:
                warnings.warn(
                    "'n_jobs' > 1 does not have any effect when"
                    " 'solver' is set to 'liblinear'. Got 'n_jobs'"
                    " = {}.".format(effective_n_jobs(self.n_jobs))
                )
            self.coef_, self.intercept_, self.n_iter_ = _fit_liblinear(
                X,
                y,
                self.C,
                self.fit_intercept,
                self.intercept_scaling,
                self.class_weight,
                self.penalty,
                self.dual,
                self.verbose,
                self.max_iter,
                self.tol,
                self.random_state,
                sample_weight=sample_weight,
            )
            return self

        if solver in ["sag", "saga"]:
            max_squared_sum = row_norms(X, squared=True).max()
        else:
            max_squared_sum = None

        n_classes = len(self.classes_)
        classes_ = self.classes_
        if n_classes < 2:
            raise ValueError(
                "This solver needs samples of at least 2 classes"
                " in the data, but the data contains only one"
                " class: %r"
                % classes_[0]
            )

        if len(self.classes_) == 2:
            n_classes = 1
            classes_ = classes_[1:]

        if self.warm_start:
            warm_start_coef = getattr(self, "coef_", None)
        else:
            warm_start_coef = None

        # Hack so that we iterate only once for the multinomial case.
        if multi_class == "multinomial":
            classes_ = [None]
            warm_start_coef = [warm_start_coef]
        if warm_start_coef is None:
            warm_start_coef = [None] * n_classes

        path_func = delayed(_logistic_regression_path)

        # The SAG solver releases the GIL so it's more efficient to use
        # threads for this solver.
        if solver in ["sag", "saga"]:
            prefer = "threads"
        else:
            prefer = "processes"

        # TODO: Refactor this to avoid joblib parallelism entirely when doing binary
        # and multinomial multiclass classification and use joblib only for the
        # one-vs-rest multiclass case.
        if (
            solver in ["lbfgs", "newton-cg"]
            and len(classes_) == 1
            and effective_n_jobs(self.n_jobs) == 1
        ):
            # In the future, we would like n_threads = _openmp_effective_n_threads()
            # For the time being, we just do
            n_threads = 1
        else:
            n_threads = 1

        fold_coefs_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer=prefer)(
            path_func(
                X,
                y,
                pos_class=class_,
                Cs=[C_],
                l1_ratio=self.l1_ratio,
                fit_intercept=self.fit_intercept,
                tol=self.tol,
                verbose=self.verbose,
                solver=solver,
                multi_class=multi_class,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                check_input=False,
                random_state=self.random_state,
                coef=warm_start_coef_,
                penalty=penalty,
                max_squared_sum=max_squared_sum,
                sample_weight=sample_weight,
                n_threads=n_threads,
            )
            for class_, warm_start_coef_ in zip(classes_, warm_start_coef)
        )

        fold_coefs_, _, n_iter_ = zip(*fold_coefs_)
        self.n_iter_ = pk.col(asarray(n_iter_), 0)

        n_features = X.shape[1]
        if multi_class == "multinomial":
            self.coef_ = fold_coefs_[0][0]
        else:
            self.coef_ = asarray(fold_coefs_)
            self.coef_ = self.coef_.reshape(
                n_classes, n_features + int(self.fit_intercept)
            )

        if self.fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_ = self.coef_[:, :-1]
        else:
            self.intercept_ = pk.zeros(n_classes)

        return self

    def predict_proba(self, X):
        """
        Probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        For a multi_class problem, if multi_class is set to be "multinomial"
        the softmax function is used to find the predicted probability of
        each class.
        Else use a one-vs-rest approach, i.e calculate the probability
        of each class assuming it to be positive using the logistic function.
        and normalize these values across all the classes.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        check_is_fitted(self)

        ovr = self.multi_class in ["ovr", "warn"] or (
            self.multi_class == "auto"
            and (self.classes_.size <= 2 or self.solver == "liblinear")
        )
        if ovr:
            return super()._predict_proba_lr(X)
        else:
            decision = asarray(self.decision_function(X))

            if decision.ndim == 1:
                # Workaround for multi_class="multinomial" and binary outcomes
                # which requires softmax prediction with only a 1D decision.
                decision_2d = pk.hstack(pk.negative(decision), decision)
            else:
                decision_2d = decision
            return softmax(decision_2d, copy=False)

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return pk.log(self.predict_proba(X))
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data matrix for which we want to get the predictions.
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the class labels for each sample.
        """
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(int)
        else:
            indices = scores.argmax(axis=1)
        
        return pk.index(self.classes_, asarray(indices, dtype=pk.int32))

def main():
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(asarray(X), asarray(y))
    clf.predict(asarray(X[:2, :]))
    clf.predict_proba(asarray(X[:2, :]))
    print(clf.score(asarray(X), asarray(y)))

main()