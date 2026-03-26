"""Backward-compatibility shim — prefer importing from calibration / evaluation directly."""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "conformal_predictions.training is deprecated. "
    "Import from conformal_predictions.calibration or "
    "conformal_predictions.evaluation instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export calibration API
from conformal_predictions.calibration import (  # noqa: F401, E402
    _compute_mu_hat,
    _get_expected_background,
    _get_expected_signal,
    _get_proportionate_beta_true,
    _get_proportionate_gamma_true,
    _nonconformity_scores,
    _random_perturbation_for_numerical_stability,
    compute_mu_hat,
    compute_nonconformity_scores,
)

# Re-export data helpers (moved to data.toy in Step 2/6)
from conformal_predictions.data.toy import (  # noqa: F401, E402
    _experiment_prefix,
    list_split_files,
)

# Re-export evaluation API
from conformal_predictions.evaluation import (  # noqa: F401, E402
    compute_confidence_interval,
    compute_empirical_coverage,
    evaluate_models,
    get_events_count,
    inference_on_test_set,
)
