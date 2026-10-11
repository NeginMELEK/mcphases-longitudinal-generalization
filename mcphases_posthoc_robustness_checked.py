#!/usr/bin/env python3
r"""Frozen POST HOC mcPHASES robustness analysis; LOCAL, AGGREGATE OUTPUT ONLY.

Python >=3.10. Dependencies: duckdb numpy pandas scipy scikit-learn.
The completed Ridge directory is READ ONLY. No Ridge estimator is fitted,
predicted with, retuned, or rerun. Its aggregate results are imported only after
all 12 exploratory pipelines have been fitted and frozen. No network operations,
participant exports, fitted-model exports, coefficients or predictions are saved.

Use the original mcphases_frozen_analysis.py saved INSIDE the completed results
directory. Its verified source supplies data cleaning, seven-day construction,
participant weights, folds, baseline and bootstrap summary functions. The exact
original imputation/scaling and prediction statements are copied below.

Only Elastic Net, RBF-SVR and Random Forest are fitted, on the unchanged seven-day
HR + steps core. All hyperparameter development uses R1 alone. No multimodal
extensions, outcome-informed exclusions, feature selection, recalibration or
grid expansion. Each target/experiment has its own R1-only tuning and final fit.
Five original seeded participant folds; pooled OOF original-unit PB MAE;
first enumerated numerical tie (rtol=atol=1e-12). Targets remain log1p without
additional outcome standardization. Predictions are expm1, truncated at zero.

Ridge-vs-exploratory POINT differences are reported on the verified identical
observations. Paired difference CIs against Ridge CANNOT be recovered from its
aggregate files: participant-level Ridge errors/predictions were not saved.
These CIs are explicitly NOT_COMPUTABLE_FROM_SAVED_AGGREGATES. Marginal Ridge
CIs are copied unchanged. No independent-CI subtraction is substituted. All new
model metrics and paired comparisons with the unchanged constant baseline use
10,000 participant resamples and descriptive 95% intervals, with no refitting.

Every failed candidate, convergence failure or nonfinite prediction STOPS the
run with SEARCH_INCOMPLETE; no candidates are silently removed. All inputs and
completed Ridge files are hashed before/after; a change invalidates the run.

Example (PowerShell, from D:\New paper):
  py .\mcphases_posthoc_robustness.py --self-test
  py .\mcphases_posthoc_robustness.py --data "D:\New paper\mcphases" `
     --ridge-results "D:\New paper\frozen_analysis_20261008_192018"
"""
from __future__ import annotations

import argparse
import ast
import collections
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap
import time
import types
import warnings

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
# Importing the original script must not create a cache in its results folder.
sys.dont_write_bytecode = True

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import ElasticNet
    from sklearn.svm import SVR
except ImportError:
    raise SystemExit('Install: py -m pip install duckdb numpy pandas scipy scikit-learn')

SEED = 20261008
NBOOT = 10000
ORIGINAL_CANONICAL_SHA256 = 'ad8f7bd2962688219842d34a3dc35994dc29434f2164a3ff15bd17b634a0bfb7'
MODELS = ('elastic_net', 'rbf_svr', 'random_forest')
EXPERIMENTS = ('temporal', 'independent')
TARGETS = ('LH', 'E3G')
GRID_AXES = {
    'elastic_net': {'alpha': [1.0, 0.1, 0.01, 0.001, 0.0001], 'l1_ratio': [0.9, 0.5, 0.1]},
    'rbf_svr': {'C': [0.1, 1.0, 10.0, 100.0], 'epsilon': [0.2, 0.1, 0.05], 'gamma': [0.01, 0.1, 'scale']},
    'random_forest': {'max_depth': [3, 6, None], 'min_samples_leaf': [20, 5], 'max_features': ['sqrt', 0.5]},
}
FIXED = {
    'elastic_net': dict(fit_intercept=True, selection='cyclic', max_iter=100000,
                        tol=1e-5, positive=False, warm_start=False, precompute=False),
    'rbf_svr': dict(kernel='rbf', tol=1e-3, shrinking=True, cache_size=512, max_iter=-1),
    'random_forest': dict(n_estimators=300, criterion='squared_error', bootstrap=False,
                         min_samples_split=2, min_weight_fraction_leaf=0.01,
                         max_samples=None, oob_score=False, ccp_alpha=0.0,
                         min_impurity_decrease=0.0, max_leaf_nodes=None,
                         warm_start=False, random_state=SEED, n_jobs=2),
}
REQUIRED_REFERENCE = (
    'run_manifest.json', 'mcphases_frozen_analysis.py',
    '00_input_file_fingerprints.csv', '01_eligibility_counts.csv',
    '05_frozen_core_models.csv', '13_feature_dictionary.csv',
    '06_confirmatory_metrics.csv', '07_participant_independent_metrics.csv',
    '06_confirmatory_baseline_comparisons.csv',
    '07_participant_independent_baseline_comparisons.csv',
)


class StopRun(Exception):
    """Only fixed, non-participant-bearing messages may enter this exception."""


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def candidates(name):
    axes = GRID_AXES[name]
    return [dict(zip(axes, v)) for v in itertools.product(*axes.values())]


def parameter_json(parameters):
    return json.dumps(parameters, sort_keys=True, separators=(',', ':'), allow_nan=False)


def load_original(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise StopRun('Original mcphases_frozen_analysis.py is required.')
    canonical = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    if canonical != ORIGINAL_CANONICAL_SHA256:
        raise StopRun('Original source differs from the verified completed Ridge script. No fitting permitted.')
    spec = importlib.util.spec_from_file_location('_verified_mcphases_original', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # __name__ is not __main__; main is NEVER called.
    if module.SEED != SEED or module.NBOOT != NBOOT:
        raise StopRun('Original seed/bootstrap settings mismatch.')
    return module


def make_estimator(name, parameters):
    constructors = {'elastic_net': ElasticNet, 'rbf_svr': SVR,
                    'random_forest': RandomForestRegressor}
    return constructors[name](**FIXED[name], **parameters)


class ExploratoryPipeline:
    def __init__(self, base, name, parameters):
        self.base = base
        self.name = name
        self.parameters = dict(parameters)
        self.diagnostic = {}

    def fit(self, X, y, ids):
        # COPY of original Pipeline.fit preprocessing, statement-for-statement.
        # Only weights() is qualified with the verified module name.
        X = np.asarray(X, dtype=float); w = self.base.weights(ids)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.fill = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        self.fill = np.nan_to_num(self.fill, nan=0.0)
        a = np.where(np.isfinite(X), X, self.fill)
        self.mean = np.average(a, axis=0, weights=w)
        self.scale = np.sqrt(np.average((a-self.mean)**2, axis=0, weights=w))
        self.scale[self.scale < 1e-12] = 1
        transformed = (a-self.mean)/self.scale
        if not np.isfinite(transformed).all() or not np.isfinite(y).all() or (np.asarray(y) < 0).any():
            raise StopRun('Nonfinite training matrix/target; search incomplete.')
        _, inverse = np.unique(np.asarray(ids), return_inverse=True)
        total_by_person = np.bincount(inverse, weights=w)
        if not np.allclose(total_by_person, len(ids)/len(total_by_person), rtol=1e-12, atol=1e-12):
            raise StopRun('Equal participant training weight assertion failed.')
        self.model = make_estimator(self.name, self.parameters)
        started = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter('always')
                self.model.fit(transformed, np.log1p(y), sample_weight=w)
        except Exception as error:
            self.diagnostic = dict(status='FAILED', exception_class=type(error).__name__)
            raise StopRun('Estimator fit failed; search incomplete. Raw exception suppressed.') from None
        self.diagnostic = dict(
            status='OK', fit_seconds=time.perf_counter()-started,
            warning_count=len(recorded),
            warning_categories='|'.join(sorted({item.category.__name__ for item in recorded})),
            convergence_warning_count=sum(issubclass(item.category, ConvergenceWarning) for item in recorded),
            train_days=len(ids), train_participants=len(total_by_person),
            participant_total_weight_min=float(total_by_person.min()),
            participant_total_weight_max=float(total_by_person.max()),
        )
        if self.name == 'elastic_net':
            self.diagnostic.update(n_iter=int(self.model.n_iter_), dual_gap=float(self.model.dual_gap_))
            if not np.isfinite(self.model.dual_gap_) or self.model.n_iter_ >= FIXED[self.name]['max_iter']:
                self.diagnostic['status'] = 'NONCONVERGED'
        elif self.name == 'rbf_svr':
            self.diagnostic.update(fit_status=int(self.model.fit_status_),
                                   n_iter=int(np.max(np.asarray(self.model.n_iter_))),
                                   effective_gamma=float(self.model._gamma))
            if self.model.fit_status_ != 0:
                self.diagnostic['status'] = 'NONCONVERGED'
        else:
            roots = np.array([tree.tree_.weighted_n_node_samples[0] for tree in self.model.estimators_])
            self.diagnostic.update(trees=len(roots), root_total_weight_max_deviation=float(np.max(np.abs(roots-w.sum()))))
            if len(roots) != 300 or not np.allclose(roots, w.sum(), rtol=1e-10, atol=1e-8):
                self.diagnostic['status'] = 'RF_WEIGHT_ASSERTION_FAILED'
        if self.diagnostic['convergence_warning_count']:
            self.diagnostic['status'] = 'NONCONVERGED'
        if self.diagnostic['status'] != 'OK':
            raise StopRun('Convergence/weight verification failed; search incomplete. No candidate dropped.')
        return self

    def predict(self, X):
        # COPY of original Pipeline.predict imputation/back-transformation.
        X = np.asarray(X, dtype=float); a = np.where(np.isfinite(X), X, self.fill)
        with np.errstate(over='ignore', invalid='ignore'):
            p = np.maximum(0, np.expm1(self.model.predict((a-self.mean)/self.scale)))
        if not np.isfinite(p).all():
            raise StopRun('Nonfinite predictions; search incomplete, analysis stopped.')
        return p


class Diagnostics:
    def __init__(self, base, out=None):
        self.base = base; self.out = out; self.rows = []

    def add(self, row):
        self.rows.append(row)

    def save(self):
        if self.out is not None and self.rows:
            self.base.atomic_csv(self.rows, self.out/'04_fit_diagnostics.csv')


def first_tied_candidate(scores):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise StopRun('Nonfinite CV score; search incomplete.')
    return int(np.flatnonzero(np.isclose(scores, scores.min(), rtol=1e-12, atol=1e-12))[0])


def train_search(base, X, y, ids, folds, name, experiment, target, diagnostics):
    ids = np.asarray(ids); fold = np.array([folds[pid] for pid in ids])
    grid = candidates(name); oof = np.full((len(y), len(grid)), np.nan)
    for k in range(5):
        tr = fold != k; va = fold == k
        if not tr.any() or not va.any() or set(ids[tr]) & set(ids[va]):
            raise StopRun('Participant grouped-fold assertion failed.')
        for index, parameters in enumerate(grid):
            pipe = ExploratoryPipeline(base, name, parameters)
            context = dict(experiment=experiment, target=target, estimator=name,
                           phase='R1_CV', fold=k, candidate_index=index,
                           parameters=parameter_json(parameters))
            try:
                pipe.fit(X[tr], y[tr], ids[tr])
                oof[va, index] = pipe.predict(X[va])
            except Exception:
                diagnostics.add(dict(context, **pipe.diagnostic, search_complete=False,
                                     prediction_or_fit_failed=True))
                diagnostics.save()
                raise StopRun('Candidate failed; search incomplete. No R2 scoring or reduced grid.') from None
            diagnostics.add(dict(context, **pipe.diagnostic, search_complete=None,
                                 prediction_or_fit_failed=False))
        diagnostics.save()
        print(f'  {experiment} {target} {name}: R1 fold {k+1}/5 completed.', flush=True)
    # The original pooled participant-balanced MAE; never an estimator score.
    scores = np.array([base.pbmae(y, oof[:, i], ids) for i in range(len(grid))])
    chosen = first_tied_candidate(scores)
    pipe = ExploratoryPipeline(base, name, grid[chosen])
    context = dict(experiment=experiment, target=target, estimator=name,
                   phase='R1_FINAL', fold=None, candidate_index=chosen,
                   parameters=parameter_json(grid[chosen]))
    try:
        pipe.fit(X, y, ids)
        # Check finiteness on R1 only. No R2 predictions occur here.
        pipe.predict(X)
    except Exception:
        diagnostics.add(dict(context, **pipe.diagnostic, search_complete=False,
                             prediction_or_fit_failed=True))
        diagnostics.save()
        raise StopRun('Final R1 fit/prediction failed; search incomplete.') from None
    diagnostics.add(dict(context, **pipe.diagnostic, search_complete=True,
                         prediction_or_fit_failed=False))
    diagnostics.save()
    rows = [dict(experiment=experiment, target=target, estimator=name,
                 candidate_index=i, parameters=parameter_json(parameters),
                 R1_grouped_OOF_PB_MAE=float(score), selected=i == chosen,
                 search_complete=True)
            for i, (parameters, score) in enumerate(zip(grid, scores))]
    return pipe, rows, chosen


def count_gate(base, targets, members, daily):
    returning = members['R1'] & members['R2']; only = members['R1'] - members['R2']
    cohorts = {}; histories = {}; counts = []; passed = True
    for target in TARGETS:
        data = targets[target]
        history = base.histories(daily, data, base.DAILY_CORE)
        keep = np.isfinite(history).all(axis=(1, 2))
        cohorts[target] = data.loc[keep].reset_index(drop=True); histories[target] = history[keep]
        for label, round_name, people, expected in (
                ('R1_all', 'R1', members['R1'], 2981 if target == 'LH' else 2980),
                ('R2_all', 'R2', members['R2'], 1677), ('R1_only', 'R1', only, 1451)):
            part = cohorts[target][(cohorts[target]._r == round_name) & cohorts[target]._id.isin(people)]
            expected_people = 20 if round_name == 'R2' else (22 if label == 'R1_only' else None)
            ok = len(part) == expected and (expected_people is None or part._id.nunique() == expected_people)
            passed = passed and ok
            counts.append(dict(target=target, cohort=label, days=len(part),
                               participants=int(part._id.nunique()), expected_days=expected,
                               expected_participants=expected_people, count_gate='PASS' if ok else 'FAIL'))
    roster_ok = (len(members['R1']) == 42 and len(members['R2']) == 20
                 and len(returning) == 20 and len(only) == 22)
    return cohorts, histories, counts, only, bool(passed and roster_ok)


def build_jobs(base, cohorts, histories, members, only, saved_models, dictionary):
    folds = {'temporal': base.foldmap(members['R1']), 'independent': base.foldmap(only)}
    jobs = {}; schemas = []; fold_counts = []
    for experiment in EXPERIMENTS:
        for target in TARGETS:
            data = cohorts[target]
            X, _ = base.features(histories[target], base.DAILY_CORE, 7)
            saved_schema = dictionary.loc[dictionary.variant == 'core7', 'feature'].tolist()
            if X.columns.tolist() != saved_schema:
                raise StopRun('Feature schema/order differs from completed Ridge. No fitting.')
            trainmask = (data._r == 'R1') & (data._id.isin(only) if experiment == 'independent' else True)
            testmask = data._r == 'R2'; tr = data[trainmask]; te = data[testmask]
            if experiment == 'independent' and set(tr._id) & set(te._id):
                raise StopRun('Participant independence assertion failed.')
            reference = saved_models[(saved_models.experiment == experiment)
                                     & (saved_models.target == target) & (saved_models.variant == 'core7')]
            if len(reference) != 1:
                raise StopRun('Completed Ridge model settings are ambiguous/missing.')
            reference = reference.iloc[0]
            expected = dict(train_days=len(tr), train_participants=tr._id.nunique(),
                            test_days=len(te), test_participants=te._id.nunique(), features=X.shape[1])
            if any(int(reference[key]) != int(value) for key, value in expected.items()):
                raise StopRun('Cohort or feature counts differ from completed Ridge. No fitting.')
            baseline = base.wmedian(tr.y.to_numpy(float), tr._id.to_numpy())
            if not np.isclose(baseline, float(reference.baseline_median), rtol=2e-9, atol=2e-9):
                raise StopRun('R1 constant baseline differs from completed Ridge. No fitting.')
            jobs[(experiment, target)] = dict(Xtrain=X.to_numpy()[trainmask], ytrain=tr.y.to_numpy(float),
                ids=tr._id.to_numpy(), Xtest=X.to_numpy()[testmask], test=te.reset_index(drop=True),
                baseline=baseline, folds=folds[experiment], **expected)
            schemas.extend(dict(variant='core7', feature=name, order=i) for i, name in enumerate(X.columns))
            fold = np.array([folds[experiment][pid] for pid in tr._id])
            for k in range(5):
                ids = tr._id.to_numpy(); a = fold != k; b = fold == k
                fold_counts.append(dict(experiment=experiment, target=target, fold=k,
                    train_days=int(a.sum()), validation_days=int(b.sum()),
                    train_participants=len(set(ids[a])), validation_participants=len(set(ids[b])),
                    participant_overlap=len(set(ids[a]) & set(ids[b]))))
    return jobs, pd.DataFrame(schemas).drop_duplicates(), fold_counts


def evaluate_all(base, data, predictions, baseline, experiment, target):
    # Original summaries/from_summaries/interval and seeded multinomial bootstrap.
    y = data.y.to_numpy(float); ids = data._id.to_numpy(); n = len(set(ids))
    mult = np.random.default_rng(SEED).multinomial(n, np.ones(n)/n, size=NBOOT)
    stats = {}; boots = {}; metrics = []; comparisons = []
    series = {'constant_median': np.full(len(y), baseline), **predictions}
    for name, p in series.items():
        if not np.isfinite(p).all():
            raise StopRun('Nonfinite R2 prediction; scoring stopped.')
        s = base.summaries(y, p, ids)
        point = base.from_summaries(s, np.ones((1, n)))
        boot = base.from_summaries(s, mult); stats[name] = point; boots[name] = boot
        for metric, value in point.items():
            lo, hi, nb = base.interval(boot[metric], .95)
            metrics.append(dict(experiment=experiment, target=target, variant='core7',
                estimator=name, metric=metric, value=float(value[0]), CI_level=.95,
                CI_low=lo, CI_high=hi, valid_bootstrap_draws=nb,
                test_participants=n, test_days=len(y), provenance='current_exploratory_run'))
        corr = s[:, 8]; corr = corr[np.isfinite(corr)]
        for metric, value in [('Spearman_included_participants', len(corr)),
                              ('Spearman_excluded_participants', n-len(corr)),
                              ('median_within_participant_Spearman', np.median(corr) if len(corr) else np.nan)]:
            metrics.append(dict(experiment=experiment, target=target, variant='core7',
                estimator=name, metric=metric, value=float(value), test_participants=n,
                test_days=len(y), provenance='current_exploratory_run'))
    for name in MODELS:
        difference = boots['constant_median']['PB_MAE'] - boots[name]['PB_MAE']
        lo, hi, nb = base.interval(difference, .95)
        point = float(stats['constant_median']['PB_MAE'][0]-stats[name]['PB_MAE'][0])
        denominator = float(stats['constant_median']['PB_MAE'][0])
        pct = np.divide(100*difference, boots['constant_median']['PB_MAE'],
                        out=np.full(NBOOT, np.nan), where=boots['constant_median']['PB_MAE'] > 0)
        pctlo, pcthi, _ = base.interval(pct, .95)
        comparisons.append(dict(experiment=experiment, target=target, variant='core7',
            estimator=name, comparison='constant_minus_model_MAE', improvement=point,
            improvement_pct=100*point/denominator if denominator > 0 else np.nan,
            CI_level=.95, CI_low=lo, CI_high=hi, valid_bootstrap_draws=nb,
            improvement_pct_CI_low=pctlo, improvement_pct_CI_high=pcthi,
            status='DESCRIPTIVE_PAIRED_PARTICIPANT_BOOTSTRAP', test_participants=n, test_days=len(y)))
    return metrics, comparisons


def load_completed_results(base, folder, jobs, new_metrics):
    # Called ONLY after all 12 exploratory pipelines are frozen.
    ridge_metrics = []; ridge_comparisons = []; comparisons = []
    for experiment, prefix in [('temporal', '06_confirmatory'), ('independent', '07_participant_independent')]:
        original = pd.read_csv(folder/(prefix+'_metrics.csv'))
        original = original[original.variant == 'core7'].copy()
        if set(original.estimator) != {'ridge', 'constant_median'}:
            raise StopRun('Completed Ridge metrics do not contain the expected estimators.')
        for target in TARGETS:
            reference = original[original.target == target]
            j = jobs[(experiment, target)]
            if reference.empty or not (reference.test_days == j['test_days']).all() or not (reference.test_participants == 20).all():
                raise StopRun('Completed Ridge R2 metric counts do not match identical-row reconstruction.')
            for name in ('ridge', 'constant_median'):
                for metric in ('PB_MAE', 'PB_RMSE', 'PB_log1p_MAE', 'PB_signed_bias',
                               'calibration_intercept', 'calibration_slope',
                               'mean_within_participant_Spearman', 'median_within_participant_Spearman',
                               'Spearman_included_participants', 'Spearman_excluded_participants'):
                    if len(reference[(reference.estimator == name) & (reference.metric == metric)]) != 1:
                        raise StopRun('Completed Ridge metrics are missing/duplicated.')
            # Establish identical baseline metrics, without recreating Ridge predictions.
            for metric in ('PB_MAE', 'PB_RMSE', 'PB_log1p_MAE', 'PB_signed_bias'):
                old_value = float(reference[(reference.estimator == 'constant_median') & (reference.metric == metric)].value.iloc[0])
                new_rows = [r for r in new_metrics if r['experiment'] == experiment and r['target'] == target
                            and r['estimator'] == 'constant_median' and r['metric'] == metric]
                if len(new_rows) != 1 or not np.isclose(old_value, new_rows[0]['value'], rtol=2e-9, atol=2e-9):
                    raise StopRun('Completed and reconstructed constant-baseline metrics differ. Results invalid.')
            ridge_mae = float(reference[(reference.estimator == 'ridge') & (reference.metric == 'PB_MAE')].value.iloc[0])
            for name in MODELS:
                new_mae = next(r['value'] for r in new_metrics if r['experiment'] == experiment
                    and r['target'] == target and r['estimator'] == name and r['metric'] == 'PB_MAE')
                comparisons.append(dict(experiment=experiment, target=target, variant='core7',
                    estimator=name, comparison='saved_Ridge_minus_model_MAE', improvement=ridge_mae-new_mae,
                    CI_level=np.nan, CI_low=np.nan, CI_high=np.nan, valid_bootstrap_draws=0,
                    status='NOT_COMPUTABLE_FROM_SAVED_AGGREGATES', test_participants=20,
                    test_days=j['test_days'],
                    note='Point comparison on verified identical rows; paired Ridge errors were not saved. Ridge not rerun.'))
        copied = original[original.estimator == 'ridge'].copy()
        copied['provenance'] = 'completed_Ridge_aggregate_import_unchanged'
        ridge_metrics.extend(copied.to_dict('records'))
        original_comparison = pd.read_csv(folder/(prefix+'_baseline_comparisons.csv'))
        original_comparison = original_comparison[original_comparison.variant == 'core7'].copy()
        if len(original_comparison) != 2 or set(original_comparison.target) != set(TARGETS):
            raise StopRun('Completed Ridge baseline comparisons are missing/ambiguous.')
        original_comparison['estimator'] = 'ridge'
        original_comparison['provenance'] = 'completed_Ridge_aggregate_import_unchanged'
        ridge_comparisons.extend(original_comparison.to_dict('records'))
    return ridge_metrics, ridge_comparisons, comparisons


def compact_table(metrics, comparisons, jobs, selected):
    rows = []; metric_names = ('PB_MAE', 'PB_RMSE', 'PB_log1p_MAE', 'PB_signed_bias',
        'calibration_intercept', 'calibration_slope', 'mean_within_participant_Spearman',
        'median_within_participant_Spearman', 'Spearman_included_participants', 'Spearman_excluded_participants')
    for experiment in EXPERIMENTS:
        for target in TARGETS:
            j = jobs[(experiment, target)]
            for name in ('constant_median', 'ridge', *MODELS):
                row = dict(experiment=experiment, target=target, estimator=name,
                    analysis_role=('training_derived_constant_baseline' if name == 'constant_median'
                                   else 'completed_confirmatory_reference' if name == 'ridge' and experiment == 'temporal'
                                   else 'completed_prespecified_secondary_reference' if name == 'ridge'
                                   else 'post_hoc_exploratory'),
                    train_days=j['train_days'], train_participants=j['train_participants'],
                    test_days=j['test_days'], test_participants=j['test_participants'],
                    parameters=parameter_json(selected[(experiment, target, name)].parameters) if name in MODELS else '',
                    baseline_median=j['baseline'])
                for metric in metric_names:
                    m = [r for r in metrics if r['experiment'] == experiment and r['target'] == target
                         and r['estimator'] == name and r['metric'] == metric]
                    if len(m) != 1:
                        raise StopRun('Ambiguous/missing compact comparison metric.')
                    row[metric] = m[0]['value']
                    if metric == 'PB_MAE':
                        for key in ('CI_level', 'CI_low', 'CI_high'):
                            row['PB_MAE_'+key] = m[0].get(key, np.nan)
                c = [r for r in comparisons if r['experiment'] == experiment and r['target'] == target
                     and r['estimator'] == name and r['comparison'] == 'constant_minus_model_MAE']
                if name != 'constant_median':
                    if len(c) != 1:
                        raise StopRun('Missing baseline comparison.')
                    for key in ('improvement', 'improvement_pct', 'CI_level', 'CI_low', 'CI_high'):
                        row['baseline_MAE_'+key] = c[0].get(key, np.nan)
                rows.append(row)
    return rows


def reference_preflight(base, folder, original_path):
    if not folder.is_dir() or any(not (folder/name).is_file() for name in REQUIRED_REFERENCE):
        raise StopRun('Completed Ridge directory must contain its original script and all required aggregate files.')
    manifest = json.loads((folder/'run_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') not in ('COMPLETE', 'CORE_COMPLETE_SOME_EXTENSIONS_NOT_ASSESSED'):
        raise StopRun('The reference Ridge run is not marked complete.')
    if manifest.get('script_sha256') != sha256(original_path):
        raise StopRun('Reference manifest and original script hashes differ. No fitting.')
    if manifest.get('seed') != SEED or manifest.get('bootstrap_draws') != NBOOT:
        raise StopRun('Reference seed/bootstrap settings mismatch.')
    if manifest.get('R2_performance_used_for_development') is not False:
        raise StopRun('Reference manifest does not attest R1-only development.')
    fingerprints = pd.read_csv(folder/'00_input_file_fingerprints.csv')
    counts = pd.read_csv(folder/'01_eligibility_counts.csv')
    settings = pd.read_csv(folder/'05_frozen_core_models.csv')
    dictionary = pd.read_csv(folder/'13_feature_dictionary.csv')
    if len(counts) != 6 or not (counts.count_gate == 'PASS').all():
        raise StopRun('Completed Ridge eligibility gate did not pass.')
    snapshots = {name: sha256(folder/name) for name in REQUIRED_REFERENCE}
    return manifest, fingerprints, counts, settings, dictionary, snapshots


def validate_counts_against_reference(counts, reference):
    current = pd.DataFrame(counts)
    for row in current.itertuples(index=False):
        old = reference[(reference.target == row.target) & (reference.cohort == row.cohort)]
        if len(old) != 1 or int(old.days.iloc[0]) != row.days or int(old.participants.iloc[0]) != row.participants:
            raise StopRun('Eligibility population counts differ from completed Ridge. No fitting.')


def study_run(args):
    root = args.data.resolve(); reference = args.ridge_results.resolve()
    original = reference/'mcphases_frozen_analysis.py'
    if not root.is_dir() or args.threads < 1:
        raise StopRun('Dataset inaccessible or invalid thread count.')
    base = load_original(original)
    old_manifest, old_inputs, old_counts, settings, dictionary, old_hashes = reference_preflight(base, reference, original)
    out = (args.out or root.parent/('robustness_analysis_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    if (out.exists() or out == root or root in out.parents or out == reference
            or reference in out.parents or out == root.parent):
        raise StopRun('Output must be a NEW directory outside the raw dataset and completed Ridge directory.')
    out.mkdir(parents=True)
    shutil.copyfile(Path(__file__), out/'mcphases_posthoc_robustness.py')
    metadata = dict(protocol='frozen_post_hoc_HR_steps_robustness', status='STARTED',
        seed=SEED, bootstrap_draws=NBOOT, descriptive_CI_level=.95,
        models=list(MODELS), grid_axes=GRID_AXES, fixed_settings=FIXED,
        candidate_enumeration='itertools.product over axes in the recorded insertion order; rightmost axis varies fastest',
        tie_rule='first numerical tie, rtol=1e-12, atol=1e-12',
        CV_selection='pooled R1 grouped OOF participant-balanced original-unit MAE',
        expected_CV_fits=1260, expected_final_fits=12,
        preprocessing='exact copied original training median/fill0/weighted mean-std/scale<1e-12 -> 1',
        R2_performance_used_for_development=False, all_12_frozen_before_R2_scoring=False,
        Ridge_refitted=False, Ridge_predict_called=False, participant_level_exports=False,
        paired_Ridge_difference_CI='NOT_COMPUTABLE_FROM_SAVED_AGGREGATES; no Ridge predictions rerun',
        script_sha256=sha256(Path(__file__)), original_script_sha256=sha256(original),
        original_canonical_sha256=ORIGINAL_CANONICAL_SHA256,
        reference_directory_name=reference.name,
        reference_file_hashes=old_hashes, reference_package_versions=old_manifest.get('versions', {}),
        versions={p: importlib.metadata.version(p) for p in ('duckdb', 'numpy', 'pandas', 'scipy', 'scikit-learn')},
        python=sys.version.split()[0], implementation_conventions=__doc__)
    base.atomic_json(metadata, out/'run_manifest.json')
    c = None; diagnostics = Diagnostics(base, out)
    try:
        paths = collections.defaultdict(list)
        for path in root.rglob('*'):
            if path.is_file() and path.name.lower().endswith(('.csv', '.csv.gz')):
                paths[base.stem(path)].append(path)
        fingerprints = []
        for table in ('hormones_and_selfreport', 'heart_rate', 'steps'):
            if len(paths.get(table, [])) != 1:
                raise StopRun('Missing or ambiguous core table. No fitting.')
            path = paths[table][0]; current_hash = sha256(path)
            previous = old_inputs[old_inputs.table == table]
            if (len(previous) != 1 or str(previous.sha256.iloc[0]) != current_hash
                    or int(previous.bytes.iloc[0]) != path.stat().st_size):
                raise StopRun('Core input bytes differ from completed Ridge. No fitting.')
            fingerprints.append(dict(table=table, bytes=path.stat().st_size, sha256=current_hash))
        base.atomic_csv(fingerprints, out/'00_input_file_fingerprints.csv')
        c = base.connect(args.memory_limit, args.threads)
        targets, members, hormone_quality = base.hormones(c, paths)
        hr, hr_quality = base.wearable(c, paths, 'heart_rate')
        steps, step_quality = base.wearable(c, paths, 'steps')
        daily = hr.merge(steps, on=['_id', '_r', '_d'], how='outer', validate='one_to_one')
        cohorts, histories, counts, only, passed = count_gate(base, targets, members, daily)
        base.atomic_csv(counts, out/'01_eligibility_counts.csv')
        base.atomic_csv(hr_quality+step_quality, out/'02_cleaning_aggregates.csv')
        base.atomic_csv(hormone_quality, out/'03_target_integrity.csv')
        if not passed:
            raise StopRun('Approved eligibility/membership count mismatch. No fitting performed; do not change eligibility.')
        validate_counts_against_reference(counts, old_counts)
        jobs, schemas, fold_counts = build_jobs(base, cohorts, histories, members, only, settings, dictionary)
        base.atomic_csv(schemas, out/'12_feature_dictionary.csv')
        base.atomic_csv(fold_counts, out/'13_R1_fold_aggregate_counts.csv')
        metadata['status'] = 'COUNT_SOURCE_SCHEMA_FOLD_GATES_PASSED'
        base.atomic_json(metadata, out/'run_manifest.json')
        selected = {}; cvrows = []; frozen = []
        # No R2 predictions, evaluation or completed Ridge metrics read here.
        for experiment in EXPERIMENTS:
            for target in TARGETS:
                j = jobs[(experiment, target)]
                for name in MODELS:
                    model, cv, chosen = train_search(base, j['Xtrain'], j['ytrain'], j['ids'],
                        j['folds'], name, experiment, target, diagnostics)
                    selected[(experiment, target, name)] = model; cvrows.extend(cv)
                    frozen.append(dict(experiment=experiment, target=target, estimator=name,
                        candidate_index=chosen, parameters=parameter_json(model.parameters),
                        fixed_parameters=parameter_json(FIXED[name]), train_days=j['train_days'],
                        train_participants=j['train_participants'], test_days=j['test_days'],
                        test_participants=j['test_participants'], features=j['features'], baseline_median=j['baseline']))
                    base.atomic_csv(cvrows, out/'05_R1_cross_validation.csv')
                    base.atomic_csv(frozen, out/'06_frozen_exploratory_models.csv')
        if len(selected) != 12:
            raise StopRun('All 12 final pipelines must be frozen before scoring.')
        metadata.update(status='ALL_12_EXPLORATORY_PIPELINES_FROZEN_BEFORE_R2_SCORING',
                        all_12_frozen_before_R2_scoring=True, final_fit_count=len(selected),
                        CV_fit_count=sum(r['phase'] == 'R1_CV' for r in diagnostics.rows))
        base.atomic_json(metadata, out/'run_manifest.json')
        print('All 12 exploratory pipelines frozen. R2 scoring starts now.', flush=True)
        metrics = []; comparisons = []
        for experiment in EXPERIMENTS:
            for target in TARGETS:
                j = jobs[(experiment, target)]
                predictions = {name: selected[(experiment, target, name)].predict(j['Xtest']) for name in MODELS}
                mr, cr = evaluate_all(base, j['test'], predictions, j['baseline'], experiment, target)
                metrics.extend(mr); comparisons.extend(cr)
                base.atomic_csv(metrics, out/'07_exploratory_metrics.csv')
                base.atomic_csv(comparisons, out/'08_exploratory_baseline_comparisons.csv')
        ridge_metrics, ridge_comparisons, ridge_point = load_completed_results(base, reference, jobs, metrics)
        base.atomic_csv(ridge_metrics, out/'09_completed_Ridge_metrics_unchanged.csv')
        base.atomic_csv(ridge_comparisons, out/'09_completed_Ridge_baseline_comparisons_unchanged.csv')
        base.atomic_csv(ridge_point, out/'10_Ridge_point_comparisons.csv')
        comparison_table = compact_table(metrics+ridge_metrics, comparisons+ridge_comparisons, jobs, selected)
        base.atomic_csv(comparison_table, out/'11_compact_comparison.csv')
        stability = [dict(kind='raw_core_table', name=r['table'],
                          unchanged=sha256(paths[r['table']][0]) == r['sha256']) for r in fingerprints]
        stability.extend(dict(kind='completed_Ridge_file', name=name,
                              unchanged=sha256(reference/name) == digest) for name, digest in old_hashes.items())
        base.atomic_csv(stability, out/'14_input_and_Ridge_stability.csv')
        if not all(r['unchanged'] for r in stability):
            raise StopRun('Input/reference changed during analysis; this run is invalid.')
        metadata.update(status='COMPLETE_EXPLORATORY_WITH_RIDGE_PAIRED_CI_UNAVAILABLE',
                        all_searches_complete=True, raw_and_Ridge_files_unchanged=True)
        base.atomic_json(metadata, out/'run_manifest.json')
        (out/'README.txt').write_text(__doc__+'\n\n'
            '01: counts asserted before any estimator fit. 04: fit/convergence diagnostics.\n'
            '05: all 63 frozen candidates per target/experiment and R1-only pooled CV scores.\n'
            '06: all 12 final choices saved before ANY R2 scoring.\n'
            '07-08: descriptive metrics and PAIRED constant-baseline comparisons (95%, 10,000 draws).\n'
            '09: completed Ridge aggregates imported unchanged, including its original CI levels.\n'
            '10: Ridge-minus-exploratory MAE point differences; paired CIs unavailable from old aggregates.\n'
            '11: 20 rows, all five estimators for both targets and both experiments.\n'
            '12-14: feature order, aggregate fold counts and raw/reference stability checks.\n'
            'Undefined calibration/correlation stays empty/NaN, never zero.\n'
            'All new results are POST HOC EXPLORATORY; no selection of a R2 winner,\n'
            'no change to the confirmatory Ridge claim and no independent confirmation.\n'
            'No participant identifiers, row predictions, participant errors, coefficients or model files exported.\n', encoding='utf-8')
        print('Finished. Aggregate robustness outputs saved in the new output directory.', flush=True)
    except Exception as error:
        diagnostics.save()
        metadata.update(status='STOPPED_SEARCH_INCOMPLETE_OR_RUN_INVALID',
            all_searches_complete=False, reason=str(error) if isinstance(error, StopRun)
            else 'Execution error; raw exception suppressed to protect restricted data.')
        base.atomic_json(metadata, out/'run_manifest.json')
        raise StopRun(metadata['reason']) from None
    finally:
        if c is not None:
            c.close()


def self_test(original):
    """Full frozen searches, synthetic data only; no raw study/Ridge outputs read."""
    base = load_original(original)
    # Even an accidental call into the original Ridge fit/train/predict must fail.
    def forbidden(*args, **kwargs):
        raise AssertionError('Ridge training/prediction forbidden in robustness tests.')
    base.Ridge = forbidden
    base.train = forbidden
    base.Pipeline.predict = forbidden
    assert [len(candidates(name)) for name in MODELS] == [15, 36, 12]
    assert candidates('elastic_net')[0] == {'alpha': 1.0, 'l1_ratio': 0.9}
    assert candidates('rbf_svr')[-1] == {'C': 100.0, 'epsilon': 0.05, 'gamma': 'scale'}
    assert candidates('random_forest')[-1] == {'max_depth': None, 'min_samples_leaf': 5, 'max_features': 0.5}
    assert first_tied_candidate([2.0, 2.0, 3.0]) == 0
    assert first_tied_candidate([2.0+5e-13, 2.0]) == 0
    print('Synthetic grid/enumeration/tie checks PASS.', flush=True)

    # Execute ONLY the preprocessing prefix of the original fit AST, never its
    # Ridge model assignment. Compare all arrays bit-for-bit against the copy.
    source = ast.parse(textwrap.dedent(inspect.getsource(base.Pipeline.fit)))
    method = source.body[0]; prefix = []
    for statement in method.body:
        if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == 'model' for t in statement.targets):
            break
        prefix.append(statement)
    method.body = prefix+[ast.Return(value=ast.Name(id='self', ctx=ast.Load()))]
    ast.fix_missing_locations(source)
    scope = {'np': np, 'warnings': warnings, 'weights': base.weights}
    exec(compile(source, '<original_preprocessing_only>', 'exec'), scope)
    rng = np.random.default_rng(671)
    ids = np.repeat(np.array(['synthetic_a', 'synthetic_b', 'synthetic_c', 'synthetic_d', 'synthetic_e']), [31, 11, 19, 7, 23])
    X = rng.normal(size=(len(ids), 8)); X[:, 3] = np.nan; X[:, 4] = 7
    X[:20, 2] = np.nan; y = np.expm1(1.8+.12*X[:, 0]+rng.normal(0, .1, len(ids)))
    oracle = scope['fit'](types.SimpleNamespace(), X, y, ids)
    for name in MODELS:
        pipe = ExploratoryPipeline(base, name, candidates(name)[0]).fit(X, y, ids)
        for attribute in ('fill', 'mean', 'scale'):
            assert np.array_equal(getattr(pipe, attribute), getattr(oracle, attribute))
        assert pipe.fill[3] == 0 and pipe.scale[3] == 1 and pipe.scale[4] == 1
        before = tuple(getattr(pipe, key).copy() for key in ('fill', 'mean', 'scale'))
        pipe.predict(np.nan_to_num(X, nan=0)+.01)
        assert all(np.array_equal(a, getattr(pipe, b)) for a, b in zip(before, ('fill', 'mean', 'scale')))
    assert np.isclose(base.pbmean(np.array([0., 0., 0., 10.]), np.array(['a', 'a', 'a', 'b'])), 5)
    print('Synthetic exact-preprocessing/equal-participant-weight checks PASS.', flush=True)

    # Original raw-file cleaning on a tiny, explicitly synthetic local fixture.
    with tempfile.TemporaryDirectory(prefix='mcphases_synthetic_') as directory:
        directory = Path(directory); paths = collections.defaultdict(list)
        for table, field in [('heart_rate', 'bpm'), ('steps', 'steps')]:
            rows = [dict(id='SYNTHETIC_ONLY', study_interval='2022', day_in_study=d,
                         timestamp='12:00:00', **{field: 60+d if table == 'heart_rate' else 100*d})
                    for d in range(1, 13)]
            rows.append(rows[0].copy())  # exact duplicate must not double steps
            conflict = rows[3].copy(); conflict[field] += 1; rows.append(conflict)
            path = directory/(table+'.csv'); pd.DataFrame(rows).to_csv(path, index=False)
            paths[table].append(path)
        connection = base.connect('256MB', 1)
        try:
            hr, aq = base.wearable(connection, paths, 'heart_rate')
            steps, bq = base.wearable(connection, paths, 'steps')
        finally:
            connection.close()
        assert steps.loc[steps._d == 1, 'steps_total'].iloc[0] == 100
        assert 4 not in set(hr._d) and 4 not in set(steps._d)
        assert aq[0]['conflicting_record_keys'] == bq[0]['conflicting_record_keys'] == 1
        daily = hr.merge(steps, on=['_id', '_r', '_d'], how='outer', validate='one_to_one')
        times = pd.DataFrame([('SYNTHETIC_ONLY', 'R1', 8), ('SYNTHETIC_ONLY', 'R1', 12),
                              ('SYNTHETIC_ONLY', 'R2', 12)], columns=['_id', '_r', '_d'])
        history = base.histories(daily, times, base.DAILY_CORE)
        assert not np.isfinite(history[0]).all() and np.isfinite(history[1]).all() and np.isnan(history[2]).all()
        prior = history[1].copy(); daily.loc[daily._d == 12, 'hr_median'] = 999999
        assert np.array_equal(prior, base.histories(daily, times.iloc[[1]], base.DAILY_CORE)[0])
        feature, _ = base.features(history[1:2], base.DAILY_CORE, 7)
        assert feature.hr_median_last.iloc[0] == 71
    print('Synthetic exact-duplicate/conflict, gap, round and same-day exclusion checks PASS.', flush=True)

    # Build the APPROVED counts with synthetic identifiers and measurements.
    # This verifies the positive count gate as well as the fail-before-fit gate.
    members = {'R1': {f'S{i:02}' for i in range(42)}, 'R2': {f'S{i:02}' for i in range(22, 42)}}
    daily_rows = []; target_rows = []
    for i in range(42):
        pid = f'S{i:02}'; days = (65 if i < 21 else (86 if i == 21 else (76 if i < 41 else 86)))
        for round_name, first, n in [('R1', 1, days)]+([('R2', 905, 83 if i < 41 else 100)] if i >= 22 else []):
            for day in range(first, first+n+7):
                daily_rows.append((pid, round_name, day, 70+math.sin(day), 4+math.cos(day), 5000+100*math.sin(day/2)))
            for day in range(first+7, first+n+7):
                target_rows.append((pid, round_name, day, 10+math.sin(day)))
    synthetic_daily = pd.DataFrame(daily_rows, columns=['_id', '_r', '_d']+base.DAILY_CORE)
    lh = pd.DataFrame(target_rows, columns=['_id', '_r', '_d', 'y'])
    e3g = lh.copy(); e3g.y = e3g.y*10
    remove = (e3g._id == 'S41') & (e3g._r == 'R1') & (e3g._d == 93)
    assert remove.sum() == 1; e3g = e3g[~remove].copy()
    targets = {'LH': lh, 'E3G': e3g}
    cohorts, histories, counts, only, passed = count_gate(base, targets, members, synthetic_daily)
    assert passed and all(row['count_gate'] == 'PASS' for row in counts)
    mutated = {'LH': lh.iloc[1:].copy(), 'E3G': e3g}
    assert count_gate(base, mutated, members, synthetic_daily)[-1] is False
    # Independently reconstruct the original seeded participant assignment.
    for people in (members['R1'], only):
        ordered = np.array(sorted(people), dtype=object); np.random.default_rng(SEED).shuffle(ordered)
        assert base.foldmap(people) == {pid: int(i % 5) for i, pid in enumerate(ordered)}
    example, _ = base.features(histories['LH'], base.DAILY_CORE, 7)
    dictionary = pd.DataFrame(dict(variant='core7', feature=example.columns))
    settings = []
    for experiment in EXPERIMENTS:
        for target, data in cohorts.items():
            mask = (data._r == 'R1') & (data._id.isin(only) if experiment == 'independent' else True)
            tr = data[mask]; te = data[data._r == 'R2']
            settings.append(dict(experiment=experiment, target=target, variant='core7',
                train_days=len(tr), train_participants=tr._id.nunique(), test_days=len(te),
                test_participants=te._id.nunique(), features=example.shape[1],
                baseline_median=base.wmedian(tr.y.to_numpy(float), tr._id.to_numpy())))
    _, _, fold_counts = build_jobs(base, cohorts, histories, members, only, pd.DataFrame(settings), dictionary)
    assert all(row['participant_overlap'] == 0 for row in fold_counts)
    print('Synthetic 2981/2980/1451/1677 count gates and exact fold reconstruction PASS.', flush=True)

    # Full 1,260 CV fits + 12 final fits use SMALL synthetic matrices. Forests
    # retain the frozen 300 trees and all other settings; no reduced test grid.
    train_ids = np.concatenate([np.repeat(f'S{i:02}', 5+i % 4) for i in range(42)])
    train_history = np.empty((len(train_ids), 7, 3))
    train_history[:, :, 0] = rng.normal(72, 5, (len(train_ids), 7))
    train_history[:, :, 1] = rng.uniform(2, 8, (len(train_ids), 7))
    train_history[:, :, 2] = np.exp(rng.normal(8.4, .4, (len(train_ids), 7)))
    train_X, _ = base.features(train_history, base.DAILY_CORE, 7); train_X = train_X.to_numpy()
    log_lh = 1.7+.03*(train_history[:, -1, 0]-72)+rng.normal(0, .08, len(train_ids))
    log_e3g = 4.6+.1*(train_history[:, -1, 1]-5)+rng.normal(0, .08, len(train_ids))
    synthetic_targets = {'LH': np.expm1(log_lh), 'E3G': np.expm1(log_e3g)}
    test_ids = np.repeat(np.array(sorted(members['R2'])), 12)
    test_history = np.empty((len(test_ids), 7, 3))
    test_history[:, :, 0] = rng.normal(72, 5, (len(test_ids), 7))
    test_history[:, :, 1] = rng.uniform(2, 8, (len(test_ids), 7))
    test_history[:, :, 2] = np.exp(rng.normal(8.4, .4, (len(test_ids), 7)))
    test_X, _ = base.features(test_history, base.DAILY_CORE, 7); test_X = test_X.to_numpy()
    synthetic_test_y = {'LH': np.expm1(1.7+.03*(test_history[:, -1, 0]-72)),
                        'E3G': np.expm1(4.6+.1*(test_history[:, -1, 1]-5))}
    # Altered R2 outcomes cannot reach the tuning API; verify they are not an
    # argument, then verify scoring leaves all frozen settings/scalers unchanged.
    assert 'test' not in inspect.signature(train_search).parameters
    diagnostics = Diagnostics(base); fitted = {}; all_cv = []; small_jobs = {}
    for experiment in EXPERIMENTS:
        keep = np.isin(train_ids, list(only)) if experiment == 'independent' else np.ones(len(train_ids), dtype=bool)
        folds = base.foldmap(only if experiment == 'independent' else members['R1'])
        if experiment == 'independent':
            assert not (set(train_ids[keep]) & set(test_ids))
        for target in TARGETS:
            y = synthetic_targets[target][keep]; ids = train_ids[keep]; X = train_X[keep]
            data = pd.DataFrame(dict(_id=test_ids, _r='R2', _d=np.tile(np.arange(912, 924), 20),
                                     y=synthetic_test_y[target]))
            small_jobs[(experiment, target)] = dict(test=data, Xtest=test_X,
                baseline=base.wmedian(y, ids), train_days=len(y), train_participants=len(set(ids)),
                test_days=len(data), test_participants=20)
            for name in MODELS:
                pipe, cv, chosen = train_search(base, X, y, ids, folds, name, experiment, target, diagnostics)
                assert chosen == first_tied_candidate([r['R1_grouped_OOF_PB_MAE'] for r in cv])
                fitted[(experiment, target, name)] = pipe; all_cv.extend(cv)
    assert len(fitted) == 12 and len(all_cv) == 252
    assert sum(row['phase'] == 'R1_CV' for row in diagnostics.rows) == 1260
    assert len(diagnostics.rows) == 1272 and all(row['status'] == 'OK' for row in diagnostics.rows)
    print('Synthetic ALL 12 frozen searches (1,260 CV fits, 12 final fits) PASS.', flush=True)

    # Bootstrap algebra agrees with original evaluate; label/level are changed
    # only for exploratory reporting. Its input predictions are NOT Ridge.
    metrics = []; comparisons = []; fake_reference = []; fake_comparisons = []
    for experiment in EXPERIMENTS:
        for target in TARGETS:
            j = small_jobs[(experiment, target)]
            predictions = {name: fitted[(experiment, target, name)].predict(test_X) for name in MODELS}
            mr, cr = evaluate_all(base, j['test'], predictions, j['baseline'], experiment, target)
            metrics.extend(mr); comparisons.extend(cr)
            original_metrics, _ = base.evaluate(j['test'], predictions['elastic_net'], j['baseline'], experiment, target, 'core7')
            for metric in ('PB_MAE', 'PB_RMSE', 'PB_log1p_MAE', 'PB_signed_bias',
                           'calibration_intercept', 'calibration_slope', 'mean_within_participant_Spearman'):
                old = next(r for r in original_metrics if r['estimator'] == 'ridge' and r['metric'] == metric)
                current = next(r for r in mr if r['estimator'] == 'elastic_net' and r['metric'] == metric)
                assert all(np.isclose(old[key], current[key], equal_nan=True) for key in ('value', 'CI_low', 'CI_high'))
            for name in MODELS:
                pipe = fitted[(experiment, target, name)]
                before = (pipe.mean.copy(), pipe.scale.copy(), pipe.parameters.copy())
                changed = j['test'].copy(); changed.y *= 100
                evaluate_all(base, changed, predictions, j['baseline'], experiment, target)
                assert np.array_equal(before[0], pipe.mean) and np.array_equal(before[1], pipe.scale) and before[2] == pipe.parameters
            # Use synthetic oracle aggregates to test the IMPORT path, not a
            # fitted Ridge. These are fixture files and never study findings.
            copied = [dict(r, estimator='ridge', provenance='SYNTHETIC_IMPORT_FIXTURE') for r in mr if r['estimator'] == 'elastic_net']
            fake_reference.extend(copied)
            fake_comparisons.extend([dict(r, estimator='ridge') for r in cr if r['estimator'] == 'elastic_net'])
    with tempfile.TemporaryDirectory(prefix='mcphases_synthetic_reference_') as directory:
        directory = Path(directory)
        for experiment, prefix in [('temporal', '06_confirmatory'), ('independent', '07_participant_independent')]:
            fixture_metrics = [r for r in fake_reference+metrics if r['experiment'] == experiment
                               and r['estimator'] in ('ridge', 'constant_median')]
            base.atomic_csv(fixture_metrics, directory/(prefix+'_metrics.csv'))
            base.atomic_csv([r for r in fake_comparisons if r['experiment'] == experiment], directory/(prefix+'_baseline_comparisons.csv'))
        imported, imported_comps, point_comps = load_completed_results(base, directory, small_jobs, metrics)
        assert all(r['status'] == 'NOT_COMPUTABLE_FROM_SAVED_AGGREGATES' for r in point_comps)
        assert all(np.isnan(r['CI_low']) and r['valid_bootstrap_draws'] == 0 for r in point_comps)
        table = compact_table(metrics+imported, comparisons+imported_comps, small_jobs, fitted)
        assert len(table) == 20 and not any(key in row for row in table for key in ('id', '_id', '_d', 'prediction'))
        base.atomic_csv(table, directory/'comparison.csv')
        exported = (directory/'comparison.csv').read_text(encoding='utf-8')
        assert not any(pid in exported for pid in members['R1'])
    # Ensure a failed candidate and nonfinite predictions stop the search.
    saved_fit = ExploratoryPipeline.fit
    def failing_fit(self, *args, **kwargs):
        self.diagnostic = {'status': 'SYNTHETIC_FORCED_FAILURE'}
        raise StopRun('Synthetic forced fit failure.')
    ExploratoryPipeline.fit = failing_fit
    failed = Diagnostics(base)
    try:
        try:
            train_search(base, train_X, synthetic_targets['LH'], train_ids, base.foldmap(members['R1']),
                         'elastic_net', 'temporal', 'LH', failed)
            raise AssertionError('Failed candidate was silently accepted.')
        except StopRun:
            assert len(failed.rows) == 1 and failed.rows[0]['search_complete'] is False
    finally:
        ExploratoryPipeline.fit = saved_fit
    pipe = fitted[('temporal', 'LH', 'elastic_net')]
    saved_predict = pipe.model.predict
    pipe.model.predict = lambda X: np.full(len(X), np.nan)
    try:
        try:
            pipe.predict(test_X)
            raise AssertionError('Nonfinite predictions were silently accepted.')
        except StopRun:
            pass
    finally:
        pipe.model.predict = saved_predict
    print('Synthetic bootstrap/import/no-Ridge-rerun, privacy and failure-stop checks PASS.', flush=True)
    print('SELF-TEST PASS. Only synthetic models ran. No restricted dataset or completed Ridge results accessed.', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, default=Path(r'D:\New paper\mcphases'))
    ap.add_argument('--ridge-results', type=Path, default=Path(r'D:\New paper\frozen_analysis_20261008_192018'))
    ap.add_argument('--out', type=Path)
    ap.add_argument('--memory-limit', default='4GB')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--self-test', action='store_true', help='Synthetic data only; no study data or Ridge results read.')
    ap.add_argument('--original-script', type=Path, help='For --self-test only; defaults to the reference folder copy.')
    args = ap.parse_args()
    if args.self_test:
        self_test(args.original_script or args.ridge_results/'mcphases_frozen_analysis.py')
    elif args.original_script is not None:
        raise StopRun('--original-script is for synthetic self-test only; real runs use the verified reference copy.')
    else:
        study_run(args)


if __name__ == '__main__':
    try:
        main()
    except StopRun as error:
        raise SystemExit(str(error))
    except KeyboardInterrupt:
        raise SystemExit('Stopped by user. The run is incomplete; only aggregate checkpoints were saved.')
    except Exception:
        raise SystemExit('Setup/execution error; raw exception suppressed. Check dependencies, paths and manifest.')
