import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest


def get_baseline_path(baseline_file_name, backend, version):
    """
    Construct the full path to a baseline synthesis report file.

    Args:
        baseline_file_name (str): The name of the baseline report file.
        backend (str): The backend used (e.g., 'Vivado', 'Vitis').
        version (str): The tool version (e.g., '2020.1').

    Returns:
        Path: A pathlib.Path object pointing to the baseline file location.
    """
    return Path(__file__).parent / 'baselines' / backend / version / baseline_file_name


def save_report(data, filename):
    """
    Save synthesis data to a JSON file in the same directory as this script.

    Args:
        data (dict): The synthesis output data to be saved.
        filename (str): The filename to write to (e.g., 'synthesis_report_test_x.json').

    Raises:
        OSError: If the file cannot be written.
    """
    out_path = Path(__file__).parent / filename
    with open(out_path, 'w') as fp:
        json.dump(data, fp, indent=4)


def compare_dicts(data, baseline, tolerances):
    """
    Compare two flat dictionaries with tolerances.

    Args:
        report (dict): The generated report dictionary.
        baseline (dict): The expected/baseline dictionary.
        tolerances (dict): Dictionary of tolerances per key.

    Raises:
        AssertionError: If values differ outside the allowed tolerance.
    """
    for key, expected in baseline.items():
        actual = data.get(key)
        tolerance = tolerances.get(key, 0.02)  # Default tolerance of 1%

        try:
            actual = float(actual)
            expected = float(expected)
            assert actual == pytest.approx(expected, rel=tolerance), (
                f'{key}: expected {expected}, got {actual} (tolerance={tolerance * 100}%)'
            )
        except ValueError:
            assert actual == expected, f"{key}: expected '{expected}', got '{actual}'"


def compare_vitis_backend(data, baseline):
    """
    Compare reports from Vivado/Vitis backends.

    Args:
        data (dict): The current synthesis report.
        baseline (dict): The expected synthesis report.
    """

    tolerances = {
        'TargetClockPeriod': 0.01,
        'EstimatedClockPeriod': 0.01,
        'BestLatency': 0.02,
        'WorstLatency': 0.02,
        'IntervalMin': 0.02,
        'IntervalMax': 0.02,
        'FF': 0.1,
        'LUT': 0.1,
        'BRAM_18K': 0.1,
        'DSP': 0.1,
        'URAM': 0.1,
        'AvailableBRAM_18K': 0.1,
        'AvailableDSP': 0.1,
        'AvailableFF': 0.1,
        'AvailableLUT': 0.1,
        'AvailableURAM': 0.1,
    }

    compare_dicts(data['CSynthesisReport'], baseline['CSynthesisReport'], tolerances)


def compare_oneapi_backend(data, baseline):
    """
    Compare reports from the oneAPI backend.

    Args:
        data (dict): The current synthesis report.
        baseline (dict): The expected synthesis report.
    """

    tolerances = {
        'HLS': {
            'total': {'alut': 0.1, 'reg': 0.1, 'ram': 0.1, 'dsp': 0.1, 'mlab': 0.1},
            'available': {'alut': 0.1, 'reg': 0.1, 'ram': 0.1, 'dsp': 0.1, 'mlab': 0.1},
        },
        'Loop': {'worstFrequency': 0.1, 'worstII': 0.1, 'worstLatency': 0.1},
    }

    data = data['report']
    baseline = baseline['report']

    compare_dicts(data['HLS']['total'], baseline['HLS']['total'], tolerances['HLS']['total'])
    compare_dicts(data['HLS']['available'], baseline['HLS']['available'], tolerances['HLS']['available'])
    compare_dicts(data['Loop'], baseline['Loop'], tolerances['Loop'])


COMPARE_FUNCS = {
    'Vivado': compare_vitis_backend,
    'VivadoAccelerator': compare_vitis_backend,
    'Vitis': compare_vitis_backend,
    'oneAPI': compare_oneapi_backend,
}


EXPECTED_REPORT_KEYS = {
    'Vivado': {'CSynthesisReport'},
    'VivadoAccelerator': {'CSynthesisReport'},
    'Vitis': {'CSynthesisReport'},
    'oneAPI': {'report'},
}

IMPLEMENTATION_EXPECTED_REPORT_KEYS = {
    'VivadoAccelerator': {'CSynthesisReport', 'VivadoSynthReport', 'TimingReport'},
}

BITFILE_REQUIRED_BACKENDS = {'VivadoAccelerator'}

IMPLEMENTATION_REQUIRED_METADATA_FIELDS = {
    'VivadoAccelerator': {'board', 'part'},
}

DEFAULT_IMPLEMENTATION_DATASET_DIR = Path(__file__).parent / 'implementation'
IMPLEMENTATION_DATASET_DIR_ENV = 'IMPLEMENTATION_DATASET_DIR'


def _resolve_commit_sha():
    commit_sha = os.getenv('CI_COMMIT_SHA')
    if commit_sha:
        return commit_sha

    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def _collect_bitfiles(output_dir):
    output_path = Path(output_dir)
    return sorted(str(path.relative_to(output_path)) for path in output_path.rglob('*.bit'))


def _save_implementation_dataset(data, test_case_id):
    dataset_dir = Path(os.getenv(IMPLEMENTATION_DATASET_DIR_ENV, str(DEFAULT_IMPLEMENTATION_DATASET_DIR)))
    dataset_dir.mkdir(parents=True, exist_ok=True)
    out_path = dataset_dir / f'implementation_dataset_{test_case_id}.json'
    with open(out_path, 'w') as fp:
        json.dump(data, fp, indent=4, sort_keys=True)


def _validate_implementation_metadata(backend, metadata):
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise AssertionError('Implementation collection metadata must be a dictionary.')

    required_fields = IMPLEMENTATION_REQUIRED_METADATA_FIELDS.get(backend, set())
    missing_fields = sorted(field for field in required_fields if not metadata.get(field))
    if missing_fields:
        raise AssertionError(
            f'Missing required metadata for backend {backend}: {missing_fields}. '
            f'Provided metadata keys: {sorted(metadata.keys())}'
        )

    return metadata


def run_synthesis_test(config, hls_model, baseline_file_name, backend):
    """
    Run HLS synthesis and compare the output with a stored baseline report.

    If synthesis is disabled via the configuration (`run_synthesis=False`),
    no synthesis is executed and the method silently returns.

    Args:
        config (dict): Test-wide synthesis configuration fixture.
        hls_model (object): hls4ml model instance to build and synthesize.
        baseline_file_name (str): The name of the baseline file for comparison.
        backend (str): The synthesis backend used (e.g., 'Vivado', 'Vitis').
    """
    if not config.get('run_synthesis', False):
        return

    # Skip Quartus backend
    if backend == 'Quartus':
        return

    # Run synthesis
    build_args = config['build_args']
    try:
        data = hls_model.build(**build_args.get(backend, {}))
    except Exception as e:
        pytest.fail(f'hls_model.build failed: {e}')

    # Save synthesis report
    save_report(data, f'synthesis_report_{baseline_file_name}')

    # Check synthesis report keys
    expected_keys = EXPECTED_REPORT_KEYS.get(backend, set())
    assert data and expected_keys.issubset(data.keys()), (
        f'Synthesis failed: Missing expected keys in synthesis report: expected {expected_keys}, got {set(data.keys())}'
    )

    # Load baseline report
    version = config['tools_version'].get(backend)
    baseline_path = get_baseline_path(baseline_file_name, backend, version)
    try:
        with open(baseline_path) as fp:
            baseline = json.load(fp)
    except FileNotFoundError:
        pytest.fail(f"Baseline file '{baseline_path}' not found.")

    # Compare report against baseline using backend-specific rules
    compare_func = COMPARE_FUNCS.get(backend)
    if compare_func is None:
        raise AssertionError(f'No comparison function defined for backend: {backend}')

    compare_func(data, baseline)


def run_implementation_collection_test(config, hls_model, test_case_id, backend, metadata=None):
    """
    Run an implementation-oriented backend build and write a dataset artifact.

    This helper is intended for implementation collection tests (not baseline comparison tests).
    It runs the backend build using implementation-specific build arguments, validates that
    expected report sections exist, optionally validates backend-specific output artifacts
    (e.g. bitfile presence), and saves a JSON dataset artifact with report data + metadata.

    Args:
        config (dict): Test configuration fixture, expected to contain tool versions and build args.
        hls_model (object): hls4ml model instance to build.
        test_case_id (str): Unique test identifier used in output dataset filename.
        backend (str): Backend name (e.g. 'VivadoAccelerator').
        metadata (dict, optional): Backend metadata.
            Required fields are backend-specific and validated by
            ``IMPLEMENTATION_REQUIRED_METADATA_FIELDS``.
            Example for VivadoAccelerator: ``{'board': 'zcu102', 'part': 'xczu9eg-ffvb1156-2-e'}``.

    Raises:
        AssertionError: If required report keys are missing, required metadata is missing,
            metadata type is invalid, or required output artifacts (e.g. bitfile) are missing.
        pytest.fail: If backend build execution fails.
    """
    build_args = config.get('implementation_build_args', config.get('build_args', {}))
    try:
        report = hls_model.build(**build_args.get(backend, {}))
    except Exception as e:
        pytest.fail(f'hls_model.build failed: {e}')

    expected_keys = IMPLEMENTATION_EXPECTED_REPORT_KEYS.get(backend, set())
    assert report and expected_keys.issubset(report.keys()), (
        f'Implementation collection failed: Missing expected keys in report: '
        f'expected {expected_keys}, got {set(report.keys()) if report else set()}'
    )

    bitfiles = []
    if backend in BITFILE_REQUIRED_BACKENDS:
        bitfiles = _collect_bitfiles(hls_model.config.get_output_dir())
        assert bitfiles, 'Bitfile generation failed: no .bit file was found in the output directory.'

    metadata = _validate_implementation_metadata(backend, metadata)

    dataset_metadata = {
        'test_id': test_case_id,
        'backend': backend,
        'tool_version': config.get('tools_version', {}).get(backend, 'unknown'),
        'commit_sha': _resolve_commit_sha(),
        'collected_at_utc': datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
        'bitfiles': bitfiles,
    }
    dataset_metadata.update(metadata)

    dataset = {'metadata': dataset_metadata, 'report': report}
    _save_implementation_dataset(dataset, test_case_id)
