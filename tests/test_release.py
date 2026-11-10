import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import reproduce
from mellm_metrics import focal_terms

class ReleaseTests(unittest.TestCase):

    def test_focal_derivative(self):
        p = np.linspace(-4, 4, 12)
        y = np.tile([0, 1], 6)
        step = 1e-05
        _, g, h = focal_terms(p, y, 0.75, 0.75)
        plus = focal_terms(p + step, y, 0.75, 0.75)[0]
        minus = focal_terms(p - step, y, 0.75, 0.75)[0]
        np.testing.assert_allclose(g, (plus - minus) / (2 * step), rtol=1e-06, atol=1e-08)
        h_numeric = (focal_terms(p + step, y, 0.75, 0.75)[1] - focal_terms(p - step, y, 0.75, 0.75)[1]) / (2 * step)
        np.testing.assert_allclose(h, h_numeric, rtol=1e-06, atol=1e-08)

    def test_formulas_and_time_boundaries(self):
        raw = reproduce.load_panel(ROOT / 'data/synthetic_demo.csv')
        X, key = reproduce.build_matrix(raw, 'final')
        self.assertEqual(X.shape, (450, 125))
        changed = raw.copy()
        changed.loc[changed['year'].ge(2022), reproduce.BASE] *= 10
        X2, _ = reproduce.build_matrix(changed, 'final')
        np.testing.assert_allclose(X[key['year'].le(2021)], X2[key['year'].le(2021)], equal_nan=True)

    def test_common_selection_rule_allows_different_counts(self):
        from unittest.mock import Mock, patch
        from types import SimpleNamespace
        sys.path.insert(0, str(ROOT / 'workflow'))
        import finalize
        for pool_count, full_auc, expected in [(91, .61, 18), (100, .75, 100)]:
            with self.subTest(pool_count=pool_count), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                pool = SimpleNamespace(exprs=[f'F{i}' for i in range(pool_count)], weights=[1.] * pool_count)
                def score(expressions, record=False):
                    count = len(expressions)
                    auc = full_auc if count == pool_count else .5 + .01 * min(count, 18)
                    return {'validation': {'AUC': auc}}, None
                evaluator = Mock()
                evaluator.score.side_effect = score
                config = {'numeric_workers': 2, 'selection': {'min_factors': 10, 'max_factors': 20}}
                with patch.object(finalize, 'RUN', run), patch.object(finalize, 'status'):
                    chosen, _ = finalize.select(evaluator, pool, config)
                self.assertEqual(len(chosen), expected)
                self.assertEqual(finalize.read(run / 'final_selection.json')['selection'], 'selected' if expected == 18 else 'full')

    def test_legacy_workflow_imports_without_network(self):
        import socket
        from unittest.mock import patch
        sys.path.insert(0, str(ROOT / 'workflow'))
        with patch.object(socket.socket, 'connect', side_effect=AssertionError('Unexpected network')):
            import run_experiment, feedback, generation_context, finalize
            from fraud_alpha.llm.client import ChatClient
            self.assertTrue(callable(run_experiment.make_pool))
            self.assertTrue(callable(finalize.select))

    def test_generation_requires_explicit_opt_in(self):
        import subprocess
        result = subprocess.run([sys.executable, str(ROOT / 'run_generation.py')], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--allow-api', result.stderr)

    def test_saved_response_reuse_and_uncertain_outcome_stop(self):
        import os
        from unittest.mock import patch
        sys.path.insert(0, str(ROOT / 'workflow'))
        import common
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'MELLM_API_KEY': 'unit-test-placeholder'}):
            folder = Path(temp)
            with patch.object(common, 'RUN', folder / 'run'):
                api = common.LocalLuna({'llm': {'model': 'gpt-6-luna', 'base_url': 'https://example.invalid/v1'}})
                response = {'model': 'gpt-6-luna', 'choices': [{'finish_reason': 'stop', 'message': {'content': 'synthetic test response'}}]}
                with patch.object(api, '_http', return_value=response) as network:
                    self.assertEqual(api.call(folder / 'success', 'system', 'prompt', 'test'), 'synthetic test response')
                    self.assertEqual(api.call(folder / 'success', 'system', 'prompt', 'test'), 'synthetic test response')
                    self.assertEqual(network.call_count, 1)
                    with self.assertRaises(AssertionError):
                        api.call(folder / 'success', 'system', 'changed prompt', 'test')
                    self.assertEqual(network.call_count, 1)
                with patch.object(api, '_http', side_effect=TimeoutError('simulated timeout')) as network:
                    with self.assertRaises(RuntimeError):
                        api.call(folder / 'uncertain', 'system', 'prompt', 'test')
                    with self.assertRaises(RuntimeError):
                        api.call(folder / 'uncertain', 'system', 'prompt', 'test')
                    self.assertEqual(network.call_count, 1)
if __name__ == '__main__':
    unittest.main()
