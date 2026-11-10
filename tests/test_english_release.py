"""Regression checks for language-sensitive feedback and local data guidance."""
import ast
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'workflow'))


class EnglishReleaseTests(unittest.TestCase):
    def test_english_source_and_decoded_literals(self):
        pattern = re.compile('[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')
        paths = list(ROOT.glob('*.py'))
        for folder in ('fraud_alpha', 'workflow', 'tests'):
            paths.extend((ROOT / folder).rglob('*.py'))
        for path in paths:
            source = path.read_text(encoding='utf-8')
            self.assertIsNone(pattern.search(source), path.name)
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if path == Path(__file__).resolve() and node.value == pattern.pattern:
                        continue
                    self.assertIsNone(pattern.search(node.value), path.name)

    def test_signed_feedback_and_rejection_categories(self):
        from fraud_alpha.feedback_memory import classify_gate_stage, make_feedback_record
        from fraud_alpha.llm.prompts import format_rich_rejection_analysis, build_compression_prompt
        for message, stage in (
            ('Raw AUC too low (0.4)', 'raw_auc'),
            ('Fast AUC too low (0.5)', 'fast_auc'),
            ('Correlation too high (0.99)', 'correlation_dedup'),
            ('Structural similarity too high (0.99)', 'structure_dedup'),
            ('No incremental AUC gain (0.71<=0.72)', 'pool_auc'),
        ):
            self.assertEqual(classify_gate_stage(message, 'rejected'), stage)
        record = make_feedback_record(
            candidate_id=1, formula='Add(N1,N2)', background='test', pool_hash='test',
            status='rejected', reason='No incremental AUC gain (0.71<=0.72)',
            auc_before=.72, auc_after=None, source_round=1,
        )
        self.assertTrue(record.evaluated)
        self.assertAlmostEqual(record.delta_signed, -.01)
        rows = [{'expr_str': 'Add(N1,N2)', 'reason': record.rejection_reason}]
        self.assertIn('| No incremental gain (insufficient marginal contribution) | 1 |', format_rich_rejection_analysis(rows))
        self.assertIn('No incremental gain', build_compression_prompt('records', '1', rows))

    def test_prompt_policy_and_lossless_context(self):
        import run_experiment
        from common import dumps
        from lossless_context import PREFIX, pack_prompt, restore_prompt
        from fraud_alpha.llm.prompts import format_pool_full_analysis
        pool = SimpleNamespace(size=20, best_auc=.7)
        reflection = SimpleNamespace(
            get_iteration_mode=lambda *a, **k: 'explore',
            generate_reflection=lambda *a, **k: '**Strictly prohibited**: using Mean as a root operator, even if you consider it effective.',
        )
        cfg = {'cold_start_rounds': 5, 'cold_start_pool_size': 20, 'cold_candidates': 15,
               'steady_candidates': 5, 'rounds': 100, 'stm_window': 3}
        records = [{'iteration': 1, 'candidates': [], 'evidence': 'synthetic context ' * 100}]
        _, count, system, prompt, _ = run_experiment.build_prompt(
            cfg, 7, pool, [], reflection, 1, {}, {}, records, [], {}, 'No prior diagnosis.'
        )
        self.assertEqual(count, 5)
        self.assertIn('complexity is not inherently better', system)
        self.assertNotIn('**Ignore existing factor directions completely**', prompt)
        self.assertNotIn('**Strictly prohibited**', prompt)
        self.assertIn('Avoid repeated identical structures rooted in Mean', prompt)
        self.assertEqual(prompt.count(PREFIX), 1)
        packed, receipt = pack_prompt(prompt)
        self.assertEqual(restore_prompt(packed), prompt)
        self.assertEqual(receipt['fields_omitted'], 0)
        source = format_pool_full_analysis([{'expr_str': 'Add(N1,N2)', 'weight': 1}], {'size': 1}, 'explore')
        self.assertIn('ignore existing factor directions completely', source)

    def test_panel_checks_distinguish_demo_from_real_sample(self):
        from validate_data import inspect_panel
        import pandas as pd
        report = inspect_panel(ROOT / 'data/synthetic_demo.csv')
        self.assertTrue(report['schema_valid'])
        self.assertFalse(report['paper_counts_match'])
        self.assertEqual(report['observed']['rows'], 450)
        self.assertEqual(report['api_requests'], 0)
        with tempfile.TemporaryDirectory() as folder:
            frame = pd.read_csv(ROOT / 'data/synthetic_demo.csv')
            duplicate = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
            path = Path(folder) / 'duplicate.csv'
            duplicate.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Duplicate firm-year'):
                inspect_panel(path)


if __name__ == '__main__':
    unittest.main()
