"""Generate, parse, and evaluate candidates with an explicitly configured LLM client."""
from typing import Dict, List, Optional
import re
import random
import logging
from fraud_alpha.llm.client import ChatClient
from fraud_alpha.llm.prompts import build_runtime_system_prompt, create_generation_prompt, create_feedback_prompt
from fraud_alpha.alpha_pool import AlphaPool
from fraud_alpha.parser import build_parser, ExpressionParsingError, extract_expression_candidates
from fraud_alpha.expression import Expression, Feature, TSRank, Log, Abs, Div, Sub, Add, DeltaTime
from fraud_alpha.discovery_summary import FeatureDiscoverySummary, extract_variables, extract_operator_type
logger = logging.getLogger(__name__)

class LLMFactorGenerator:

    def __init__(self, client: ChatClient, pool: AlphaPool, verbose: bool=True, use_detailed_init: bool=True, discovery_summary: Optional[FeatureDiscoverySummary]=None, init_batch_size: int=10, init_max_calls_per_round: int=8):
        self.client = client
        self.pool = pool
        self.verbose = verbose
        self.use_detailed_init = use_detailed_init
        self.discovery_summary = discovery_summary
        self.init_batch_size = init_batch_size
        self.init_max_calls_per_round = init_max_calls_per_round
        self.parser = build_parser()
        self.generated_count = 0
        self.valid_count = 0
        self.added_count = 0

    def generate_initial_pool(self, n_factors: int, n_rounds: int=1) -> None:
        BATCH_SIZE = self.init_batch_size
        MAX_CALLS_PER_ROUND = self.init_max_calls_per_round
        print(f'=== Initializing factor pool with the LLM ===')
        print(f'Target: {n_factors} factors, {n_rounds} generation rounds, requesting {BATCH_SIZE} items')
        runtime_system_prompt = build_runtime_system_prompt(Feature.get_available_features())
        if self.use_detailed_init:
            print('Using a dynamic system prompt aligned with available features (initialization)')
        else:
            print('Using a dynamic system prompt aligned with available features')
        self.client.set_system_prompt(runtime_system_prompt)
        for round_idx in range(n_rounds):
            print(f'\n--- Round {round_idx + 1}/{n_rounds} ---')
            if self.pool.size >= n_factors:
                print(f'Factor pool is full ({self.pool.size}/{n_factors})')
                break
            calls_this_round = 0
            prev_size = self.pool.size
            while self.pool.size < n_factors and calls_this_round < MAX_CALLS_PER_ROUND:
                remaining = n_factors - self.pool.size
                n_to_generate = min(BATCH_SIZE, remaining)
                pool_state = self.pool.get_state() if self.pool.size > 0 else None
                calls_this_round += 1
                print(f'  [Call {calls_this_round}/{MAX_CALLS_PER_ROUND}] Requesting {n_to_generate} factors (pool={self.pool.size}/{n_factors})')
                self._generate_and_add(n_to_generate, pool_state, use_detailed=self.use_detailed_init)
                if self.pool.size == prev_size:
                    if calls_this_round >= 2:
                        print(f'  No new factors in consecutive attempts; ending this round')
                        break
                prev_size = self.pool.size
            print(f'  Round complete: pool={self.pool.size}/{n_factors}')
        print('\nResetting runtime system prompt to match currently available features')
        self.client.set_system_prompt(build_runtime_system_prompt(Feature.get_available_features()))
        print(f'\n=== Initialization complete ===')
        print(f'Factor pool size: {self.pool.size}/{n_factors}')
        print(f'Best AUC: {self.pool.best_auc:.4f}')

    def update_pool(self, n_factors: int, n_rounds: int=5) -> None:
        print(f'\n=== Iterative LLM factor-pool optimization ===')
        print(f'Factors generated per round: {n_factors}, total {n_rounds} rounds')
        for round_idx in range(n_rounds):
            print(f'\n--- Iteration {round_idx + 1}/{n_rounds} ---')
            pool_state = self.pool.get_state()
            self._generate_and_add(n_factors, pool_state)
            print(f'Current factor pool: {self.pool.size} factors, AUC={self.pool.best_auc:.4f}')
        print(f'\n=== Iterative optimization complete ===')

    def _generate_and_add(self, n_factors: int, pool_state: Optional[dict]=None, use_detailed: bool=False) -> None:
        if self.discovery_summary is not None and pool_state is not None:
            ds_text = self.discovery_summary.format_for_prompt()
            recent_rejections = None
            top_weighted = None
            if pool_state.get('exprs') and pool_state.get('weights') and pool_state.get('aucs'):
                combined = list(zip(pool_state['exprs'], pool_state['weights'], pool_state['aucs']))
                combined.sort(key=lambda x: x[1], reverse=True)
                top_weighted = [{'expr_str': str(e), 'weight': w, 'auc': a} for e, w, a in combined[:5]]
            op_freq = dict(self.discovery_summary.get_top_operator_patterns(20))
            prompt = create_feedback_prompt(n_factors=n_factors, pool_state=pool_state, discovery_summary=ds_text, recent_rejections=recent_rejections, top_weighted_factors=top_weighted, operator_frequency=op_freq if op_freq else None)
        else:
            prompt = create_generation_prompt(n_factors, pool_state, use_detailed)
        if self.verbose:
            print(f'Requesting {n_factors} factors from the LLM...')
        try:
            response = self.client.chat(prompt)
            exprs = self._parse_response(response)
            if self.verbose:
                print(f'LLM returned {len(exprs)} expressions')
            added_to_pool = 0
            contributed = 0
            for expr in exprs:
                try:
                    reward, is_added, reason = self.pool.try_new_expr(expr)
                    if self.discovery_summary is not None:
                        if is_added:
                            variables = extract_variables(expr)
                            op_type = extract_operator_type(expr)
                            auc_val = self.pool.aucs[-1] if self.pool.aucs else 0.0
                            self.discovery_summary.record_accepted(expr, auc_val, op_type, variables)
                        else:
                            self.discovery_summary.record_rejected(expr, reason)
                    if self.verbose:
                        if is_added and reward > 0:
                            print(f'  [OK] Admitted with contribution: {expr}')
                            print(f'      -> {reason}')
                            added_to_pool += 1
                            contributed += 1
                        elif is_added and reward == 0:
                            print(f'  [OK] Admitted without contribution: {expr}')
                            print(f'      -> {reason}')
                            added_to_pool += 1
                        else:
                            print(f'  [NO] Rejected: {expr}')
                            print(f'      -> {reason}')
                except Exception as e:
                    if self.verbose:
                        print(f'  [ERR] Error: {expr}')
                        print(f'      -> {e}')
            self.added_count += added_to_pool
            self.pool.flush_optimization()
            if self.verbose:
                print(f'Factor pool status: admitted {added_to_pool}/{len(exprs)} candidates, pool size={self.pool.size}, AUC={self.pool.best_auc:.4f}')
        except Exception as e:
            print(f'LLM generation failed: {e}')

    def _extract_expression(self, line: str) -> str:
        candidates = extract_expression_candidates(line)
        return candidates[0] if candidates else line.strip()

    def _parse_response(self, response: str) -> List[Expression]:
        exprs = []
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('```'):
                continue
            if re.match('^\\s*delete\\s*:', line, re.IGNORECASE):
                continue
            line = re.sub('^\\d+\\.\\s*', '', line)
            line = re.sub('^[-*]\\s*', '', line)
            if '//' in line:
                expr_part, _ = line.split('//', 1)
                line = expr_part.strip()
            candidates = extract_expression_candidates(line)
            parsed = False
            last_error = None
            failed_expr = candidates[0] if candidates else line
            for candidate in candidates or [line]:
                try:
                    expr = self.parser.parse(candidate)
                    exprs.append(expr)
                    self.valid_count += 1
                    parsed = True
                    break
                except ExpressionParsingError as e:
                    last_error = e
                    failed_expr = candidate
                except Exception as e:
                    last_error = e
                    failed_expr = candidate
            self.generated_count += 1
            if not parsed and self.verbose and (last_error is not None):
                print(f'  Parsing failed: {failed_expr} - {last_error}')
        return exprs

    def _parse_delete_instructions(self, response: str) -> List[str]:
        if not response or not response.strip():
            return []
        identifiers = []
        pattern = re.compile('^\\s*delete\\s*:\\s*(.+)', re.IGNORECASE)
        for line in response.strip().split('\n'):
            match = pattern.match(line.strip())
            if match:
                identifier = match.group(1).strip()
                if identifier:
                    identifiers.append(identifier)
        return identifiers

    def generate_and_apply(self, n_factors: int, prompt: str) -> Dict:
        empty_result = {'added': [], 'deleted': [], 'raw_response': '', 'raw_responses': [], 'selected_response_index': None, 'candidate_summaries': []}
        try:
            if self.verbose:
                print(f'Calling the LLM to generate {n_factors} factors (including DELETE parsing)...')
            response = self.client.chat(prompt)
            replay_meta = self.client.get_replay_metadata() if hasattr(self.client, 'get_replay_metadata') else None
            if not response or not response.strip():
                if self.verbose:
                    print('LLM returned an empty response; skipping this round')
                return empty_result
            if self.verbose:
                print(f'LLM response length: {len(response)} characters')
            added_exprs = self._parse_response(response)
            added = [(expr, str(expr)) for expr in added_exprs]
            deleted = self._parse_delete_instructions(response)
            if self.verbose:
                print(f'Parsing result: ADD={len(added)}, DELETE={len(deleted)} items')
                for expr, expr_str in added:
                    print(f'  [ADD] {expr_str}')
                for ident in deleted:
                    print(f'  [DEL] {ident}')
            return {'added': added, 'deleted': deleted, 'raw_response': replay_meta.get('raw_response', response) if replay_meta else response, 'raw_responses': list(replay_meta.get('raw_responses') or [response]) if replay_meta else [response], 'selected_response_index': replay_meta.get('selected_response_index', 0) if replay_meta else 0, 'candidate_summaries': (list(replay_meta.get('round_meta', {}).get('candidate_summaries') or []) if replay_meta else []) or [{'index': replay_meta.get('selected_response_index', 0) if replay_meta else 0, 'n_parsed': len(added), 'n_deleted': len(deleted), 'score': None, 'selected': True, 'response_length': len(response)}]}
        except Exception as e:
            if self.verbose:
                print(f'generate_and_apply failed: {e}')
            empty_result['raw_response'] = str(e)
            return empty_result

    def generate_and_apply_best_of_n(self, n_factors: int, prompt: str, n_samples: int=6) -> Dict:
        if n_samples <= 1:
            return self.generate_and_apply(n_factors, prompt)
        empty_result = {'added': [], 'deleted': [], 'raw_response': '', 'raw_responses': [], 'selected_response_index': None, 'candidate_summaries': []}
        if not hasattr(self.client, 'parallel_chat'):
            if self.verbose:
                print(f'Client does not support parallel_chat; falling back to a single call')
            return self.generate_and_apply(n_factors, prompt)
        try:
            if self.verbose:
                print(f'Best-of-{n_samples}: parallel LLM calls to generate {n_factors} factors from the LLM...')
            responses = self.client.parallel_chat(prompt, n_samples=n_samples)
            replay_meta = self.client.get_replay_metadata() if hasattr(self.client, 'get_replay_metadata') else None
            if not responses:
                if self.verbose:
                    print('All parallel calls failed; skipping this round')
                return empty_result
            if replay_meta and replay_meta.get('raw_responses'):
                responses = list(replay_meta.get('raw_responses') or responses)
            empty_result['raw_responses'] = list(responses)
            candidates = []
            for resp_idx, response in enumerate(responses):
                if not response or not response.strip():
                    continue
                added_exprs = self._parse_response(response)
                added = [(expr, str(expr)) for expr in added_exprs]
                deleted = self._parse_delete_instructions(response)
                score = self._score_candidate_group(added)
                candidates.append({'score': score, 'added': added, 'deleted': deleted, 'raw_response': response, 'n_parsed': len(added), 'idx': resp_idx})
            if not candidates:
                if self.verbose:
                    print('All responses failed parsing; skipping this round')
                return empty_result
            replay_selected_idx = replay_meta.get('selected_response_index') if replay_meta else None
            if replay_selected_idx is not None:
                best = next((c for c in candidates if c['idx'] == replay_selected_idx), None)
                if best is None:
                    raise ValueError(f'Invalid replay selected_response_index: {replay_selected_idx}')
            else:
                best = max(candidates, key=lambda c: c['score'])
            if self.verbose:
                print(f'Best-of-{n_samples}Scoring results:')
                ranked_candidates = sorted(candidates, key=lambda c: c['score'], reverse=True)
                for c in ranked_candidates:
                    print(f"  Group #{c['idx'] + 1}: parsed={c['n_parsed']}, score={c['score']:.4f}")
                print(f"  -> Selected group #{best['idx'] + 1} (score={best['score']:.4f}, {best['n_parsed']} factors)")
            return {'added': best['added'], 'deleted': best['deleted'], 'raw_response': replay_meta.get('raw_response', best['raw_response']) if replay_meta else best['raw_response'], 'raw_responses': list(responses), 'selected_response_index': best['idx'], 'candidate_summaries': (list(replay_meta.get('round_meta', {}).get('candidate_summaries') or []) if replay_meta else []) or [{'index': c['idx'], 'n_parsed': c['n_parsed'], 'n_deleted': len(c['deleted']), 'score': c['score'], 'selected': c['idx'] == best['idx'], 'response_length': len(c['raw_response'])} for c in candidates]}
        except Exception as e:
            if self.verbose:
                print(f'generate_and_apply_best_of_n failed: {e}')
            empty_result['raw_response'] = str(e)
            return empty_result

    def _score_candidate_group(self, added: List[tuple]) -> float:
        if not added:
            return 0.0
        import numpy as np
        calculator = self.pool.calculator
        raw_threshold = getattr(self.pool, 'raw_auc_threshold', 0.52)
        passed_aucs = []
        all_aucs = []
        for expr, expr_str in added:
            try:
                raw_auc = calculator.calc_raw_auc(expr, use_valid=True)
                all_aucs.append(raw_auc)
                if raw_auc >= raw_threshold:
                    passed_aucs.append(raw_auc)
            except Exception:
                continue
        n_passed = len(passed_aucs)
        avg_auc = float(np.mean(passed_aucs)) if passed_aucs else 0.0
        diversity_bonus = float(np.std(all_aucs)) if len(all_aucs) >= 2 else 0.0
        score = n_passed * 1.0 + avg_auc * 0.1 + diversity_bonus * 0.01
        return score

    def generate_and_collect_all(self, n_factors: int, prompt: str, n_samples: int=6) -> Dict:
        if n_samples <= 1:
            result = self.generate_and_apply(n_factors, prompt)
            result['per_group_counts'] = [len(result.get('added', []))]
            return result
        empty_result = {'added': [], 'deleted': [], 'raw_response': '', 'raw_responses': [], 'selected_response_index': None, 'candidate_summaries': [], 'per_group_counts': []}
        if not hasattr(self.client, 'parallel_chat'):
            if self.verbose:
                print(f'Client does not support parallel_chat; falling back to a single call')
            result = self.generate_and_apply(n_factors, prompt)
            result['per_group_counts'] = [len(result.get('added', []))]
            return result
        try:
            if self.verbose:
                print(f'Collect-All-{n_samples}: parallel LLM calls to generate {n_factors} factors from the LLM...')
            responses = self.client.parallel_chat(prompt, n_samples=n_samples)
            replay_meta = self.client.get_replay_metadata() if hasattr(self.client, 'get_replay_metadata') else None
            if not responses:
                if self.verbose:
                    print('All parallel calls failed; skipping this round')
                return empty_result
            if replay_meta and replay_meta.get('raw_responses'):
                responses = list(replay_meta.get('raw_responses') or responses)
            empty_result['raw_responses'] = list(responses)
            seen_exprs = set()
            all_added = []
            all_deleted = []
            per_group_counts = []
            candidate_summaries = []
            for resp_idx, response in enumerate(responses):
                if not response or not response.strip():
                    per_group_counts.append(0)
                    continue
                added_exprs = self._parse_response(response)
                added = [(expr, str(expr)) for expr in added_exprs]
                deleted = self._parse_delete_instructions(response)
                group_count = 0
                for expr, expr_str in added:
                    if expr_str not in seen_exprs:
                        seen_exprs.add(expr_str)
                        all_added.append((expr, expr_str))
                        group_count += 1
                all_deleted.extend(deleted)
                per_group_counts.append(len(added))
                candidate_summaries.append({'index': resp_idx, 'n_parsed': len(added), 'n_unique_contributed': group_count, 'n_deleted': len(deleted), 'response_length': len(response)})
            if self.verbose:
                print(f'Collect-All-{n_samples}: collected {len(all_added)} unique candidate factors')
                for cs in candidate_summaries:
                    print(f"  Group #{cs['index'] + 1}: parsed={cs['n_parsed']}, contribution={cs['n_unique_contributed']} new factors")
            return {'added': all_added, 'deleted': all_deleted, 'raw_response': replay_meta.get('raw_response', responses[0]) if replay_meta else responses[0], 'raw_responses': list(responses), 'selected_response_index': None, 'candidate_summaries': candidate_summaries, 'per_group_counts': per_group_counts}
        except Exception as e:
            if self.verbose:
                print(f'generate_and_collect_all failed: {e}')
            empty_result['raw_response'] = str(e)
            return empty_result

    def get_stats(self) -> dict:
        return {'generated': self.generated_count, 'valid': self.valid_count, 'added': self.added_count, 'valid_rate': self.valid_count / max(1, self.generated_count), 'add_rate': self.added_count / max(1, self.valid_count)}

    @staticmethod
    def generate_seed_expressions(required_vars: List[str], n_seeds: int=5) -> List[Expression]:
        if not required_vars:
            return []
        all_vars = Feature.get_available_features()
        seeds: List[Expression] = []
        templates = ['rank', 'log_abs', 'div', 'sub_rank']
        for i, var in enumerate(required_vars):
            if len(seeds) >= n_seeds:
                break
            feat = Feature(var)
            tmpl = templates[i % len(templates)]
            if tmpl == 'rank':
                seeds.append(TSRank(feat, 3))
            elif tmpl == 'log_abs':
                seeds.append(Log(Abs(feat)))
            elif tmpl == 'div':
                candidates = [v for v in all_vars if v != var]
                if not candidates:
                    seeds.append(TSRank(feat, 3))
                    continue
                other = random.choice(candidates)
                seeds.append(Div(feat, Feature(other)))
            elif tmpl == 'sub_rank':
                candidates = [v for v in all_vars if v != var]
                if not candidates:
                    seeds.append(TSRank(feat, 3))
                    continue
                other = random.choice(candidates)
                seeds.append(TSRank(Sub(feat, Feature(other)), 3))
        logger.info(f'Generating {len(seeds)} seed expressions: {[str(s) for s in seeds]}')
        return seeds
