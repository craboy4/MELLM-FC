"""Structured candidate feedback and memory utilities.

This module is deliberately independent of the model and pool implementation.  It
provides a lossless record for every candidate evaluation so that downstream
prompts can distinguish a measured negative result from a candidate that was
rejected before any AUC was measured.  In particular, ``None`` is used for an
unmeasured delta; zero is reserved for a measured no-change result.
"""
from __future__ import annotations
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
_FLOAT = '[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][-+]?\\d+)?'

@dataclass
class CandidateFeedback:
    candidate_id: int
    formula: str
    background: str
    pool_hash: str
    gate_stage: str
    status: str
    evaluation_status: str
    auc_before: Optional[float]
    auc_after: Optional[float]
    delta_signed: Optional[float]
    rejection_reason: str
    evaluated: bool
    source_round: Optional[int]
    trial_auc: Optional[float] = None
    agent_id: Optional[int] = None
    strategy_name: str = ''
    rationale: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def pool_state_hash(pool: Any) -> str:
    exprs = [str(expr) for expr in getattr(pool, 'exprs', [])]
    payload = {'exprs': exprs, 'best_auc': round(float(getattr(pool, 'best_auc', 0.0)), 12), 'fold_aucs': {str(k): round(float(v), 12) for k, v in sorted((getattr(pool, 'fold_aucs', {}) or {}).items())}}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()

def extract_trial_auc(reason: str) -> Optional[float]:
    text = str(reason or '')
    patterns = (f'(?:combined|trial|pool)\\s*AUC\\s*[=:]\\s*({_FLOAT})', f'best attempt\\s*[=:]\\s*({_FLOAT})', f'No incremental AUC gain\\s*\\(\\s*({_FLOAT})\\s*<=', f'best\\s*trial\\s*AUC\\s*[=:]\\s*({_FLOAT})')
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (TypeError, ValueError):
                return None
    return None

def classify_gate_stage(reason: str, status: str='') -> str:
    text = str(reason or '')
    lower = text.lower()
    if status == 'error' or 'error' in text or 'exception' in lower:
        return 'error'
    if 'Duplicate expression' in text or 'duplicate' in lower:
        return 'duplicate'
    if 'structural similarity' in text or 'structural' in lower:
        return 'structure_dedup'
    if 'raw auc' in lower or 'raw_auc' in lower:
        return 'raw_auc'
    if 'correlation' in text or 'corr' in lower:
        return 'correlation_dedup'
    if 'fast auc' in lower or 'fast_auc' in lower or 'logistic regression' in text:
        return 'fast_auc'
    if 'single-factor auc' in lower or 'single' in lower:
        return 'single_auc'
    if 'combined auc' in lower or 'incremental auc' in lower or 'pool_auc' in lower or ('best attempt' in text):
        return 'pool_auc'
    if status == 'added':
        return 'accepted'
    if status == 'rejected':
        return 'rejected_other'
    return 'unknown'

def make_feedback_record(*, candidate_id: int, formula: str, background: str, pool_hash: str, status: str, reason: str, auc_before: Optional[float], auc_after: Optional[float], source_round: Optional[int], agent_id: Optional[int]=None, strategy_name: str='', rationale: str='') -> CandidateFeedback:
    trial_auc = extract_trial_auc(reason)
    evaluated = trial_auc is not None
    if auc_after is not None and auc_before is not None and (status == 'added'):
        delta_signed: Optional[float] = float(auc_after) - float(auc_before)
    elif trial_auc is not None and auc_before is not None:
        delta_signed = float(trial_auc) - float(auc_before)
    else:
        delta_signed = None
    evaluation_status = 'measured' if evaluated else 'error' if status == 'error' else 'unmeasured'
    return CandidateFeedback(candidate_id=int(candidate_id), formula=str(formula), background=str(background), pool_hash=str(pool_hash), gate_stage=classify_gate_stage(reason, status), status=str(status), evaluation_status=evaluation_status, auc_before=None if auc_before is None else float(auc_before), auc_after=None if auc_after is None else float(auc_after), delta_signed=delta_signed, rejection_reason=str(reason or ''), evaluated=bool(evaluated), source_round=None if source_round is None else int(source_round), trial_auc=trial_auc, agent_id=None if agent_id is None else int(agent_id), strategy_name=str(strategy_name or ''), rationale=str(rationale or ''))

def summarize_feedback(records: Iterable[Mapping[str, Any] | CandidateFeedback]) -> Dict[str, Any]:
    rows = [r.to_dict() if isinstance(r, CandidateFeedback) else dict(r) for r in records]
    measured = [r for r in rows if r.get('evaluation_status') == 'measured']
    positives = [r for r in measured if r.get('delta_signed') is not None and float(r['delta_signed']) > 0]
    negatives = [r for r in measured if r.get('delta_signed') is not None and float(r['delta_signed']) < 0]
    ties = [r for r in measured if r.get('delta_signed') is not None and float(r['delta_signed']) == 0]
    unmeasured = [r for r in rows if r.get('evaluation_status') == 'unmeasured']
    stages = Counter((str(r.get('gate_stage', 'unknown')) for r in rows))
    signed = [float(r['delta_signed']) for r in measured if r.get('delta_signed') is not None]
    return {'n_records': len(rows), 'n_measured': len(measured), 'n_positive': len(positives), 'n_negative': len(negatives), 'n_tie': len(ties), 'n_unmeasured': len(unmeasured), 'gate_stage_counts': dict(sorted(stages.items())), 'signed_delta_mean': sum(signed) / len(signed) if signed else None, 'signed_delta_min': min(signed) if signed else None, 'signed_delta_max': max(signed) if signed else None, 'positive': positives, 'negative': negatives, 'unmeasured': unmeasured}

def format_feedback_summary(summary: Mapping[str, Any], max_items: int=5) -> str:
    lines = ['## Structured candidate feedback (signed; unevaluated is not zero)']
    lines.append('- Evaluated: {n_measured}; positive gain: {n_positive}; negative gain: {n_negative}; unchanged: {n_tie}; unevaluated: {n_unmeasured}'.format(**summary))
    stages = summary.get('gate_stage_counts', {})
    if stages:
        lines.append('- Gate stage: ' + ', '.join((f'{k}={v}' for k, v in stages.items())))
    mean = summary.get('signed_delta_mean')
    if mean is not None:
        lines.append(f'- Mean evaluated delta={float(mean):+.6f} (signs preserved)')
    for label, key in (('positive gain', 'positive'), ('negative gain', 'negative'), ('unevaluated', 'unmeasured')):
        items = list(summary.get(key, []))[:max_items]
        if not items:
            continue
        lines.append(f'### {label}')
        for item in items:
            delta = item.get('delta_signed')
            delta_text = 'NA' if delta is None else f'{float(delta):+.6f}'
            lines.append(f"- {item.get('formula', '')} | stage={item.get('gate_stage', '')} | delta={delta_text} | reason={item.get('rejection_reason', '')}")
    return '\n'.join(lines)
