"""Lossless references for repeated JSON in complete recent-round records.

This is serialization, not summarization: every field and value is recoverable.
No LLM/API call, token threshold, text clipping, or field selection is used.
"""
from collections import Counter
import hashlib
import json
from common import dumps
PREFIX = '## Complete candidate evaluation records for the last 3 active rounds\n'
NOTE = '[Lossless reference serialization] The following JSON preserves every field, value, list order, and text in these three rounds. Repeated identical objects or lists are stored once in definitions. A {"$mellm_ref":"identifier"} in root or a definition expands to that identifier\'s complete value at every occurrence. All definitions are included in this input; no external files are needed. No records have been removed, summarized, or filtered. Read each candidate\'s pool_before, pool_after, measured increment, and case references using the complete definitions.\n'

def pack(value):
    counts = Counter()

    def count(x):
        if isinstance(x, (dict, list)):
            if isinstance(x, dict):
                assert '$mellm_ref' not in x, 'Reserved reference key in original data'
            text = dumps(x)
            if len(text) >= 1024:
                counts[text] += 1
            for child in x.values() if isinstance(x, dict) else x:
                count(child)
    count(value)
    ids = {}
    definitions = []

    def encode(x, define=True):
        if isinstance(x, (dict, list)):
            text = dumps(x)
            if define and counts[text] > 1:
                if text not in ids:
                    key = f'R{len(ids) + 1:04d}'
                    ids[text] = key
                    definitions.append(dict(id=key, value=encode(x, False)))
                return {'$mellm_ref': ids[text]}
            if isinstance(x, dict):
                return {k: encode(v) for k, v in x.items()}
            return [encode(v) for v in x]
        return x
    root = encode(value)
    return dict(encoding='complete_json_references_v1', definitions=definitions, root=root)

def unpack(payload):
    assert payload['encoding'] == 'complete_json_references_v1'
    definitions = {d['id']: d['value'] for d in payload['definitions']}
    assert len(definitions) == len(payload['definitions'])
    used = set()

    def decode(x, stack=()):
        if isinstance(x, dict):
            if '$mellm_ref' in x:
                assert set(x) == {'$mellm_ref'}
                key = x['$mellm_ref']
                assert key in definitions and key not in stack
                used.add(key)
                return decode(definitions[key], stack + (key,))
            return {k: decode(v, stack) for k, v in x.items()}
        if isinstance(x, list):
            return [decode(v, stack) for v in x]
        return x
    value = decode(payload['root'])
    assert used == set(definitions)
    return value

def pack_prompt(source):
    assert source.count(PREFIX) == 1
    start = source.index(PREFIX) + len(PREFIX)
    value, end = json.JSONDecoder().raw_decode(source, start)
    assert source[start:end] == dumps(value)
    payload = pack(value)
    assert unpack(payload) == value
    sent = source[:start] + NOTE + json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + source[end:]
    assert restore_prompt(sent) == source
    return (sent, dict(encoding=payload['encoding'], definitions=len(payload['definitions']), source_characters=len(source), sent_characters=len(sent), source_sha256=hashlib.sha256(source.encode()).hexdigest(), sent_sha256=hashlib.sha256(sent.encode()).hexdigest(), decoded_source_sha256=hashlib.sha256(restore_prompt(sent).encode()).hexdigest(), input_truncated=False, fields_omitted=0, model_calls_for_encoding=0))

def restore_prompt(sent):
    assert sent.count(PREFIX) == 1
    start = sent.index(PREFIX) + len(PREFIX)
    assert sent.startswith(NOTE, start)
    payload, end = json.JSONDecoder().raw_decode(sent, start + len(NOTE))
    return sent[:start] + dumps(unpack(payload)) + sent[end:]
