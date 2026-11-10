"""Complete-input memory summaries with an audited context-error fallback.

Try the unchanged full request first. Only an explicit model context rejection
enables splitting whole round records. Every record reaches a leaf request;
the resulting summaries are combined by the same local model. No token caps,
text slicing, quality rerolls, extra candidates, or model fallbacks are used.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from common import ContextWindowExceeded, clean, dumps, read, save, digest, identity, now, event
PART_GUIDANCE = '\nThe API context limit requires splitting the complete ten-round summary task into groups of complete rounds. Summarize this group under the instructions above. Identify its rounds and contemporaneous pool context, retain contrary evidence and unresolved issues, and do not infer results from missing groups. All groups will subsequently be combined.'
MERGE_GUIDANCE = '\nThe input contains all group summaries for the same long-term memory task. Integrate every group, preserving rounds, training/validation distinctions, pool context, positive and negative evidence, and case references. Do not directly pool increments from different factor-pool backgrounds or interpret them causally. Group summaries are not independent experiments; do not rerank or rescreen candidates.'

def summarize_memory(api, folder, system, prompt, purpose):
    folder = Path(folder)
    if (folder / 'input_revision.json').exists():
        revision = read(folder / 'input_revision.json')
        assert digest(folder / 'request.json') == revision['previous_request_sha256']
        assert identity(dict(system=system, prompt=prompt)) == revision['input_identity'], 'Memory input differs from explicit recovery revision'
        return summarize_memory(api, folder / revision['active_folder'], system, prompt, purpose + '_input_revision_2')
    try:
        return api.call(folder, system, prompt, purpose)
    except ContextWindowExceeded:
        pass
    records = json.loads(prompt)
    assert isinstance(records, list) and all((isinstance(r, dict) and 'iteration' in r for r in records))
    if len(records) < 2:
        raise ContextWindowExceeded('One complete round exceeds the local model context; original input retained')
    sizes = [len(dumps(r)) for r in records]
    cut = min(range(1, len(records)), key=lambda i: abs(sum(sizes[:i]) - sum(sizes[i:])))
    groups = [records[:cut], records[cut:]]
    parts = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(summarize_memory, api, folder / f'part_{i + 1}', system + PART_GUIDANCE, dumps(group), purpose + f'_part_{i + 1}') for i, group in enumerate(groups)]
        for i, future in enumerate(futures):
            text = future.result()
            parts.append(dict(folder=f'part_{i + 1}', source_rounds=[r['iteration'] for r in groups[i]], content=text))
    merge_prompt = dumps(dict(task='combine_all_memory_parts', source_rounds=[r['iteration'] for r in records], parts=parts))
    summary = api.call(folder / 'merge', system + MERGE_GUIDANCE, merge_prompt, purpose + '_merge')
    (folder / 'combined_summary.txt').write_text(summary, encoding='utf-8')
    save(folder / 'result.json', dict(status='completed', strategy='complete_round_groups_then_merge', original_request_sha256=digest(folder / 'request.json'), source_records_sha256=identity(clean(records)), source_rounds=[r['iteration'] for r in records], groups=[dict(folder=p['folder'], source_rounds=p['source_rounds']) for p in parts], final_response='merge/response.json', final_response_sha256=digest(folder / 'merge/response.json'), combined_summary_sha256=digest(folder / 'combined_summary.txt'), input_truncated=False, utc=now()))
    event('memory_summary_grouped', purpose=purpose, source_rounds=[r['iteration'] for r in records], groups=[p['source_rounds'] for p in parts], input_truncated=False)
    return summary
