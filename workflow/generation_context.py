"""Register and send complete generation inputs using lossless JSON references."""
import json
from common import RUN,read,save,digest,identity,is_context_rejection
from lossless_context import pack_prompt

def immutable_json(path,value):
    if path.exists():assert read(path)==value,f'Changed immutable context artifact: {path}'
    else:save(path,value)

def prepare(folder,iteration,system,source):
    policy_path=RUN/'generation_context_policy.json'
    if not policy_path.exists():return source,[folder/f'generation_{g}' for g in range(1,4)],{}
    assert digest(policy_path)==read(RUN/'source_manifest.json')['generation_context_policy_sha256']
    policy=read(policy_path)
    if iteration<policy['first_round']:return source,[folder/f'generation_{g}' for g in range(1,4)],{}
    sent,proof=pack_prompt(source)
    assert len(sent)<len(source),'Lossless encoding did not reduce this input'
    source_path=folder/'generation_source_prompt.txt'
    if source_path.exists():assert source_path.read_text(encoding='utf-8')==source
    else:source_path.write_text(source,encoding='utf-8')
    proof.update(policy_sha256=digest(policy_path),source_file=source_path.name)
    immutable_json(folder/'generation_context.json',proof)
    original=dict(model='gpt-6-luna',messages=[dict(role='system',content=system),dict(role='user',content=source)],reasoning_effort='medium',service_tier='priority',stream=False)
    actual={**original,'messages':[dict(role='system',content=system),dict(role='user',content=sent)]}
    folders=[]
    for g in range(1,4):
        root=folder/f'generation_{g}';root.mkdir(exist_ok=True)
        failures={}
        if (root/'request.json').exists():
            assert read(root/'request.json')==original,'Original failed request source changed'
            assert not (root/'response.json').exists(),'Never replace an already received response'
            starts=list(root.glob('attempt_*.started.json'));assert starts
            for start in starts:
                fail=start.with_name(start.name.replace('.started.json','.failed.json'))
                assert fail.exists() and is_context_rejection(read(fail)['message']),'Original outcome is not a definite context rejection'
                failures[fail.name]=digest(fail)
        immutable_json(root/'source_request.json',original)
        revision=dict(encoding='complete_json_references_v1',active_folder='lossless_v1',source_request_sha256=digest(root/'source_request.json'),
            active_request_hash=identity(actual),source_request_hash=identity(original),known_original_context_failures=failures,
            previous_request_sha256=digest(root/'request.json') if (root/'request.json').exists() else None,
            proof_sha256=digest(folder/'generation_context.json'),policy_sha256=digest(policy_path),input_truncated=False,fields_omitted=0)
        immutable_json(root/'input_revision.json',revision)
        folders.append(root/'lossless_v1')
    return sent,folders,dict(source_characters=len(source),source_prompt_sha256=proof['source_sha256'],serialization=proof['encoding'],
        characters_are_after_lossless_serialization=True,section_characters_are_before_serialization=True)
