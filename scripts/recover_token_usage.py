"""Verify the token table from archived metadata without changing source files."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = Path('/Users/zfy/Projects/meta/llm-log-analysis/logs/IL_racecar')
items, hashes = [], {}
for k in range(6):
    usages = []
    for suffix in ['', '_tmp']:
        path = RUN / f'prompt_iter{k:03d}{suffix}.json.meta.json'
        content = path.read_text()
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if content.startswith('1['):
            content = content[1:]  # Documented stray prefix; leave original untouched.
        usages.append(json.loads(content)[0]['usage'])
    items.append({
        'round': k,
        'input_tokens': sum(d['input_tokens'] for d in usages),
        'output_tokens': sum(d['output_tokens'] for d in usages),
        'reasoning_tokens': sum(d['output_tokens_details']['reasoning_tokens'] for d in usages),
    })
summary = {name: {'mean': float(np.mean([d[name] for d in items])),
                  'population_std': float(np.std([d[name] for d in items]))}
           for name in ['input_tokens', 'output_tokens']}
payload = {
    'aggregation': 'Two calls per refinement round; population std across six rounds.',
    'source_note': 'prompt_iter000_tmp.json.meta.json begins with a stray 1; ignored only during parsing.',
    'source_hashes': hashes, 'rounds': items, 'summary': summary,
}
(ROOT / 'data/evidence/analysis_token_usage.json').write_text(json.dumps(payload, indent=2) + '\n')
print(json.dumps(summary, indent=2))
