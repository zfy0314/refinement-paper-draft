"""Read-only checks of numeric claims and generated-figure provenance."""
import json
from decimal import Decimal, ROUND_HALF_UP
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/evidence'
manifest=json.loads((DATA/'provenance.json').read_text())
source=Path(manifest['source_root'])
for relative,expected in manifest['files'].items():
    assert hashlib.sha256((source/relative).read_bytes()).hexdigest()==expected, relative
for k in [0,3]:
    original=pd.read_csv(source/'logs/IL_racecar'/f'logs_iter{k:03d}.csv')
    selected=pd.read_csv(DATA/f'rollout_round{k}_selected.csv')
    assert len(selected)==1000
    for column in selected.columns:
        if column!='step':np.testing.assert_allclose(selected[column],original[column],rtol=1e-12,atol=1e-12)
    if k==3:
        p=np.maximum(original.latent_wheel_abs_mean-original.latent_speed,0)
        q=p/(1+(original.weight_slip_scale+1e-6)*p)
        np.testing.assert_allclose(q,original.latent_wheel_slip_pos_norm,atol=1e-7)
        assert (original.weight_speed_slip<=0).all()
        assert (q>=0).all()
    np.testing.assert_allclose(original.latent_speed.to_numpy()[1:],original.obs_current_speed.to_numpy()[:-1],atol=0,rtol=0)
summary=json.loads((DATA/'feedback_summary.json').read_text())
assert summary['code_means'][:5]==[762.58,890.60,892.32,839.30,881.99]
assert min(np.array(summary['no_feedback_means'])-np.array(summary['no_feedback_stds']))<0
assert (Decimal(100)*403/2000).quantize(Decimal('.1'),rounding=ROUND_HALF_UP)==Decimal('20.2')
assert round(100*(1-.110/.548),1)==79.9
print('Source hashes, all plotted trace rows, revised slip computation, timestamp alignment, feedback arrays, and headline arithmetic verified.')
