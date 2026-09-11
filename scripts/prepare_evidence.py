"""Recover paper evidence without importing or executing generated policies.

Run with a Python environment containing numpy, pandas, and matplotlib:
    MPLCONFIGDIR=/tmp/nemo-mpl python scripts/prepare_evidence.py
"""
import ast
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/Users/zfy/Projects/meta/llm-log-analysis')
RUN = SOURCE / 'logs/IL_racecar'
OUT = ROOT / 'data/evidence'
FIG = ROOT / 'figures/generated'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
manifest = {'source_root': str(SOURCE), 'run': 'logs/IL_racecar', 'files': {}, 'metrics': {}}
def record(path):
    manifest['files'][str(path.relative_to(SOURCE))] = hashlib.sha256(path.read_bytes()).hexdigest()

# Recover the actual initial policy from the request sent in this run.
# The generic models/model_racecar_iter000.py is a different policy.
prompt0 = json.loads((RUN / 'prompt_iter000.json').read_text())
user0 = next(m['content'] for m in prompt0 if m['role'] == 'user')
code0 = re.search(r'\[Policy Structure\].*?```(?:py|python)?\s*\n(.*?)```', user0, re.S).group(1)
(OUT / 'policy_round0.py').write_text(code0.rstrip() + '\n')
record(RUN / 'prompt_iter000.json')
for k in [1,2,3]:
    f = RUN / f'model_racecar_iter{k:03d}.py'
    (OUT / f'policy_round{k}.py').write_bytes(f.read_bytes())
    record(f)
for k in [0,1,2]:
    f = RUN / f'prompt_iter{k:03d}.json'
    d = json.loads(f.read_text())
    rationale = [m['content'] for m in d if m['role']=='assistant'][-1]
    rationale = rationale.split('[Updated Model]')[0]
    (OUT / f'revision_after_round{k}.txt').write_text(rationale)
    record(f)
    f = RUN / f'results_iter{k:03d}.txt'
    (OUT / f'diagnostics_round{k}.txt').write_bytes(f.read_bytes())
    record(f)
    f = RUN / f'tests_iter{k:03d}.py'
    (OUT / f'tests_round{k}.py').write_bytes(f.read_bytes())
    record(f)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False})
for k in [0,1,2,3,4]:
    f = RUN / f'logs_iter{k:03d}.csv'; record(f)
    df = pd.read_csv(f)
    qcol = 'latent_wheel_slip' if k==0 else ('latent_wheel_slip_rel' if k==1 else ('latent_wheel_slip_pos' if k==2 else 'latent_wheel_slip_pos_norm'))
    manifest['metrics'][f'round{k}'] = {
        'rows': len(df), 'episodes': int(df.episodeID.nunique()),
        'slip_weight': float(df.weight_speed_slip.iloc[0]), 'slip_latent': qcol,
        'slip_latent_range': [float(df[qcol].min()),float(df[qcol].max())],
    }
    # Confirm the recorder uses successor observations next to current-step latents.
    for lat,obs in [('latent_speed','obs_current_speed'),('latent_closest_x','obs_tile_0_x')]:
        same = np.abs(df[lat].to_numpy()-df[obs].to_numpy()).mean()
        aligned = np.abs(df[lat].to_numpy()[1:]-df[obs].to_numpy()[:-1]).mean()
        manifest['metrics'][f'round{k}'][f'{lat}_same_row_mae'] = float(same)
        manifest['metrics'][f'round{k}'][f'{lat}_shifted_mae'] = float(aligned)
    if k in [0,3]:
        columns=['episodeID',qcol,'latent_speed','latent_desired_speed','latent_speed_error','weight_speed_slip','latent_accel_raw','latent_brake_raw']
        selected=df[columns].copy();selected.insert(0,'step',np.arange(len(df)))
        selected.to_csv(OUT/f'rollout_round{k}_selected.csv',index=False)
        fig,axs=plt.subplots(2,1,figsize=(3.1,2.25),sharex=True,layout='constrained')
        x=np.arange(len(df))
        axs[0].plot(x,df[qcol],color='#007f86',lw=1.1)
        axs[0].set_ylim(-.03,.62);axs[0].set_ylabel('Slip proxy $q$')
        axs[0].set_title('Initial proxy: wheel mean' if k==0 else 'Revised proxy: positive, compressed',fontsize=9,loc='left')
        axs[1].plot(x,df.latent_desired_speed,label='Target speed',color='#a66017',lw=1.0)
        axs[1].plot(x,df.latent_speed,label='Current speed',color='#435eb0',lw=.9,alpha=.85)
        axs[1].set_ylim(-.05,1.05);axs[1].set_ylabel('Speed');axs[1].set_xlabel('Rollout step')
        axs[1].legend(ncol=2,loc='upper right',fontsize=7,frameon=False)
        for ax in axs:ax.grid(alpha=.2,lw=.5);ax.set_xlim(0,999)
        fig.savefig(FIG/f'latent_round{k}.pdf',bbox_inches='tight');plt.close(fig)

# Parse original numeric arrays, without executing the experiment script.
count_path=SOURCE/'count.py';record(count_path)
values={}
for node in ast.parse(count_path.read_text()).body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        name=node.targets[0].id
        if name.endswith(('_means','_stds')) and isinstance(node.value,ast.Call): values[name]=ast.literal_eval(node.value.args[0])
        elif name=='human':values[name]=ast.literal_eval(node.value)
(OUT/'feedback_summary.json').write_text(json.dumps(values,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(2.85,2.15),layout='constrained')
for name,label,color,marker in [('code','Generated analysis','#2673b8','o'),('table','Trajectory table','#d98224','s'),('no_feedback','No feedback','#3b9256','^')]:
    mean=np.array(values[name+'_means'][:5]);std=np.array(values[name+'_stds'][:5]);x=np.arange(5)
    ax.plot(x,mean,label=label,color=color,marker=marker,ms=4,lw=1.5)
    ax.fill_between(x,mean-std,mean+std,color=color,alpha=.16,linewidth=0)
ax.axhline(values['human'],color='#aa455b',ls='--',lw=1,label='Demonstration return')
ax.set_yscale('linear')
ax.set_yticks([0,300,600,900])
ax.set_ylim(-150,1000);ax.set_xticks(range(5));ax.set_xlabel('Number of refinements',fontsize=8)
ax.set_ylabel('Return',fontsize=8);ax.tick_params(labelsize=8);ax.grid(axis='y',alpha=.22)
ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False,
          fontsize=7,columnspacing=.65,handlelength=1.3,handletextpad=.35)
fig.savefig(FIG/'refinement_rewards_linear.pdf',bbox_inches='tight')
fig.savefig(FIG/'refinement_rewards_linear.svg',bbox_inches='tight');plt.close(fig)
for f in [SOURCE/'utils.py',SOURCE/'exp_il.py',SOURCE/'exp_rl.py',SOURCE/'env/racecar_env.py']:record(f)
(OUT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest['metrics'],indent=2))
