"""Plot iteration-3 recorded offset/steering; reproduce the off-track filter."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/Users/zfy/Projects/meta/llm-log-analysis/logs/IL_racecar/logs_iter003.csv')
with SOURCE.open() as f: rows=list(csv.DictReader(f))
x=np.array([float(r['latent_closest_x']) for r in rows])
a=np.array([float(r['output_clipped_steer']) for r in rows])
half_track_width = 0.2  # Half of the normalized 0.4-wide track in the task description.
off=np.abs(x)>half_track_width
assert off.sum()==74 and ((a*x>0)&off).sum()==11
plt.rcParams.update({'font.size':7,'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,1,figsize=(3.15,1.78),sharex=True)
fig.subplots_adjust(left=.32,right=.98,bottom=.20,top=.80,hspace=.22)
for ax,values,label,color in zip(axes,[x,a],['Lateral\noffset','Steering\ncommand'],['#2868a0','#47886e']):
 ax.plot(values,lw=.75,color=color)
 ax.fill_between(np.arange(len(x)),0,1,where=off,transform=ax.get_xaxis_transform(),color='#e7ad54',alpha=.28,lw=0)
 ax.axhline(0,color='.6',lw=.5)
 ax.set_ylabel(label,rotation=0,ha='right',va='center',labelpad=6,fontsize=7)
 ax.tick_params(labelsize=6,length=2,pad=2)
 ax.set_xlim(0,999)
axes[0].set_yticks([-half_track_width,0,half_track_width])
axes[0].axhline(half_track_width,color='.5',lw=.65,ls='--')
axes[0].axhline(-half_track_width,color='.5',lw=.65,ls='--')
axes[1].set_xlabel('Rollout step',fontsize=7,labelpad=3)
fig.legend(handles=[Patch(facecolor='#e7ad54',alpha=.4,label='Off-track: |offset| > half_track_width')],loc='upper center',bbox_to_anchor=(.59,1.02),frameon=False,fontsize=6.5)
fig.savefig(ROOT/'figures/generated/recovery_edit_traces.pdf')
print('Off-track frames:',off.sum(),'center-directed:',((a*x>0)&off).sum())
