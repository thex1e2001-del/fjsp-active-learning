from pathlib import Path
p=Path(__file__).parent
import matplotlib.pyplot as plt
import numpy as np
instances=['mk01','mk02','mk03','mk04','mk05','mk06','mk07','mk08','mk09','mk10']
bks=np.array([40,26,204,60,172,58,139,523,307,198],float)
data={
'AL-RF':[45.0,39.5,240.0,71.0,185.0,110.0,206.5,532.5,377.0,310.5],
'RF-H':[45.0,44.5,242.0,71.0,185.0,113.0,208.5,540.0,380.5,312.0],
'LB':[45.0,40.5,241.5,71.0,184.5,111.0,204.0,537.0,382.0,329.0],
'HeurSeq':[45.0,39.5,243.0,71.0,186.0,114.0,208.5,531.5,376.5,313.0],
'Random':[48.0,40.5,247.5,71.0,192.5,116.5,209.0,558.5,380.5,314.5],
}
styles={'AL-RF':dict(color='#0072B2',marker='o',lw=2.2),'RF-H':dict(color='#D55E00',marker='s',lw=1.3,ls='--'),'LB':dict(color='#009E73',marker='^',lw=1.3,ls='-.'),'HeurSeq':dict(color='#CC79A7',marker='D',lw=1.3,ls=':'),'Random':dict(color='#6f6f6f',marker='x',lw=1.3,ls='--')}
fig,ax=plt.subplots(figsize=(7.0,3.25),dpi=220)
for name,vals in data.items():
    y=np.array(vals)/bks*100
    ax.plot(instances,y,label=name,**styles[name],markersize=4)
ax.set_ylabel('Mean best feasible makespan / BKS (%)')
ax.set_xlabel('Brandimarte instance')
ax.set_xticks(range(len(instances)),instances)
ax.grid(axis='y',color='#d9d9d9',lw=.6)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(ncol=5,loc='upper center',bbox_to_anchor=(.5,1.18),frameon=False,fontsize=8,handlelength=2.2,columnspacing=1.0)
fig.tight_layout(rect=[0,0,1,.9])
fig.savefig(str(p/'active_method_comparison.png'),bbox_inches='tight')
