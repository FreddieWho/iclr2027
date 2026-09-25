from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter
import numpy as np

P=Path(__file__).resolve().parent
D=P/'figures'; D.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.labelsize':10,'axes.titlesize':11,'legend.frameon':False,'figure.facecolor':'white','savefig.facecolor':'white'})
BLUE='#387CA3'; ORANGE='#CE8452'; DARK='#263445'; GREEN='#437F72'
def save(fig,name):
 for ext in ['pdf','png','svg']:fig.savefig(D/(name+'.'+ext),dpi=220,bbox_inches='tight',pad_inches=.08)
 plt.close(fig)
def box(ax,x,y,w,h,t,c='#E7EEF2'):
 ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.015,rounding_size=0.025',fc=c,ec='#78909C',lw=.8))
 ax.text(x,y,t,ha='center',va='center',fontsize=10,color=DARK)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#84919B',lw=1.3))
fig,axs=plt.subplots(1,2,figsize=(6.8,2.65),gridspec_kw={'width_ratios':[1,1.16]})
fig.subplots_adjust(left=.02,right=.98,bottom=.05,top=.83,wspace=.22)
for ax in axs:ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
a=axs[0];a.set_title('a  Oracle-defined composition',loc='left',pad=14,fontweight='bold')
box(a,.10,.5,.16,.22,'P\ny = 0');box(a,.47,.82,.18,.23,'A\ny = 0');box(a,.47,.18,.18,.23,'B\ny = 0');box(a,.86,.5,.21,.23,'AB\ny = 1','#F3E4D8')
for st,en in [((.2,.56),(.35,.77)),((.2,.44),(.35,.23)),((.59,.78),(.73,.56)),((.59,.22),(.73,.44))]:arrow(a,st,en)
a.text(.48,-.03,'Two preserving edits; one changing composition',ha='center',fontsize=8)
a=axs[1];a.set_title('b  Two outcomes of endpoint repair',loc='left',pad=14,fontweight='bold')
box(a,.18,.50,.30,.22,'1  1  0')
box(a,.77,.78,.31,.22,'1  1  1','#DCECE5');box(a,.77,.24,.36,.22,'0 1 1 / 1 0 1\nor 0 0 1','#F3E4D8')
arrow(a,(.36,.57),(.58,.74));arrow(a,(.36,.43),(.55,.27))
a.text(.77,.95,'Full repair',ha='center',fontsize=10,color=GREEN)
a.text(.77,.05,'Error relocation',ha='center',fontsize=10,color=ORANGE)
a.text(.18,.31,'Baseline',ha='center',fontsize=9)
a.text(.18,.13,'Bits: correctness\non A, B, AB',ha='center',fontsize=8,color=DARK)
save(fig,'fig1_protocol')

core=json.loads((P.parent.parent/'reports/submission_audit_20260925/core/claims.json').read_text())
row=next(r for r in core['claims'] if r['id']=='CORE-P3-REPAIR-MIGRATION')['recomputation']['values']['P3_by_seed']
seeds=['11','23','47'];full=np.array([row[s]['R_full'] for s in seeds]);migration=np.array([row[s]['M'] for s in seeds])
fig,ax=plt.subplots(figsize=(6.8,2.8));fig.subplots_adjust(left=.12,right=.97,bottom=.19,top=.80)
x=np.arange(3);ax.bar(x,full,.52,color=BLUE,label='Full repair');ax.bar(x,migration,.52,bottom=full,color=ORANGE,label='Error relocation')
for i,s in enumerate(seeds):ax.text(i,full[i]+migration[i]+.025,f"{100*row[s]['M_over_R_endpoint']:.1f}% relocated",ha='center',fontsize=10)
ax.set(xticks=x,xticklabels=[f'Seed {s}\n(n = {row[s]["H_n"]})' for s in seeds],ylim=(0,.60),ylabel='Fraction of baseline failures');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(loc='upper left',bbox_to_anchor=(0,1.27),ncol=2);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15)
save(fig,'fig2_relocation')

seeds=['803','805','806'];clean=[.142857,.25,.214286];direct=[.571429,.607143,.535714];delta=[.428572,.357143,.321428];low=[.222,.172,.138];high=[.633,.556,.517];colors=[BLUE,GREEN,ORANGE]
fig,axs=plt.subplots(1,2,figsize=(6.8,2.8),gridspec_kw={'width_ratios':[1,1.15]});fig.subplots_adjust(left=.10,right=.98,bottom=.22,top=.82,wspace=.42)
a=axs[0]
for i,s in enumerate(seeds):a.plot([0,1],[clean[i],direct[i]],'-o',lw=1.8,color=colors[i],label=s,ms=5)
a.set(xticks=[0,1],xticklabels=['Clean only','Full singleton'],xlim=(-.2,1.2),ylim=(0,.73),ylabel='Joint correctness $J_3$');a.set_title('a  Same exposure, different supervision',loc='left',fontsize=10,pad=12);a.legend(title='Seed',fontsize=8,title_fontsize=8,loc='upper left',bbox_to_anchor=(0,.97));a.grid(axis='y',alpha=.15)
a=axs[1]
for i,s in enumerate(seeds):a.errorbar(delta[i],2-i,xerr=[[delta[i]-low[i]],[high[i]-delta[i]]],fmt='o',color=colors[i],capsize=4,lw=1.7,ms=5)
a.set(yticks=[2,1,0],yticklabels=seeds,xlim=(-.02,.7),ylim=(-.5,2.5),xlabel='Paired difference in $J_3$');a.axvline(0,ls='--',lw=1,color='#7E8B93');a.set_title('b  Gain with parent-cluster intervals',loc='left',fontsize=10,pad=12);a.grid(axis='x',alpha=.15)
save(fig,'fig3_visual')
(D/'FIGURE_SOURCES.json').write_text(json.dumps({'fig1_protocol':{'type':'conceptual schematic','data':'No empirical or simulated measurements; binary correctness accounting'},'fig2_relocation':{'source':'reports/submission_audit_20260925/core/claims.json','fields':'CORE-P3-REPAIR-MIGRATION recomputation.values.P3_by_seed','uncertainty':'Point estimates; intervals in supplement'},'fig3_visual':{'source':'reports/submission_audit_20260925/routes/AUDIT.md','quartets':28,'parents':19,'uncertainty':'95% parent-cluster percentile CI, 10000 draws, existing estimates copied without recomputation'}},indent=2)+'\n')
print('3 figures saved as PDF, PNG, SVG')
