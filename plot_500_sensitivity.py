import pandas as pd,numpy as np, matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt
from pathlib import Path
P=Path(__file__).resolve().parent;s=pd.read_csv(P/'results/impact_of_500_monthly.csv')
fig,ax=plt.subplots(figsize=(9,4))
ax.plot(s.month,s.observed_mean,'o-',label='Observed mean',lw=2)
ax.plot(s.month,s.TSAVP_500_retained,'s--',label='Both suspect sites removed; 500 retained',lw=2)
ax.plot(s.month,s.TSAVP_500_omitted,'o-',label='Both sites and 500 removed',lw=2)
ax.set_xticks(range(1,13));ax.set_xlabel('Month (2024)');ax.set_ylabel('PM$_{2.5}$ AQI (index points)');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout();out=P/'results/tehran_figures';fig.savefig(out/'daily_500_sensitivity_monthly.png',dpi=300,bbox_inches='tight');fig.savefig(out/'daily_500_sensitivity_monthly.pdf',bbox_inches='tight');plt.close(fig)
