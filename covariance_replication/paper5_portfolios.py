"""Check whether the Appendix E distance-loss mitigation also helps portfolio risk.

Uses the project's existing open-to-close, no-cost portfolio evaluation. This
is a research sensitivity check on the reused public sample, not trading advice.
"""
import argparse
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from market_data import load_market, minute_panel
from models import forecast_grid
from evaluation import portfolio_table


def run(data_dir, output_dir):
    panel=load_market('syncster',data_dir,'2016-01-04','2024-12-31')
    reference=minute_panel('SPY','syncster',data_dir,'2016-01-04','2024-12-31').index
    output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for calendar in [False,True]:
        for transformation in ['levels','log_fisher']:
            label=f'{transformation}_{"calendar_safe" if calendar else "retained"}'
            print('RUNNING',label,flush=True)
            forecast=forecast_grid(panel,horizons=(1,),transformation=transformation,reference_calendar=reference if calendar else None)[1]
            table=portfolio_table(forecast,panel.portfolio_return,selected=['HAR-DRD','GHAR(GL,L)','HAR-Cholesky'])
            rows.append(table.assign(variant=label,origins=len(forecast.dates)))
            pd.concat(rows,ignore_index=True).to_csv(output_dir/'portfolios.csv',index=False)
            print(table.round(6).to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data_dir,args.output_dir)
