import pandas as pd
from PMCMRplus import gesdTest

def proc(sids_select, errfns):
    pcheck = errfns.str.contains('|'.join(sids_select))
    errfns_select = errfns[pcheck]
    dall = pd.DataFrame()
    for fn in errfns_select:
        d = pd.read_pickle(fn)
        dall = pd.concat([dall, d])
  
    out = gesdTest(dall['angle'], 20)
  
    dall2 = dall[~dall.index.isin(out['ix'].iloc[:7])]
  
    dall_summ = dall2.groupby(['errblock', 'condition']).agg(error=('angle', 'mean'), 
                                                            sd=('angle', 'std'),
                                                            n=('angle', 'count'))
    dall_summ['se'] = dall_summ['sd'] / dall_summ['n']
    dall_summ['upper'] = dall_summ['error'] + dall_summ['sd']
    dall_summ['lower'] = dall_summ['error'] - dall_summ['sd']

    return dall_summ
