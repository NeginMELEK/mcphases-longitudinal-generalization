#!/usr/bin/env python3
r"""Frozen amended mcPHASES analysis. OFFLINE, aggregate output only.
Python >=3.10; dependencies: duckdb numpy pandas scipy scikit-learn.
Run:
  py mcphases_frozen_analysis.py --data "D:\New paper\mcphases"

No network code, row-level exports, saved fitted models, or disk spill. Raw data
and fitted pipelines stay in RAM. Input files are never changed. The OS can use
virtual memory and background sync; choose a local nonsynced output location.

Primary: R1->R2, 7 prior days, HR median/IQR + step total, two separate targets.
Secondary: R1-only participants -> returning R2 participants, same core.
Core eligibility and cleaning exactly follow the conservative execution audit:
one valid HR and steps event on every preceding day, full-record conflicts
removed, exact parsed-text duplicates collapsed, valid time-only clocks.
Target eligibility: one unique finite nonnegative value per participant/day.
No target imputation, phase/calendar predictors, RHR, sleep, stress, glucose,
PdG or previous hormone predictors. Date gaps and round boundaries stay intact.

Run stops BEFORE ANY FIT if eligible counts differ from the approved audit.
Both core pipelines and constants freeze before R2 scoring. Confirmatory outputs
are saved first, then participant-independent core outputs. Only then do the
fixed secondary windows, missingness comparator, and multimodal extensions run.
No code path selects features/models based on R2 performance.

Secondary variants, each for BOTH experiments and targets: window1, window3,
missingness-only, core+temperature, core+HRV, core+respiration, core+all three.
They use the same 7-day-core eligible rows; extension missingness never drops
rows. A missing/invalid extension table is recorded as NOT_ASSESSED; core
results are preserved. No unknown schema or clock format is silently guessed.

Implementation conventions fixed before execution:
* daily HR median/IQR use equal weight per distinct timestamp; steps sum.
* HRV and respiration: daily median; temperature: longest nonoverlapping valid
  completed episode by elapsed duration (earliest end breaks a tie).
* 7-day features per daily measure: last, mean3, mean7, SD7 (ddof=1), slope7;
  slope/SD need >=3 observed days. Counts3/7 and missing indicators accompany
  those features. 3-day variant: last, mean3, SD3, slope3, count3 and indicators.
  1-day variant: last, count1 and missing indicator. Counts mean observed days,
  not sample counts. Core missingness-only columns may be constant by design.
* medians for imputation use training rows only; fully missing columns fill 0
  with missing flags preserved. Scaling is training-only participant-weighted.
* row weights n_rows/(n_people*n_rows_for_person): equal total person weight,
  mean weight 1. Ridge on log1p y, SVD solver; alpha=10**[-4,...,4].
* five fixed seeded participant folds per experiment, shared across targets;
  penalty minimizes pooled out-of-fold participant-balanced original-unit MAE;
  numerical ties choose larger alpha. No holdout values enter these choices.
* median baseline is participant-weighted ORIGINAL-unit training median.
* PB RMSE = sqrt(mean participant MSE); signed bias = prediction minus truth.
* calibration is descriptive weighted OLS: observed = intercept+slope*predicted;
  undefined for constant predictions; never used to update predictions.
* Spearman: unweighted mean of within-person coefficients, >=10 pairs and
  nonconstant prediction/truth; report median and included/excluded people.
* 10,000 paired participant bootstrap draws, entire series kept together;
  no model refitting. Confidence intervals describe test-person uncertainty,
  conditional on fitted training data. Primary co-endpoints use 97.5% intervals
  on MAE improvements; all other intervals descriptive 95%.

Round 2 outcome distributions were previously audited; predictive performance
must not be used for redevelopment. This is retrospective lagged prediction,
not verified real-time forecasting or proof of assay equivalence.
"""
from __future__ import annotations
import argparse
import collections
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import warnings

# Bound numerical-library worker counts without creating worker data files.
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")

try:
    import duckdb
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
except ImportError:
    raise SystemExit('Install: py -m pip install duckdb numpy pandas scipy scikit-learn')

SEED=20261008
ALPHAS=10.0**np.arange(-4,5)
NBOOT=10000
NULLS=['','NA','N/A','NaN','nan','NULL','null','None']
CLOCK=r'^([01]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9](\.[0-9]+)?)?$'
SPECS={
 'heart_rate':('heart_rate','bpm','instant'),
 'steps':('steps','steps','instant'),
 'temperature':('computed_temperature','nightly_temperature','episode'),
 'HRV':('heart_rate_variability_details','rmssd','instant'),
 'respiration':('respiratory_rate_summary','full_sleep_breathing_rate','instant'),
}
DAILY_CORE=['hr_median','hr_iqr','steps_total']
ROUND_SQL="""CASE lower(trim(study_interval))
WHEN '2022' THEN 'R1' WHEN '2022.0' THEN 'R1' WHEN '1' THEN 'R1' WHEN '1.0' THEN 'R1' WHEN 'r1' THEN 'R1' WHEN 'round 1' THEN 'R1'
WHEN '2024' THEN 'R2' WHEN '2024.0' THEN 'R2' WHEN '2' THEN 'R2' WHEN '2.0' THEN 'R2' WHEN 'r2' THEN 'R2' WHEN 'round 2' THEN 'R2' END"""

class StopRun(Exception): pass

def qi(s):return '"'+str(s).replace('"','""')+'"'
def qs(s):return "'"+str(s).replace("'","''")+"'"
def num(c):return 'try_cast('+qi(c)+' AS DOUBLE)'
def stem(p):return p.name.lower().removesuffix('.gz').removesuffix('.csv')
def weights(ids):
    ids=np.asarray(ids); _,inv,c=np.unique(ids,return_inverse=True,return_counts=True)
    return len(ids)/(len(c)*c[inv])
def pbmean(v,ids):
    w=weights(ids);return float(np.average(v,weights=w))
def pbmae(y,p,ids):return pbmean(np.abs(y-p),ids)
def wmedian(y,ids):
    w=weights(ids); order=np.argsort(y,kind='stable'); k=np.searchsorted(np.cumsum(w[order]),w.sum()/2,side='left')
    return float(y[order[min(k,len(order)-1)]])
def atomic_csv(rows,path):
    df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    tmp=path.with_suffix('.tmp');df.to_csv(tmp,index=False,float_format='%.10g');os.replace(tmp,path)
def atomic_json(obj,path):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
def filehash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def connect(limit,threads):
    c=duckdb.connect(':memory:');c.execute("SET temp_directory = ''")
    c.execute('SET memory_limit='+qs(limit));c.execute('SET threads='+str(threads))
    c.execute('SET enable_progress_bar=false');c.execute('SET preserve_insertion_order=false')
    return c

def source(c,paths,table,day):
    if len(paths.get(table,[]))!=1:raise StopRun('Missing or ambiguous required table: '+table)
    for obj in ['events','grouped']:
        c.execute('DROP TABLE IF EXISTS '+obj)
    for obj in ['b','b_source','raw']:
        c.execute('DROP VIEW IF EXISTS '+obj)
    path=paths[table][0]
    c.execute('CREATE VIEW raw AS SELECT * FROM read_csv('+qs(str(path))+',header=true,all_varchar=true,nullstr=['+','.join(map(qs,NULLS))+'],parallel=false,ignore_errors=false)')
    cols=[r[0] for r in c.execute('DESCRIBE raw').fetchall()]
    if not {'id','study_interval',day}.issubset(cols):raise StopRun('Required fields missing: '+table)
    if len(cols)!=len(set(cols)) or any(re.sub(r'[^a-z0-9]+','_',x.lower()).strip('_')!=x for x in cols):raise StopRun('Noncanonical schema: '+table)
    c.execute(f'''CREATE VIEW b_source AS SELECT *,nullif(trim(id),'') _id,{ROUND_SQL} _r,
      CASE WHEN isfinite({num(day)}) AND {num(day)}=floor({num(day)}) AND {num(day)}>=1
      THEN try_cast({qi(day)} AS BIGINT) END _d FROM raw''')
    c.execute('CREATE VIEW b AS SELECT * FROM b_source')
    invalid=c.execute('SELECT count(*) FROM b WHERE _id IS NULL OR _r IS NULL OR _d IS NULL').fetchone()[0]
    if invalid:raise StopRun('Invalid identity/round/day records: '+table)
    return cols

def hormones(c,paths):
    cols=source(c,paths,'hormones_and_selfreport','day_in_study')
    if not {'lh','estrogen'}.issubset(cols):raise StopRun('Missing hormone targets')
    h=c.execute('SELECT _id,_r,_d,lh,estrogen FROM b').df()
    members={r:set(h.loc[h._r==r,'_id']) for r in ['R1','R2']}
    targets={};audit=[]
    for name,field in [('LH','lh'),('E3G','estrogen')]:
        v=pd.to_numeric(h[field],errors='coerce');v=v.where(np.isfinite(v)&(v>=0))
        x=h[['_id','_r','_d']].assign(y=v)
        count=x.groupby(['_id','_r','_d']).y.nunique()
        good=set(count[count==1].index)
        x=x.dropna(subset=['y']).drop_duplicates(['_id','_r','_d','y'])
        targets[name]=x.loc[[tuple(row) in good for row in x[['_id','_r','_d']].itertuples(index=False,name=None)]].copy()
        for r in ['R1','R2']:
            audit.append(dict(target=name,round=r,usable_target_days=int((targets[name]._r==r).sum()),conflicting_target_days=int(sum(rr==r and n>1 for (_,rr,_),n in count.items()))))
    return targets,members,audit

def wearable(c,paths,mod):
    table,value,kind=SPECS[mod];day='sleep_end_day_in_study' if kind=='episode' else 'day_in_study'
    cols=source(c,paths,table,day)
    req=['sleep_start_day_in_study','sleep_start_timestamp','sleep_end_day_in_study','sleep_end_timestamp'] if kind=='episode' else ['timestamp']
    if not set(req+[value]).issubset(cols):raise StopRun('Missing event fields: '+table)
    persons=[x[0] for x in c.execute('SELECT DISTINCT _id FROM b ORDER BY _id').fetchall()]
    key=list(dict.fromkeys(['_id','_r','_d']+req)); keysql=','.join(map(qi,key)); allcols=','.join(map(qi,cols))
    complete=' AND '.join(qi(k)+' IS NOT NULL' for k in key)
    valid=f'isfinite({num(value)}) AND {num(value)}'+('>=0' if mod=='steps' else '>0')
    clock='sleep_end_timestamp' if kind=='episode' else 'timestamp'
    timeok=f'regexp_full_match({qi(clock)},{qs(CLOCK)})'
    if kind=='episode':
        st=num('sleep_start_day_in_study')
        start=f'({st}*86400+epoch(try_cast(sleep_start_timestamp AS TIME)))'
        end='(_d*86400+epoch(try_cast(sleep_end_timestamp AS TIME)))'
        timeok+=f' AND regexp_full_match(sleep_start_timestamp,{qs(CLOCK)}) AND isfinite({st}) AND {st}=floor({st}) AND {st}>=1 AND {end}>{start}'
    rows=[];audit=collections.defaultdict(collections.Counter)
    print('Aggregating '+mod+' locally ...',flush=True)
    for ip,pid in enumerate(persons):
        for obj in ['events','grouped']:c.execute('DROP TABLE IF EXISTS '+obj)
        c.execute('CREATE OR REPLACE VIEW b AS SELECT * FROM b_source WHERE _id='+qs(pid))
        c.execute(f'''CREATE TABLE grouped AS SELECT {keysql},count(*) n,
          count(DISTINCT row({allcols})) versions,
          count(DISTINCT {num(value)}) FILTER(WHERE {valid}) nv,
          min({num(value)}) FILTER(WHERE {valid}) measurement,
          count(*) FILTER(WHERE NOT coalesce(({valid}),false)) bad_values
          FROM b WHERE {complete} GROUP BY {keysql}''')
        for r,n,versions,nv,bad in c.execute('SELECT _r,sum(n),count(*) FILTER(WHERE versions>1),count(*) FILTER(WHERE nv>1),count(*) FILTER(WHERE bad_values>0) FROM grouped GROUP BY _r').fetchall():
            audit[r].update(dict(keyed_source_rows=int(n),conflicting_record_keys=versions,measurement_conflict_keys=nv,invalid_value_keys=bad))
        c.execute(f'CREATE TABLE events AS SELECT * FROM grouped WHERE versions=1 AND nv=1 AND bad_values=0 AND ({timeok})')
        if kind=='episode':
            e=c.execute(f'SELECT _id,_r,_d,{start} s,{end} e,measurement FROM events ORDER BY _id,_r,s,e').fetchall()
            active=[];last=None;bad=set()
            for i,row in enumerate(e):
                person=(row[0],row[1])
                if person!=last:active=[];last=person
                active=[j for j in active if e[j][4]>row[3]]
                for j in active:bad.update((i,j))
                active.append(i)
            best={}
            for i,row in enumerate(e):
                if i in bad:continue
                k=row[:3]; rank=(row[4]-row[3],-row[4])
                if k not in best or rank>best[k][0]:best[k]=(rank,row[5])
            rows.extend((*k,v[1]) for k,v in best.items())
        else:
            expr={'heart_rate':'median(measurement),quantile_cont(measurement,0.75)-quantile_cont(measurement,0.25)',
                  'steps':'sum(measurement)','HRV':'median(measurement)','respiration':'median(measurement)'}[mod]
            rows.extend(c.execute(f'SELECT _id,_r,_d,{expr} FROM events GROUP BY _id,_r,_d').fetchall())
        if (ip+1)%10==0:print('  Internal partitions completed: '+str(ip+1),flush=True)
    names={'heart_rate':['hr_median','hr_iqr'],'steps':['steps_total'],'temperature':['temperature'],'HRV':['HRV'],'respiration':['respiration']}[mod]
    df=pd.DataFrame(rows,columns=['_id','_r','_d']+names)
    if df.duplicated(['_id','_r','_d']).any():raise StopRun('Daily aggregation is not unique: '+table)
    counts=[]
    for r in ['R1','R2']:
        f=df[df._r==r]
        counts.append(dict(modality=mod,round=r,**dict(audit[r]),retained_days=len(f),retained_participants=f._id.nunique()))
    return df,counts

def histories(daily,cohort,columns):
    # Exact key lookup: no rolling row positions, interpolation or round bridging.
    look={row[:3]:row[3:] for row in daily[['_id','_r','_d']+columns].itertuples(index=False,name=None)}
    a=np.full((len(cohort),7,len(columns)),np.nan)
    for i,(pid,r,d) in enumerate(cohort[['_id','_r','_d']].itertuples(index=False,name=None)):
        for j in range(7):
            vals=look.get((pid,r,d-7+j))
            if vals is not None:a[i,j]=vals
    return a

def features(history,names,width):
    arr=history[:,-width:,:];result={};misscols=[]
    for j,name in enumerate(names):
        a=arr[:,:,j];finite=np.isfinite(a);count=finite.sum(axis=1)
        feats={'last':a[:,-1]}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            if width>=3:
                feats['mean3']=np.nanmean(a[:,-3:],axis=1)
                if width==7:feats['mean7']=np.nanmean(a,axis=1)
                sd=np.nanstd(a,axis=1,ddof=1);sd[count<3]=np.nan
                feats['sd'+str(width)]=sd
                t=np.arange(width,dtype=float)[None,:]
                tm=np.divide((finite*t).sum(axis=1),count,out=np.zeros(len(a)),where=count>0)
                ym=np.divide(np.where(finite,a,0).sum(axis=1),count,out=np.zeros(len(a)),where=count>0)
                den=np.where(finite,(t-tm[:,None])**2,0).sum(axis=1)
                numer=np.where(finite,(t-tm[:,None])*(a-ym[:,None]),0).sum(axis=1)
                slope=np.divide(numer,den,out=np.full(len(a),np.nan),where=(den>0)&(count>=3))
                feats['slope'+str(width)]=slope
        for label,values in feats.items():
            key=name+'_'+label;result[key]=values
            flag=key+'_missing';result[flag]=(~np.isfinite(values)).astype(float);misscols.append(flag)
        key=name+'_count'+str(width);result[key]=count.astype(float);misscols.append(key)
        if width==7:
            key=name+'_count3';result[key]=finite[:,-3:].sum(axis=1).astype(float);misscols.append(key)
    f=pd.DataFrame(result)
    return f,misscols

class Pipeline:
    def __init__(self,alpha):self.alpha=float(alpha)
    def fit(self,X,y,ids):
        X=np.asarray(X,dtype=float);w=weights(ids)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);self.fill=np.nanmedian(np.where(np.isfinite(X),X,np.nan),axis=0)
        self.fill=np.nan_to_num(self.fill,nan=0.0)
        a=np.where(np.isfinite(X),X,self.fill)
        self.mean=np.average(a,axis=0,weights=w)
        self.scale=np.sqrt(np.average((a-self.mean)**2,axis=0,weights=w));self.scale[self.scale<1e-12]=1
        self.model=Ridge(alpha=self.alpha,solver='svd').fit((a-self.mean)/self.scale,np.log1p(y),sample_weight=w)
        return self
    def predict(self,X):
        X=np.asarray(X,dtype=float);a=np.where(np.isfinite(X),X,self.fill)
        with np.errstate(over='ignore',invalid='ignore'):p=np.maximum(0,np.expm1(self.model.predict((a-self.mean)/self.scale)))
        if not np.isfinite(p).all():raise StopRun('Nonfinite predictions; protocol unchanged, analysis stopped')
        return p

def foldmap(ids):
    p=np.array(sorted(set(ids)),dtype=object)
    if len(p)<5:raise StopRun('Fewer than five training participants')
    np.random.default_rng(SEED).shuffle(p)
    return {pid:int(i%5) for i,pid in enumerate(p)}

def train(X,y,ids,folds):
    ids=np.asarray(ids);fold=np.array([folds[p] for p in ids]);oof=np.full((len(y),len(ALPHAS)),np.nan)
    for k in range(5):
        tr=fold!=k;va=fold==k
        if not tr.any() or not va.any():raise StopRun('Empty participant CV fold')
        if set(ids[tr])&set(ids[va]):raise StopRun('Participant leakage assertion failed')
        for ia,alpha in enumerate(ALPHAS):
            model=Pipeline(alpha).fit(X[tr],y[tr],ids[tr]);oof[va,ia]=model.predict(X[va])
    scores=np.array([pbmae(y,oof[:,i],ids) for i in range(len(ALPHAS))]);best=scores.min()
    chosen=int(np.flatnonzero(np.isclose(scores,best,rtol=1e-12,atol=1e-12))[-1])
    fit=Pipeline(ALPHAS[chosen]).fit(X,y,ids)
    info=[dict(alpha=float(a),R1_grouped_OOF_PB_MAE=float(s),selected=(i==chosen)) for i,(a,s) in enumerate(zip(ALPHAS,scores))]
    return fit,info

def summaries(y,p,ids):
    vals=[]
    for pid in sorted(set(ids)):
        at=ids==pid;a=y[at];b=p[at];corr=np.nan
        if len(a)>=10 and np.ptp(a)>0 and np.ptp(b)>0:corr=float(spearmanr(a,b).statistic)
        vals.append([np.mean(np.abs(a-b)),np.mean((a-b)**2),np.mean(np.abs(np.log1p(a)-np.log1p(b))),np.mean(b-a),
                     np.mean(a),np.mean(b),np.mean(a*b),np.mean(b*b),corr])
    return np.asarray(vals)

def from_summaries(s,multiplicity):
    # rows of multiplicity are participant bootstrap counts, summing to N.
    n=multiplicity.sum(axis=1);a=(multiplicity@s[:,:8])/n[:,None]
    den=a[:,7]-a[:,5]**2
    slope=np.divide(a[:,6]-a[:,4]*a[:,5],den,out=np.full(len(a),np.nan),where=den>1e-12)
    valid=np.isfinite(s[:,8]);cn=multiplicity@valid.astype(float)
    corr=np.divide(multiplicity@np.nan_to_num(s[:,8],nan=0),cn,out=np.full(len(a),np.nan),where=cn>0)
    return {'PB_MAE':a[:,0],'PB_RMSE':np.sqrt(np.maximum(0,a[:,1])),'PB_log1p_MAE':a[:,2],
            'PB_signed_bias':a[:,3],'calibration_intercept':a[:,4]-slope*a[:,5],
            'calibration_slope':slope,'mean_within_participant_Spearman':corr}

def interval(a,level):
    a=np.asarray(a);a=a[np.isfinite(a)]
    if not len(a):return np.nan,np.nan,0
    lo,hi=np.quantile(a,[(1-level)/2,1-(1-level)/2]);return float(lo),float(hi),len(a)

def evaluate(data,pred,baseline,experiment,target,variant,nboot=NBOOT):
    y=data.y.to_numpy(float);ids=data._id.to_numpy();ps=sorted(set(ids));N=len(ps)
    rng=np.random.default_rng(SEED)
    mult=rng.multinomial(N,np.ones(N)/N,size=nboot)
    rows=[];stats={};boots={}
    for name,p in [('ridge',pred),('constant_median',np.full(len(y),baseline))]:
        s=summaries(y,p,ids);point=from_summaries(s,np.ones((1,N)))
        boot=from_summaries(s,mult);stats[name]=point;boots[name]=boot
        for metric,value in point.items():
            lo,hi,nb=interval(boot[metric],.95)
            rows.append(dict(experiment=experiment,target=target,variant=variant,estimator=name,metric=metric,value=float(value[0]),
                             CI_level=.95,CI_low=lo,CI_high=hi,valid_bootstrap_draws=nb,test_participants=N,test_days=len(y)))
        corr=s[:,8];corr=corr[np.isfinite(corr)]
        for metric,value in [('Spearman_included_participants',len(corr)),('Spearman_excluded_participants',N-len(corr)),('median_within_participant_Spearman',np.median(corr) if len(corr) else np.nan)]:
            rows.append(dict(experiment=experiment,target=target,variant=variant,estimator=name,metric=metric,value=float(value),test_participants=N,test_days=len(y)))
    level=.975 if experiment=='temporal' and variant=='core7' else .95
    improvement=boots['constant_median']['PB_MAE']-boots['ridge']['PB_MAE']
    point=float(stats['constant_median']['PB_MAE'][0]-stats['ridge']['PB_MAE'][0])
    base=float(stats['constant_median']['PB_MAE'][0]);lo,hi,nb=interval(improvement,level)
    comp=dict(experiment=experiment,target=target,variant=variant,comparison='constant_minus_model_MAE',improvement=point,
              improvement_pct=100*point/base if base>0 else np.nan,CI_level=level,CI_low=lo,CI_high=hi,valid_bootstrap_draws=nb,
              primary_endpoint_supported=bool(lo>0) if level==.975 else None)
    pct=np.divide(100*improvement,boots['constant_median']['PB_MAE'],out=np.full(nboot,np.nan),where=boots['constant_median']['PB_MAE']>0)
    comp['improvement_pct_CI_low'],comp['improvement_pct_CI_high'],_=interval(pct,level)
    return rows,comp

def paired_comparison(data,core,p,experiment,target,variant):
    ids=data._id.to_numpy();y=data.y.to_numpy(float);a=summaries(y,core,ids)[:,0];b=summaries(y,p,ids)[:,0]
    n=len(a);mult=np.random.default_rng(SEED).multinomial(n,np.ones(n)/n,size=NBOOT)
    diff=(mult@(a-b))/n;lo,hi,nb=interval(diff,.95)
    return dict(experiment=experiment,target=target,variant=variant,comparison='core7_minus_variant_MAE',improvement=float(np.mean(a-b)),CI_level=.95,CI_low=lo,CI_high=hi,valid_bootstrap_draws=nb,note='Descriptive paired comparison on identical test rows')


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data',type=Path,default=Path(r'D:\New paper\mcphases'))
    ap.add_argument('--out',type=Path)
    ap.add_argument('--memory-limit',default='4GB')
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:self_test();return
    root=args.data.resolve()
    if not root.is_dir():raise StopRun('Dataset directory not accessible on this computer')
    if args.threads<1:raise StopRun('Threads must be positive')
    out=(args.out or root.parent/('frozen_analysis_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    if out==root or root in out.parents or out.exists():raise StopRun('Choose a NEW output directory outside the dataset')
    out.mkdir(parents=True)
    # Reproducibility artifacts contain code/settings only, no identities or predictions.
    shutil.copyfile(Path(__file__),out/'mcphases_frozen_analysis.py')
    metadata=dict(protocol='amended_HR_steps_retrospective_lagged_R1_R2',seed=SEED,bootstrap_draws=NBOOT,
                  alpha_grid=ALPHAS.tolist(),status='STARTED',R2_performance_used_for_development=False,
                  participant_level_exports=False,script_sha256=filehash(Path(__file__)),
                  versions={p:importlib.metadata.version(p) for p in ['duckdb','numpy','pandas','scipy','scikit-learn']},
                  python=sys.version.split()[0],implementation_conventions=__doc__)
    atomic_json(metadata,out/'run_manifest.json')
    paths=collections.defaultdict(list)
    for p in root.rglob('*'):
        if p.is_file() and p.name.lower().endswith(('.csv','.csv.gz')):paths[stem(p)].append(p)
    c=connect(args.memory_limit,args.threads);quality=[];fingerprints=[]
    try:
        for table in ['hormones_and_selfreport']+[spec[0] for spec in SPECS.values()]:
            if len(paths.get(table,[]))==1:
                path=paths[table][0];fingerprints.append(dict(table=table,bytes=path.stat().st_size,sha256=filehash(path)))
        atomic_csv(fingerprints,out/'00_input_file_fingerprints.csv')
        targets,members,ha=hormones(c,paths)
        returning=members['R1']&members['R2'];only=members['R1']-members['R2']
        hr,a=wearable(c,paths,'heart_rate');quality+=a
        steps,a=wearable(c,paths,'steps');quality+=a
        daily=hr.merge(steps,on=['_id','_r','_d'],how='outer',validate='one_to_one')
        cohorts={};hist={};counts=[];mismatch=False
        for t,data in targets.items():
            a=histories(daily,data,DAILY_CORE);keep=np.isfinite(a).all(axis=(1,2))
            cohorts[t]=data.loc[keep].reset_index(drop=True);hist[t]=a[keep]
            for label,r,ids,expected in [('R1_all','R1',members['R1'],2981 if t=='LH' else 2980),('R2_all','R2',members['R2'],1677),('R1_only','R1',only,1451)]:
                at=(cohorts[t]._r==r)&cohorts[t]._id.isin(ids);d=cohorts[t][at]
                expected_people=20 if r=='R2' else (22 if label=='R1_only' else None)
                passed=len(d)==expected and (expected_people is None or d._id.nunique()==expected_people)
                mismatch |= not passed
                counts.append(dict(target=t,cohort=label,days=len(d),participants=d._id.nunique(),expected_days=expected,expected_participants=expected_people,count_gate='PASS' if passed else 'FAIL'))
        roster_ok=len(members['R1'])==42 and len(members['R2'])==20 and len(returning)==20 and len(only)==22
        atomic_csv(counts,out/'01_eligibility_counts.csv');atomic_csv(quality,out/'02_cleaning_aggregates.csv');atomic_csv(ha,out/'03_target_integrity.csv')
        if mismatch or not roster_ok:raise StopRun('Approved membership/eligibility count mismatch. No fitting performed; inspect aggregate count files. Do not alter the protocol to force a match.')
        folds={'temporal':foldmap(members['R1']),'independent':foldmap(only)}
        jobs={};cvrows=[];fitrows=[];schemas=[]
        # ALL core fits finish before any R2 predictions or scoring.
        for experiment in ['temporal','independent']:
            for t,data in cohorts.items():
                X,miss=features(hist[t],DAILY_CORE,7)
                trainmask=(data._r=='R1') & (data._id.isin(only) if experiment=='independent' else True)
                testmask=data._r=='R2'
                tr=data[trainmask];te=data[testmask]
                if experiment=='independent' and set(tr._id)&set(te._id):raise StopRun('Participant independence assertion failed')
                model,cv=train(X.to_numpy()[trainmask],tr.y.to_numpy(float),tr._id.to_numpy(),folds[experiment])
                base=wmedian(tr.y.to_numpy(float),tr._id.to_numpy())
                jobs[(experiment,t)]=dict(model=model,baseline=base,test=te.reset_index(drop=True),Xtest=X.to_numpy()[testmask],trainmask=trainmask,testmask=testmask)
                cvrows.extend(dict(experiment=experiment,target=t,variant='core7',**row) for row in cv)
                fitrows.append(dict(experiment=experiment,target=t,variant='core7',alpha=model.alpha,train_participants=tr._id.nunique(),train_days=len(tr),test_participants=len(set(te._id)),test_days=len(te),features=X.shape[1],baseline_median=base))
                schemas.extend(dict(variant='core7',feature=s) for s in X.columns)
        atomic_csv(cvrows,out/'04_R1_cross_validation.csv');atomic_csv(fitrows,out/'05_frozen_core_models.csv')
        metadata['status']='BOTH_CORE_PIPELINES_FROZEN_BEFORE_R2_SCORING';atomic_json(metadata,out/'run_manifest.json')
        # Confirmatory results saved before the independent or other secondary scoring.
        corepred={};primary_comp=[]
        for experiment,prefix in [('temporal','06_confirmatory'),('independent','07_participant_independent')]:
            metrics=[];comparisons=[]
            for t in ['LH','E3G']:
                j=jobs[(experiment,t)];p=j['model'].predict(j['Xtest']);corepred[(experiment,t)]=p
                mr,cr=evaluate(j['test'],p,j['baseline'],experiment,t,'core7');metrics+=mr;comparisons.append(cr)
            atomic_csv(metrics,out/(prefix+'_metrics.csv'));atomic_csv(comparisons,out/(prefix+'_baseline_comparisons.csv'))
            if experiment=='temporal':
                primary_comp=comparisons
                atomic_json(dict(status='CONFIRMATORY_SAVED',joint_primary_supported=all(r['primary_endpoint_supported'] for r in comparisons),interpretation='Both 97.5% paired participant-bootstrap lower bounds must exceed zero'),out/'06_confirmatory_decision.json')
            print(prefix+' aggregate results saved.',flush=True)
        # Extensions are processed only AFTER confirmatory and independent core saves.
        extensions={};statuses=[];extension_coverage=[]
        for mod in ['temperature','HRV','respiration']:
            try:
                e,a=wearable(c,paths,mod);extensions[mod]=e;quality+=a
                for target,data in cohorts.items():
                    hh=histories(e,data,[mod])[:,:,0]
                    for rr in ['R1','R2']:
                        mask=(data._r==rr).to_numpy();has=np.isfinite(hh[mask])
                        extension_coverage.append(dict(extension=mod,target=target,round=rr,core_eligible_days=int(mask.sum()),days_with_any_extension_history=int(has.any(axis=1).sum()),days_with_complete_extension_history=int(has.all(axis=1).sum()),mean_observed_history_days=float(has.sum(axis=1).mean())))
                statuses.append(dict(extension=mod,status='AVAILABLE',detail='Prior-day lagged exported measurements; vendor provenance not certified'))
            except Exception:
                statuses.append(dict(extension=mod,status='NOT_ASSESSED',detail='Schema/read/resource failure; no substitution or protocol changes'))
        variants=[('window1',[],1),('window3',[],3),('missingness_only',[],7),('plus_temperature',['temperature'],7),('plus_HRV',['HRV'],7),('plus_respiration',['respiration'],7),('plus_all_extensions',['temperature','HRV','respiration'],7)]
        smetrics=[];scompare=[]
        for variant,mods,width in variants:
            if any(m not in extensions for m in mods):
                statuses.append(dict(extension=variant,status='NOT_ASSESSED',detail='Required extension unavailable; core results preserved'));continue
            for experiment in ['temporal','independent']:
                for t,data in cohorts.items():
                    history=hist[t];names=DAILY_CORE.copy()
                    for m in mods:
                        history=np.concatenate([history,histories(extensions[m],data,[m])],axis=2);names.append(m)
                    X,miss=features(history,names,width)
                    if variant=='missingness_only':X=X[miss]
                    j=jobs[(experiment,t)];tr=data[j['trainmask']]
                    model,cv=train(X.to_numpy()[j['trainmask']],tr.y.to_numpy(float),tr._id.to_numpy(),folds[experiment])
                    p=model.predict(X.to_numpy()[j['testmask']])
                    mr,cr=evaluate(j['test'],p,j['baseline'],experiment,t,variant);smetrics+=mr;scompare.append(cr)
                    scompare.append(paired_comparison(j['test'],corepred[(experiment,t)],p,experiment,t,variant))
                    cvrows.extend(dict(experiment=experiment,target=t,variant=variant,**row) for row in cv)
                    fitrows.append(dict(experiment=experiment,target=t,variant=variant,alpha=model.alpha,train_participants=tr._id.nunique(),train_days=len(tr),test_participants=j['test']._id.nunique(),test_days=len(j['test']),features=X.shape[1],baseline_median=j['baseline']))
                    schemas.extend(dict(variant=variant,feature=s) for s in X.columns)
            # Aggregate checkpoints preserve completed variants without saving predictions.
            atomic_csv(smetrics,out/'08_secondary_metrics.csv');atomic_csv(scompare,out/'09_secondary_comparisons.csv')
            print('Secondary '+variant+' saved.',flush=True)
        snapshot=[]
        for original in fingerprints:
            path=paths[original['table']][0];unchanged=(filehash(path)==original['sha256'])
            snapshot.append(dict(table=original['table'],unchanged_during_analysis=unchanged))
        atomic_csv(snapshot,out/'10_input_stability_check.csv')
        if not all(r['unchanged_during_analysis'] for r in snapshot):
            raise StopRun('An input file changed during analysis. Results from this run are invalid; no protocol adaptation performed.')
        if extension_coverage:atomic_csv(extension_coverage,out/'16_extension_coverage_on_core_cohort.csv')
        atomic_csv(cvrows,out/'11_all_R1_cross_validation.csv');atomic_csv(fitrows,out/'12_all_model_settings.csv')
        atomic_csv(pd.DataFrame(schemas).drop_duplicates(),out/'13_feature_dictionary.csv')
        atomic_csv(statuses,out/'14_extension_status.csv');atomic_csv(quality,out/'15_all_cleaning_aggregates.csv')
        metadata['status']='COMPLETE' if all(r['status']=='AVAILABLE' for r in statuses) else 'CORE_COMPLETE_SOME_EXTENSIONS_NOT_ASSESSED'
        atomic_json(metadata,out/'run_manifest.json')
        (out/'README.txt').write_text(__doc__+'\n\n01 verifies the approved cohort counts before fitting.\n'
            '04-05 save Round 1 selection and frozen core settings before test scoring.\n'
            '06 files are confirmatory results; 07 files are participant-independent results.\n'
            '08-09 are prespecified secondary analyses and matched-row comparisons.\n'
            '10-15 provide file fingerprints, development scores, feature names and processing status.\n'
            'NaN/empty calibration or correlation means undefined; never replace with zero.\n'
            'All CI values use paired participant resampling; no day-level independence assumption.\n'
            'No subject-level table, trajectory, prediction, coefficient file, or fitted model is saved.\n',encoding='utf-8')
        print('Finished. Aggregate results saved to the new frozen_analysis directory.',flush=True)
    except Exception as e:
        metadata['status']='STOPPED';metadata['reason']=str(e) if isinstance(e,StopRun) else 'Execution error; raw exception suppressed to prevent participant-data exposure'
        atomic_json(metadata,out/'run_manifest.json')
        raise StopRun(metadata['reason']) from None
    finally:c.close()


def self_test():
    # Tests use synthetic data only; they are not study results.
    rng=np.random.default_rng(10)
    daily=pd.DataFrame([('a','R1',d,float(d),float(d+1),float(d+2)) for d in range(1,9)],columns=['_id','_r','_d']+DAILY_CORE)
    cohort=pd.DataFrame([('a','R1',8),('a','R2',8)],columns=['_id','_r','_d'])
    h=histories(daily,cohort,DAILY_CORE);assert np.isfinite(h[0]).all() and np.isnan(h[1]).all()
    x,_=features(h[:1],DAILY_CORE,7);assert x.hr_median_last.iloc[0]==7 and x.hr_median_slope7.iloc[0]==1
    daily.loc[daily._d==8,'hr_median']=999999
    assert np.array_equal(h[:1],histories(daily,cohort.iloc[:1],DAILY_CORE))
    ids=np.repeat(np.arange(10).astype(str),15);X=rng.normal(size=(len(ids),5));y=np.exp(.2*X[:,0]+1)
    folds=foldmap(ids);m,cv=train(X,y,ids,folds);p=m.predict(X)
    assert len(cv)==9 and np.isfinite(p).all()
    a=Pipeline(1).fit(X,y,ids);before=a.mean.copy();a.predict(X*5);assert np.array_equal(before,a.mean)
    s=summaries(y,p,ids);point=from_summaries(s,np.ones((1,10)))
    assert np.isclose(point['PB_MAE'][0],pbmae(y,p,ids))
    # Unequal days: each participant still has half total weight.
    assert np.isclose(pbmean(np.array([0.,0.,0.,10.]),np.array(['a','a','a','b'])),5)
    # Paired identical predictions must have exactly zero MAE difference.
    assert np.allclose(summaries(y,p,ids)[:,0]-summaries(y,p.copy(),ids)[:,0],0)
    print('Synthetic leakage/weighting/pipeline checks PASS. No restricted dataset accessed.')

if __name__=='__main__':
    try:main()
    except StopRun as e:raise SystemExit(str(e))
    except KeyboardInterrupt:raise SystemExit('Stopped by user. Only completed aggregate checkpoints are valid.')
    except Exception:raise SystemExit('Setup error; no raw exception displayed. Check dependencies and local paths.')
