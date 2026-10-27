import json, time, random, math, argparse
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from calibrated_screening import ROOT,FJSPInstance,make_pool,feature_matrix,build_feature_cols,BatchPredictor,solve_assignment
from calibrated_screening import train_rf
from assignment_generators import AssignmentGenerator

def run_one(name,seed,ntrain,ntest,calibration,verify_time,k):
 inst=FJSPInstance.from_json_file(str(ROOT/'data/instances/brandimarte'/f'{name}.json'))
 train=make_pool(inst,seed,size=ntrain); test=make_pool(inst,seed+10000,size=ntest)
 feats=build_feature_cols(inst.num_machines); bp=BatchPredictor(None,feats,inst)
 seq=__import__('heuristic_sequencer').HeuristicSequencer(inst,seed=seed)
 X=feature_matrix(bp,train,feats); Xt=feature_matrix(bp,test,feats)
 yh=np.asarray([seq.schedule(a,'best')[0] for a in train],float)
 ytest=np.asarray([seq.schedule(a,'best')[0] for a in test],float)
 cache={}; order=np.argsort(yh); idxs=np.unique(np.linspace(0,len(order)-1,min(calibration,len(order)),dtype=int))
 labels=[]
 for pos in order[idxs]:
  rr=solve_assignment(inst,train[int(pos)],verify_time,cache)
  if rr['has_solution'] and rr['makespan'] is not None: labels.append((int(pos),rr))
 if len(labels)>=5:
  direct=RandomForestRegressor(n_estimators=100,max_depth=12,min_samples_leaf=2,random_state=seed,n_jobs=1).fit(X[[i for i,_ in labels]],np.array([r['makespan'] for _,r in labels]))
  direct_test=direct.predict(Xt)
 else: direct_test=np.full(len(test),np.mean([r['makespan'] for _,r in labels]) if labels else np.inf)
 lb=np.maximum(Xt[:,feats.index('max_machine_load')],Xt[:,feats.index('max_job_time')])
 rf=train_rf(X,yh,seed); rf_test=rf.predict(Xt)
 scores={'AL-RF':direct_test,'RF-H':rf_test,'LB':lb,'HeurSeq':ytest,'Random':np.asarray(random.Random(seed+70001).sample(range(len(test)),len(test)),float)}
 out=[]
 for m,v in scores.items():
  inds=np.argsort(v)[:k]; sel=[]
  for i in inds: sel.append(solve_assignment(inst,test[int(i)],verify_time,cache))
  fs=[x['makespan'] for x in sel if x['has_solution'] and x['makespan'] is not None]
  out.append({'scorer':m,'best':min(fs) if fs else None,'feasible':len(fs),'labels':len(labels)})
 return {'instance':name,'seed':seed,'rows':out}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--instances',nargs='+',default=['mk01','mk04','mk06','mk08']); ap.add_argument('--seeds',nargs='+',type=int,default=[42,43]); ap.add_argument('--output',default='results/active_screening_dev.json'); ap.add_argument('--verify-time',type=float,default=5); ap.add_argument('--n-train',type=int,default=200); ap.add_argument('--n-test',type=int,default=100); ap.add_argument('--calibration',type=int,default=30); ap.add_argument('--k',type=int,default=5); a=ap.parse_args(); out=[]
 for n in a.instances:
  for s in a.seeds:
   print(n,s,flush=True); out.append(run_one(n,s,a.n_train,a.n_test,a.calibration,a.verify_time,a.k)); Path(a.output).write_text(json.dumps(out,indent=2))
 print('saved',len(out))
if __name__=='__main__': main()


