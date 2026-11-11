"""Audit and summarize the frozen ten-instance, five-seed experiment.

Safe while a batch is running: complete result.json files only are included;
partial groups are explicitly marked, and inference requires all 50 cases.
"""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/camera_ready_full_five_seed_20261111'
TOLERANCE = 0.001


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    temporary=path.with_suffix('.tmp')
    with temporary.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def exact_signed_rank(differences):
    values=np.array([0 if abs(d)<TOLERANCE else float(d) for d in differences])
    nonzero=values[values!=0]
    ranks=rankdata(abs(np.round(nonzero,3)),method='average')
    total=float(sum(ranks))
    positive=float(sum(ranks[nonzero>0]))
    statistic=min(positive,total-positive)
    permutations=[min(float(sum(ranks*np.array(sign))),total-float(sum(ranks*np.array(sign))))
                  for sign in itertools.product([0,1],repeat=len(nonzero))]
    return {'n_instances':len(values),'nonzero_pairs':len(nonzero),'statistic':statistic,
            'p_two_sided':sum(t<=statistic+1e-9 for t in permutations)/len(permutations),
            'wins':int(sum(values<0)),'ties':int(sum(values==0)),'losses':int(sum(values>0)),
            'mean_difference':float(np.mean(values)) if len(values) else None,
            'note':'Instance is the unit; exact sign enumeration, average tied ranks; zero tolerance 0.001; rank magnitudes rounded to 0.001 to remove numerical solver noise'}


def row_for(run,method):
    variant='corrected' if method in ('AL-RF','RF-H') else 'shared'
    return next(r for r in run['summaries'] if r['method']==method and r['variant']==variant)


def compare(rows,instances,seeds,endpoint,baseline):
    diffs=[]; included=[]; excluded=[]
    for name in instances:
        cases=[rows.get((name,s)) for s in seeds]
        if any(r is None for r in cases):
            excluded.append({'instance':name,'reason':'missing case'})
            continue
        a=[row_for(r,'AL-RF')[endpoint] for r in cases]
        b=[row_for(r,baseline)[endpoint] for r in cases]
        if any(v is None for v in a+b):
            excluded.append({'instance':name,'reason':'no finite incumbent for at least one paired seed'})
            continue
        av=float(np.mean(a)); bv=float(np.mean(b))
        included.append({'instance':name,'AL-RF':av,baseline:bv,'difference':av-bv})
        diffs.append(av-bv)
    return {'seeds':seeds,'endpoint':endpoint,'baseline':baseline,'included':included,'excluded':excluded,
            'signed_rank':exact_signed_rank(diffs)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--require-complete',action='store_true')
    args=ap.parse_args()
    cfg=read(ROOT/'protocols/camera_ready_full_five_seed_20261111.json')
    expected={(name,s) for name in cfg['instances'] for s in cfg['seeds']}
    results={}
    audit=[]; flat=[]
    for path in sorted(OUT.glob('mk*/result.json')):
        r=read(path); key=(r['instance'],r['seed'])
        assert key in expected and key not in results
        assert r['original_replay_parity']
        bankrows=[json.loads(line) for line in (path.parent/'solves.jsonl').read_text().splitlines()]
        bank={x['assignment_id']:x for x in bankrows}
        assert len(bank)==len(bankrows)==r['physical_unique_solves']
        assert all(x['old_corrected_fingerprints_equal'] for x in bankrows)
        assert all(x['schedule_validated'] and math.isfinite(x['makespan']) for x in bankrows if x['has_solution'])
        prep=read(path.parent/'prepared.json')
        train={x['id'] for x in prep['pools']['train']}; test={x['id'] for x in prep['pools']['test']}
        assert len(train)==200 and len(test)==100 and not train&test
        for row in r['summaries']:
            owned=[bank[k] for k in row['assignment_ids']]
            assert len(owned)==row['charged_unique_solves']==row['logical_requests']
            assert len(owned)==(35 if row['method'] in ('AL-RF','HeurSeq-35') else 5)
            value=min((x['makespan'] for x in owned if x['has_solution']),default=None)
            assert value==row['combined_mip_best']
            assert abs(sum(x['optimize_seconds'] for x in owned)-row['charged_optimize_seconds'])<1e-5
            selected=[bank[prep['pools']['test'][i]['id']] for i in row['test_indices']]
            assert row['test_best']==min((x['makespan'] for x in selected if x['has_solution']),default=None)
        for method in ['AL-RF','RF-H','LB','HeurSeq','Random','HeurSeq-35']:
            row=row_for(r,method)
            flat.append({'instance':r['instance'],'seed':r['seed'],'method':method,
                         **{k:row[k] for k in ['test_best','calibration_best','combined_mip_best','final_feasible','calibration_feasible',
                                               'logical_requests','charged_unique_solves','charged_optimize_seconds','fit_predict_seconds']}})
        results[key]=r
        audit.append({'instance':r['instance'],'seed':r['seed'],'unique_solves':len(bank),
                      'feasible':sum(x['has_solution'] for x in bankrows),'validated':sum(x['schedule_validated'] for x in bankrows),
                      'timeouts':sum(x['status']=='TIME_LIMIT' for x in bankrows),'no_incumbent':sum(not x.get('solver_has_incumbent', x['has_solution']) for x in bankrows),
                      'invalid_incumbents':sum(x.get('rejected_invalid_incumbent',False) for x in bankrows),
                      'invocation_seconds':r['current_invocation_seconds']})
    missing=sorted(expected-set(results))
    if args.require_complete:
        assert not missing, missing
    write_csv(OUT/'per_seed_summary.csv',flat)
    aggregates=[]
    for name in cfg['instances']:
        for method in ['AL-RF','RF-H','LB','HeurSeq','Random','HeurSeq-35']:
            matching=[r for r in flat if r['instance']==name and r['method']==method]
            if not matching:
                continue
            row={'instance':name,'method':method,'completed_seeds':len(matching),'expected_seeds':5}
            for field in ['test_best','combined_mip_best','calibration_best','charged_optimize_seconds']:
                available=[r[field] for r in matching if r[field] is not None]
                row[field+'_finite_count']=len(available)
                # Never let a four-seed conditional mean look like a five-seed result.
                full=len(matching)==5 and len(available)==5
                row[field+'_mean']=float(np.mean(available)) if full else None
                row[field+'_sample_sd']=float(np.std(available,ddof=1)) if full else None
            aggregates.append(row)
    write_csv(OUT/'instance_summary.csv',aggregates)
    report={'complete':not missing,'completed_cases':len(results),'expected_cases':len(expected),'missing_cases':missing,
            'physical_unique_solves':sum(x['unique_solves'] for x in audit),'validated_schedules':sum(x['validated'] for x in audit),
            'no_incumbent_solves':sum(x['no_incumbent'] for x in audit),
            'invalid_incumbents_rejected':sum(x['invalid_incumbents'] for x in audit),'case_checks':audit}
    if not missing:
        report['primary']=compare(results,cfg['instances'],cfg['seeds'],'combined_mip_best','HeurSeq-35')
        report['secondary']=compare(results,cfg['instances'],cfg['seeds'],'test_best','HeurSeq')
        report['added_three_descriptive']=compare(results,cfg['instances'],[44,45,46],'combined_mip_best','HeurSeq-35')
    write(OUT/'analysis.json',report)
    status=["# Five-seed budget experiment", "", f"Status: {len(results)}/50 complete.",
            "", "Frozen primary comparison: AL-RF best including calibration vs HeurSeq Top-35; five-seed means per instance.",
            "Equal requested calls; actual solver time reported separately. No equal end-to-end wall-clock claim.",
            "Original historical two-seed means are not mixed with new results. MK06 seeds 42/43 are reused from the compatible audit with recorded provenance.", "",
            f"Physical unique solves: {report['physical_unique_solves']}; validated schedules: {report['validated_schedules']}; no-incumbent calls: {report['no_incumbent_solves']}; invalid incumbents rejected: {report['invalid_incumbents_rejected']}."]
    if not missing:
        for name in ['primary','secondary']:
            s=report[name]['signed_rank']
            status += ["",f"{name}: wins/ties/losses = {s['wins']}/{s['ties']}/{s['losses']}; p={s['p_two_sided']:.6g}; mean difference={s['mean_difference']}."]
        status += ["", "See per_seed_summary.csv, instance_summary.csv and analysis.json for complete results and exclusions."]
    else:
        status += ["", "Batch incomplete: no final significance test or overall superiority conclusion is reported."]
    (OUT/'REPORT.md').write_text('\n'.join(status)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['missing_cases','case_checks','primary','secondary','added_three_descriptive']},indent=2))
    if not missing:
        print(json.dumps({'primary':report['primary']['signed_rank'],'secondary':report['secondary']['signed_rank']},indent=2))


if __name__=='__main__':
    main()
