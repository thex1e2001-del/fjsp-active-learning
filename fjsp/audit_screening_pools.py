"""Reconstruct saved candidate pools and audit overlap without MIP solves."""
import argparse
import hashlib
import json
from pathlib import Path

from calibrated_screening import ROOT, FJSPInstance, assignment_key, make_pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/calibrated_screening_five_seeds.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/calibrated_screening_five_seeds.pool_audit.json')
    args = parser.parse_args()
    raw = args.input.read_bytes()
    rows = json.loads(raw)
    checks = []
    for row in rows:
        inst_path = ROOT / 'data/instances/brandimarte' / (row['instance'] + '.json')
        inst = FJSPInstance.from_json_file(str(inst_path))
        meta = row['calibration']
        train = {assignment_key(a) for a in make_pool(inst, row['seed'], meta['n_train'])}
        test = {assignment_key(a) for a in make_pool(inst, row['seed'] + 10000, meta['n_test'])}
        assert len(train) == meta['n_train'] and len(test) == meta['n_test']
        for scorer in row['scorers']:
            assert all(a['assignment_id'] in test for a in scorer['selected']), 'Selected assignment not in reconstructed pool'
        for label in meta['calibration_rows']:
            assert label['assignment_id'] in train, 'Calibration assignment not in training pool'
        checks.append(dict(instance=row['instance'], seed=row['seed'],
                           train_count=len(train), test_count=len(test),
                           overlap_ids=sorted(train & test),
                           instance_sha256=hashlib.sha256(inst_path.read_bytes()).hexdigest()))
    report = dict(source=args.input.name, source_sha256=hashlib.sha256(raw).hexdigest(),
                  runs=len(checks), overlapping_runs=sum(bool(x['overlap_ids']) for x in checks),
                  total_overlap=sum(len(x['overlap_ids']) for x in checks), checks=checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f"Audited {report['runs']} runs; overlapping runs={report['overlapping_runs']}; total overlap={report['total_overlap']}")
    if report['total_overlap']:
        raise SystemExit('Train/test overlap detected; do not claim held-out evaluation')


if __name__ == '__main__':
    main()
