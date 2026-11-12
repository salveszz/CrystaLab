# FILE PURPOSE: Independently reconstruct the executed temporal diagnostic from native masks and clocks.
import hashlib,json,math,sys
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

import argparse
parser=argparse.ArgumentParser(description="Audit the executed fixed 80-frame/75-window optical diagnostic")
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
DATA=args.source.resolve()
RUN=args.run.resolve()
if args.output.exists():raise FileExistsError("Choose a fresh audit output")
read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(v) for v in Path(p).read_text().splitlines()]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def near(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)
def mask(p):
    value=cv2.imread(str(p),0);assert value is not None and set(np.unique(value))<={0,255};return value>0
run=read(RUN/'manifest.json');experiment=read(RUN/'experiment.json');cohort=RUN/'cohort'
assert run['status']=='complete' and not run['final_test_scored'] and not run['fitted_new_parameters']
assert sha(ROOT/experiment['area_forecast']['existing_model'])==experiment['area_forecast']['model_file_sha256']
assert sha(RUN/'frozen_forecast/frozen_model.json')==experiment['area_forecast']['model_file_sha256']
model=read(RUN/'frozen_forecast/frozen_model.json');coefficient=model['calibration']['coefficient'];prior=model['calibration']['mean_fraction']
original={r['sample_id']:r for r in rows(DATA/'samples.jsonl')}
samples=rows(cohort/'samples.jsonl');scores=rows(RUN/'frozen_forecast/scores.jsonl')
assert len(samples)==75 and len(scores)==300
assert all(s['split'] in ('train','validation') and not s['family_id'].endswith('005') for s in samples)
assert not (cohort/'observations/optical_temporal_005').exists()
observations={p.name:rows(p/'measurements.jsonl') for p in (cohort/'observations').iterdir()}
assert len(observations)==5 and sum(map(len,observations.values()))==80
by_score={(r['sample_id'],r['method']):r for r in scores};errors=defaultdict(list);state_bias=defaultdict(list)
for sample in samples:
    i=sample['cut_frame_index'];family=sample['family_id'];records=observations[family];r=records[i]
    assert sample['target_frame_index']==i+1 and r['valid'] and not r['requires_inspection']
    near(sample['target_time_source_s']-sample['cut_time_source_s'],.25)
    observed=int(mask(cohort/'observations'/family/r['mask_path']).sum());assert observed==r['projected_area_px']
    truth=[]
    for key,index in (('current',i),('future',i+1)):
        original_row=original[records[index]['source_frame']['source_sample_id']]
        assert original_row['scene_id']==family and original_row['split']==sample['split']
        seed=next(o for o in original_row['objects'] if o['id']=='seed')
        assert sha(DATA/seed['mask_path'])==sha(cohort/sample['label_refs'][key]['mask_path'])
        truth.append(int(mask(DATA/seed['mask_path']).sum()))
    assert sample['labels']['current_proxy_area_px2']==truth[0] and sample['labels']['future_proxy_area_px2']==truth[1]
    rate_gain=None
    if i:
        previous=records[i-1];assert previous['valid'] and not previous['requires_inspection']
        before=int(mask(cohort/'observations'/family/previous['mask_path']).sum())
        rate_gain=(observed-before)*.25/(r['video_time_s']-previous['video_time_s'])
    gains=dict(persistence=0.,recent_rate=rate_gain,train_mean_fraction=observed*prior,
               calibrated_recent_rate=None if rate_gain is None else rate_gain*coefficient)
    state_bias[family].append(observed-truth[0])
    for method,gain in gains.items():
        record=by_score[sample['sample_id'],method]
        if gain is None:assert record['prediction'] is None;continue
        value=max(0.,min(256*256,observed+gain));near(record['prediction']['future_area_px2'],value)
        area_error=value-truth[1];gain_error=value-observed-(truth[1]-truth[0])
        near(record['future_area_error_px2'],area_error);near(record['gain_error_px2'],gain_error)
        relative_error=(value-observed)/observed-(truth[1]-truth[0])/truth[0]
        near(record['relative_gain_error'],relative_error);near(record['current_state_error_px2'],observed-truth[0])
        near(area_error,(observed-truth[0])+gain_error)
        assert record['common_history_support']==bool(i)
        errors[sample['split'],method,'all'].append((family,area_error,gain_error,relative_error))
        if i:errors[sample['split'],method,'common'].append((family,area_error,gain_error,relative_error))
metrics=read(RUN/'frozen_forecast/metrics.json')
for (split,method,selection),values in errors.items():
    report=metrics[split][method]['all_available' if selection=='all' else 'common_history_support']
    assert report['samples']==len(values)
    near(report['future_area_mae_px2'],sum(abs(v[1]) for v in values)/len(values))
    near(report['future_area_rmse_px2'],math.sqrt(sum(v[1]**2 for v in values)/len(values)))
    near(report['gain_mae_px2'],sum(abs(v[2]) for v in values)/len(values))
    near(report['relative_gain_mae'],sum(abs(v[3]) for v in values)/len(values))
    near(report['future_area_signed_error_px2'],sum(v[1] for v in values)/len(values))
    grouped=defaultdict(list)
    for family,error,*_ in values:grouped[family].append(error)
    near(report['family_macro_area_mae_px2'],sum(sum(abs(v) for v in errors)/len(errors) for errors in grouped.values())/len(grouped))
    if selection=='common':
        for family,errors_by_family in grouped.items():
            near(metrics[split][method]['per_family'][family]['future_area_mae_px2'],sum(abs(v) for v in errors_by_family)/len(errors_by_family))
event_cases=read(RUN/'event_summary.json')['cases'];checked_matches=0
for case in event_cases:
    scene=case['scene_id'];report=case['report'];annotation_root=RUN/'annotations'/scene
    truth=read(annotation_root/'annotations.json');analysis=RUN/'candidates'/scene
    predicted=rows(analysis/'candidate_events.jsonl');obs=rows(analysis/'candidate_observations.jsonl')
    reference={};expected_negative=[];prior_present=None
    for original_row in sorted((r for r in original.values() if r['scene_id']==scene),key=lambda r:r['frame_index']):
        present=set();index=original_row['frame_index']
        for obj in original_row['objects']:
            if obj['class_id']==0 or not obj['label_eligible']:continue
            kind='crystal_candidate' if obj['class_id']==1 else 'inclusion_candidate';present.add(kind)
            key=scene+':'+obj['id'];reference.setdefault(key,(index,obj,original_row['sample_id']))
        if prior_present is not None:
            for kind in ('crystal_candidate','inclusion_candidate'):
                if kind not in present and kind not in prior_present:expected_negative.append(dict(candidate_type=kind,start_s=(index-1)/4,end_s=index/4))
        prior_present=present
    assert {g['event_id'] for g in truth['events']}==set(reference) and truth['negative_intervals']==expected_negative
    for target in truth['events']:
        index,obj,sample_id=reference[target['event_id']]
        assert target['source_sample_id']==sample_id
        assert target['onset_interval_source_s']==[None if index==0 else (index-1)/4,index/4]
        assert sha(annotation_root/target['first_visible_mask_path'])==sha(DATA/obj['mask_path'])
    for kind in ('crystal_candidate','inclusion_candidate'):
        gt=[g for g in truth['events'] if g['candidate_type']==kind];assert len(gt)<=1
        p=[e for e in predicted if e['candidate_type']==kind];matched=[]
        for target in gt:
            t=mask(annotation_root/target['first_visible_mask_path']);low,high=target['onset_interval_source_s']
            best=None
            for prediction in p:
                if prediction['alert_time_s']<(low if low is not None else 0) or prediction['alert_time_s']>high+1:continue
                if low is not None and low<high and prediction['first_detection_time_s']<=low:continue
                q=mask(analysis/prediction['first_detection_mask_path']);iou=float((t&q).sum()/(t|q).sum())
                if iou>=.5 and (best is None or iou>best[0]):best=(iou,prediction)
            if best:
                matched.append(best[1]['event_id']);stored=next(v for v in report['matches'] if v['truth_event_id']==target['event_id'])
                near(stored['mask_iou'],best[0]);near(stored['alert_delay_after_first_visible_s'],best[1]['alert_time_s']-high);checked_matches+=1
        assert report['per_kind'][kind]['tp']==len(matched) and report['per_kind'][kind]['fp']==len(p)-len(matched)
        assert report['per_kind'][kind]['fn']==len(gt)-len(matched)
        usable=[]
        for left,right in zip(obs,obs[1:]):
            if left['observable'][kind] and right['observable'][kind] and right['video_time_s']-left['video_time_s']<=1:
                usable.append((left['video_time_s'],right['video_time_s']))
        negative=[(max(a,n['start_s']),min(b,n['end_s'])) for a,b in usable for n in truth['negative_intervals'] if n['candidate_type']==kind and max(a,n['start_s'])<min(b,n['end_s'])]
        near(sum(b-a for a,b in negative),report['negative_coverage'][kind]['usable_annotated_negative_source_seconds'])
        assert sum(e['event_id'] not in matched and any(a<=e['alert_time_s']<=b for a,b in negative) for e in p)==report['negative_coverage'][kind]['false_events_in_negative_intervals']
        eligible=0
        for target in gt:
            start=target['onset_interval_source_s'][1];end=start+1
            covered=sum(max(0.,min(b,end)-max(a,start)) for a,b in usable)
            eligible+=int(end<=obs[-1]['video_time_s']+1e-9 and abs(covered-1)<1e-8)
        assert eligible==report['observable_deadline_followup'][kind]['eligible']
assert checked_matches==sum(c['report']['per_kind'][k]['tp'] for c in event_cases for k in ('crystal_candidate','inclusion_candidate'))
by_split={}
source_splits={r['scene_id']:r['split'] for r in original.values()}
for split in ('train','validation'):
    selected=[c for c in event_cases if source_splits[c['scene_id']]==split]
    assert all(c['split']==split for c in selected)
    by_split[split]={k:{metric:sum(c['report']['per_kind'][k][metric] for c in selected) for metric in ('tp','fp','fn')} for k in ('crystal_candidate','inclusion_candidate')}
    for k,counts in by_split[split].items():
        for metric,value in counts.items():assert run['event_metrics_by_split'][split]['per_kind'][k][metric]==value
result=dict(schema='crystalab_temporal_independent_execution_audit_v1',status='passed',
    source_manifest_sha256=sha(DATA/'manifest.json'),run_manifest_sha256=sha(RUN/'manifest.json'),
    source_observations_checked=80,causal_windows_checked=75,forecast_records_checked=300,event_matches_checked=checked_matches,events_by_split=by_split,
    native_forecast_formulas_and_metrics_reconstructed=True,native_event_iou_and_matching_reconstructed=True,
    source_proxy_event_labels_and_negative_intervals_reconstructed=True,
    class_negative_coverage_and_false_counts_reconstructed=True,
    usable_deadline_counts_reconstructed=True,reserved_scene_replayed=False,model_bytes_unchanged=True,
    mean_current_area_bias_by_family={k:sum(v)/len(v) for k,v in state_bias.items()},
    scope='Independent numeric/mask reconstruction of this executed diagnostic; no physical visibility, real accuracy or control-benefit audit',
    auditor_sha256=sha(Path(__file__)))
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
