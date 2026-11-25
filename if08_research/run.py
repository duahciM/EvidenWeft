"""One frozen IF08 run, isolated spatial and P2 arms."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

from p1.io import dump
from v05.engine import export_native,resample,sheet
from if06_checkpoint.run import old_pool,draw_pairs
from if07_research.run import prior_frames,peak_mb,CFG as PRE_CFG,OLD
from identity_catalog_matching.common import read,sha
from . import bridges,chain
from .freeze import ROOT,HERE

PRE=ROOT/'if07_research_20261119'
CFG=read(HERE/'protocol.json')


def read_ledger(path,expected):
    # Hash input bytes while streaming; keep only numeric geometry, scalar motion and HSV when needed.
    import hashlib
    state=hashlib.sha256();raw={};motion=[];hsv={}
    with path.open('rb') as f:
        for line in f:
            state.update(line);r=json.loads(line)
            k=r['source_channel']
            if k=='SIFT':raw[tuple(r['subject_ids'])]=bridges.unpack(r)
            elif k=='HSV':hsv[tuple(r['subject_ids'])]=r['measurement']
            elif k=='registered_motion':motion.append(r['measurement'])
    if expected!=state.hexdigest():raise ValueError('Ledger SHA256 differs from IF07 B1 lock')
    return raw,motion,hsv


def replay_four(out,case,path,frames,raw,hsv):
    legacy='N04' if case=='room_corridor_dev' else case
    if legacy not in ('N01','N02','N03','N04'):return
    a=read(PRE/'AB'/legacy/'A1.json')['all_nonlocal_pair_gate']
    rename={'no_verified_geometry':'geometry_unverified','insufficient_strength':'insufficient_match_strength',
            'insufficient_full_quadrants':'insufficient_recorded_quadrants'}
    full=[]
    for q in a:
        i,j=q['samples'];p=raw[i,j]
        value=bridges.gate(p,frames[0]['gray'].shape,PRE_CFG['R'])
        if p['quality'] and p['verified'] and hsv[i,j]>.35:value='insufficient_strength'
        new=rename.get(value,value)
        full.append({'id':q['pair'],'frozen':q['status'],'replayed':new})
    fail=[q for q in full if q['frozen']!=q['replayed']]
    dump(out/'A_four'/(legacy+'.json'),{'state':'matched' if not fail else 'not_exact','rows':len(full),'differences':fail,
         'historical_dispositions':dict(Counter(x['frozen'] for x in full)),'replay_dispositions':dict(Counter(x['replayed'] for x in full)),
         'A0_or_A1_root':'IF07 A1 exact-gate replay; N04 shares original source, A2 time layer not used in this A gate'})
    if fail:raise AssertionError(f'{legacy}: {len(fail)} A1 gate replay differences')


def spatial(out,case,source,frame_input):
    frames,meta=frame_input
    folder=PRE/'C'/case;dst=out/'IF08'/case;dst.mkdir(parents=True)
    for n in ('frames.json','segments_coarse.json','segments_refined.json','keyframes.json','integrated_contact.png','baseline_v_contact.png','baseline_uniform_contact.png'):
        shutil.copyfile(folder/n,dst/n)
    fsha=read(folder/'freeze_B1.json')
    assert sha(folder/'frames.json')==fsha['pool_sha256'] and sha(folder/'scene_ids.json')==fsha['direct_lock']
    assert sha(folder/'segments_refined.json')==fsha['legacy_timing_sha256']
    d=root_time=time.perf_counter()
    raw,motion,hsv=read_ledger(folder/'evidence_ledger.jsonl',fsha['ledger_sha256'])
    elapsed_read=time.perf_counter()-d
    if len(raw)!=len(frames)*(len(frames)-1)//2:raise ValueError('C ledger does not cover the frozen pool')
    replay_four(out,case,folder,frames,raw,hsv)
    coarse,ref,rows=[read(folder/n) for n in ('segments_coarse.json','segments_refined.json','scene_ids.json')]
    rows=rows['assignments']
    assert [(x['source_frame'],x['pts']) for x in frames]==[(x['source_frame'],x['pts']) for x in read(dst/'frames.json')['frames']]
    key=read(dst/'keyframes.json');assert len(key['frames'])==8
    old=read(folder/'anchor_pose_graph.json')
    edits={r['sample'] for r in fsha['suspected_abrupt_conflicts']}
    d=time.perf_counter()
    labels,models,views,edges=bridges.models(frames,old['pose_hypotheses'],old['landmarks'],CFG['models'])
    dump(dst/'local_models.json',{'models':models,'primary_model_by_sample':labels,'not_a_scene_map':True})
    tmodel=time.perf_counter()-d
    d=time.perf_counter()
    groups,proposals,gates=bridges.proposals(frames,raw,labels,models,ref['occurrences'],edits,PRE_CFG['R'],CFG['bridge'])
    dump(dst/'common_bridge_candidates.json',{'groups':groups,'nonlocal_wide_candidate_frames':proposals,'gates':gates,'new_pair_requests':0,
         'ledger_hash':fsha['ledger_sha256']})
    tpropose=time.perf_counter()-d
    d=time.perf_counter()
    b1,b2=bridges.model_bridges(groups,raw,views,frames,PRE_CFG['R'],CFG['bridge'])
    dump(dst/'B1_frame_bridges.json',b1);dump(dst/'B2_model_bridges.json',b2)
    teval=time.perf_counter()-d
    probes=bridges.probe_replay(frames,raw,old,edges,read(folder/'long_closure_diagnostics.json'),b1,b2,labels,PRE_CFG['R'],CFG['bridge'])
    dump(dst/'B0_and_route_comparison.json',probes)
    history=read(folder/'R3/forward.json')['interface']
    d0=bridges.judge(frames,coarse,ref,rows,b2,edits,motion,CFG['judgment'])
    assert d0['before']==history
    assert d0['before']['occurrences']==d0['after']['occurrences'] or d0['revision'] is not None
    dump(dst/'D_judgment.json',d0)
    picked=sorted((b for b in b2 if b['status']=='independently_checked_image_bridge'),
                  key=lambda b:(-max(frames[j]['t']-frames[i]['t'] for i,j in (b['fit_samples'],b['validation_samples'])),b['windows']))[:2]
    if picked:
        if 'rgb' not in frames[0]:
            import numpy as np
            from PIL import Image
            for f in frames:
                with Image.open(f['image']) as pix:f['rgb']=np.asarray(pix.convert('RGB'))
        ov={p['id'].removesuffix(':SIFT'):bridges.legacy_raw(p)|{'samples':p['key']} for p in raw.values()
            if p['id'] in {x[k] for x in picked for k in ('fit_root','validation_root')}}
        draw_pairs(frames,ov,[b[k].removesuffix(':SIFT') for b in picked for k in ('fit_root','validation_root')],dst/'checked_image_bridges.png')
    trace=read(folder/'resample/resample_trace.json');args=trace['request']
    qry=resample(source,args['window'],dst/'resample',meta['frames'],args['fps'],args['budget'])
    assert qry['new_source_frames_vs_coarse']==trace['new_source_frames_vs_coarse'] and qry['selected_new_source_frames']==trace['selected_new_source_frames']
    dump(dst/'timing_controls_lock.json',{'IF07_R3_interface_exactly_replayed_before_D':True,
         'legacy_coarse_sha':sha(dst/'segments_coarse.json'),'legacy_refined_sha':sha(dst/'segments_refined.json'),
         'keyframe_with_frozen_V_uniform_controls_sha':sha(dst/'keyframes.json'),
         'actual_resample_source_frames_match_IF07':True})
    summary={'case':case,'model_windows':len(models),'eligible_models':len(views),'eligible_tracks_stored':sum(len(m.get('candidate_landmark_tracks',[])) for m in models),
             'windows_with_validated_links':len({w for b in b2 if b['status']=='independently_checked_image_bridge' for w in b['windows']}),
             'gate_counts':gates,'groups':len(groups),'pairs':len(raw),'wide_candidate_frames':len(proposals),
             'unique_paid_roots':len({r for row in groups for r in row['root_ids']}),
             'root_budget_unassessed':sum(len(g['not_evaluated_roots']) for g in groups),
             'B1_gates':dict(Counter(b['status'] for b in b1)), 'B2_gates':dict(Counter(b['status'] for b in b2)),
             'B2_heldout_evaluations':sum(len(b.get('evaluations',[])) for b in b2),
             'B2_fit_count':sum('fit_root' in b for b in b2),'D_revision':d0['revision'],'D_decisions':dict(Counter(d['reason'] for d in d0['decisions'])),
             'seconds':{'ledger':elapsed_read,'models':tmodel,'screen':tpropose,'bridges':teval,'case_including_query':time.perf_counter()-root_time},
             'process_peak_working_mib':peak_mb()}
    dump(dst/'stage_summary.json',summary)
    print('IF08',case,'charts',len(views),'groups',len(groups),'verified',summary['B2_gates'].get('independently_checked_image_bridge',0),flush=True)


def states(out,case,source,frame_input):
    frames,meta=frame_input
    prev=PRE/'P2'/case;out=out/'P2_03'/case;out.mkdir(parents=True)
    old=read(prev/'Psmall/keyframes.json'); roots=read(prev/'Psmall/role_trace.json')
    roots['frozen_indices']=[s['sample'] for s in old['frames']]
    assert roots['frozen_pool_sha256']==sha(PRE/'C'/case/'frames.json')
    motion=read(OLD/case/'motion_quality.json')
    start=time.perf_counter();p=chain.select(frames,motion,roots,len(roots['audit']),CFG['state']);p['elapsed_s']=time.perf_counter()-start
    assert len(p['samples'])==8 and p['patch_visits']<=len(roots['audit'])
    new=export_native(meta['source']|{'path':source},frames,p['samples'],out/'native_frames')
    rationale={i:'reused Psmall/P1 selection' for i in roots['frozen_indices']}
    for s in p['sources']:
        for i in s['intermediate']:rationale[i]='intermediate visual evidence from frozen local proposal'
        for x in s.get('extensions',[]):rationale[x['sample']]='verified stable regional plateau '+x['role']
    for f in new:f['selection_reason']=rationale.get(f['sample'],'retained observed regional state')
    refined=read(PRE/'C'/case/'segments_refined.json')
    dump(out/'Pchain_keyframes.json',old|{'frames':new,'unrepresented_intervals':[r for r in refined['segments'] if not any(r['start']<=f['t']<r['end'] for f in new)],
                                       'limitation':'visible local states, not physical object or causal inference'})
    p['controls']={a:str(prev/a/'keyframes.json') for a in ('P0','P1','Psmall','Pquery')}
    p['legacy_actual_Pquery']=[str(z) for z in sorted((prev/'Pquery').glob('*/resample_trace.json'))]
    p['Qchain']='deferred; no combined Pchain + Qchain experiment'
    dump(out/'Pchain_trace.json',p);sheet(frames,p['samples'],out/'Pchain_contact.png')
    print('Pchain',case,p['added'],p['removed'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    o=a.out.resolve();lock=read(o/'history_lock.json');state=o/'run_manifest.json'
    if state.exists():raise SystemExit('Refusing overwrite of first-run output')
    assert all(f['path'] and sha(f['path'])==f['sha256'] for f in lock['media']+list(lock['historical_sources'].values()))
    code={p.name:sha(p) for p in HERE.iterdir() if p.name in ('__init__.py','run.py','bridges.py','chain.py','freeze.py','protocol.json')}
    hashes={'prediction_code_sha256':code,'task_card':lock['card'],'label_hashes_only':{str(x):sha(x) for x in (
        o/'photographer_review_sealed.json',ROOT/'if06_checkpoint_20261118/labels_B_sealed.json',ROOT/'if06_checkpoint_20261118/labels_P2_sealed.json')},
        'source_manifest_sha256':sha(o/'history_lock.json'),'algorithm_prediction_opened_labels':False}
    dump(o/'protocol_code_lock.json',hashes);shutil.copyfile(HERE/'protocol.json',o/'protocol_locked.json')
    (o/'A_four').mkdir()
    m={'status':'running','completed':{'IF08':[],'P2_03':[]},'failures':[]}
    dump(state,m);t=time.perf_counter()
    prior=read(PRE/'run_manifest.json');cases=prior['completed']['C']
    by_sha={r['sha256']:r['path'] for r in lock['media']+list(lock['historical_sources'].values())}
    oldset={x['sha256'] for x in lock['historical_sources'].values()}
    review_history_hashes=oldset|{v['sha256'] for n,v in read(ROOT/'if06_checkpoint_20261118/prediction_manifest.json')['B_sources'].items()}
    dump(o/'media_status.json',{'media_snapshot':lock['media'],'historic_content_count':len(review_history_hashes),'independent_crossing_confirmed':False,
         'extra_unsealed_sources':[x for x in lock['media'] if x['sha256'] not in review_history_hashes],'N04_use':'deduplicated, A only, room_corridor_dev in full run'})
    for c in cases:
        try:
            met=read(PRE/'C'/c/'frames.json')
            source=by_sha[met['source']['sha256']]
            fs,meta=(prior_frames(c,source)[:2] if c in ('N01','N02','N03') else old_pool(c))
        except Exception:
            m['failures'].append({'case':c,'stage':'inputs','traceback':traceback.format_exc()})
            continue
        arms=[('IF08',spatial)]+([('P2_03',states)] if c in prior['completed']['P2'] else [])
        for stage,func in arms:
            try:func(o,c,source,(fs,meta));m['completed'][stage].append(c)
            except Exception:
                err=traceback.format_exc();m['failures'].append({'case':c,'stage':stage,'traceback':err});print(err,flush=True)
    m.update(status='predictions_complete' if not m['failures'] else 'finished_with_failures',
             seconds=time.perf_counter()-t,process_peak_working_mib=peak_mb())
    dump(state,m)
    print('Finished',[(k,len(v)) for k,v in m['completed'].items()],len(m['failures']),flush=True)
    if m['failures']:sys.exit(1)

if __name__=='__main__':main()
