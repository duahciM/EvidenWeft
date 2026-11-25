"""Post-prediction scoring; sealed labels never enter prediction."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import argparse

import numpy as np
from p1.io import dump,read_video
from v07.audit import audit_images
from if06_checkpoint.evaluate import bounds,intersection,query,selected,within
from identity_catalog_matching.evaluate import development_labels
from .freeze import ROOT,HERE,history
from .run import PRE,read,sha


def checkout_labels(case,frames,reviews,events,annotation,points):
    if case in reviews:
        r=reviews[case]
        return {f['sample']:label for f in frames for a,b,label in r['regions'] if within(f['t'],[a,b])},r['transitions'],r['visits']
    labels,reference,_=development_labels(case,frames,annotation,events,points)
    return labels,[reference] if reference else [],[]


def outcomes(lab,tuples):
    def one(x):
        a,b=(lab.get(i) for i in x)
        return ('same_reviewed_space' if a==b else 'different_reviewed_spaces') if a and b else 'unknown_review'
    facts=[one(x) for x in tuples]
    if all(x=='same_reviewed_space' for x in facts):return 'same_reviewed_space'
    if 'different_reviewed_spaces' in facts:return 'different_reviewed_spaces'
    return 'unknown_review'


def scored(lab,interface):
    h={x['sample']:x['scene_hypothesis_id'] for x in interface['sample_to_segment']}; counts=Counter()
    for a,b in combinations(lab,2):
        match=lab[a]==lab[b];name='same' if match else 'different'
        if not (h.get(a) and h.get(b)):counts[name+'_abstained']+=1
        else:
            counts[name+'_decided']+=1
            counts['same_split' if match else 'different_merged']+=int(match!=(h[a]==h[b]))
    return {'review_frames':len(lab),'decided_frames':sum(bool(h.get(i)) for i in lab),**dict(counts)}


def spatial_case(o,c,reviews,events,annotation,pts):
    folder=o/'IF08'/c; base=read(folder/'stage_summary.json'); frames=read(folder/'frames.json')['frames']
    labels,references,vis=checkout_labels(c,frames,reviews,events,annotation,pts)
    b1=read(folder/'B1_frame_bridges.json'); b2=read(folder/'B2_model_bridges.json')
    good=[x for x in b2 if x['status']=='independently_checked_image_bridge']
    bd=[x for x in b1 if x['status']=='direct_frame_bridge_only']
    qc=Counter(outcomes(labels,[x['fit_samples'],x['validation_samples']]) for x in good)
    q1=Counter(outcomes(labels,[x['samples']]) for x in bd)
    audited=[{'windows':b['windows'],'fit':b['fit_root'],'holdout':b['validation_root'],
           'source_t':[[frames[i]['t'] for i in t] for t in (b['fit_samples'],b['validation_samples'])],
           'span_s':max(frames[j]['t']-frames[i]['t'] for i,j in (b['fit_samples'],b['validation_samples'])),
           'fit_holdout_distinct':not(set(b['fit_samples'])&set(b['validation_samples'])),
           'differently_annotated_ends':outcomes(labels,[b['fit_samples'],b['validation_samples']]),
           'geometry_median_px':b['holdout_median_px'],'geometry_p90_px':b['holdout_p90_px']} for b in good]
    d=read(folder/'D_judgment.json')
    ro=read(folder/'B0_and_route_comparison.json')
    t=bounds(folder,references)
    if not references and not vis:t|={'nonoverlap_candidate_count':None,'additional_candidate_count':None}
    vi=[{'role':label,'range':[lo,hi],
         'occurrences':[a['occurrence_id'] for a in d['after']['occurrences'] if intersection(a['range'],[lo,hi])>0]} for lo,hi,label in vis]
    cand=read(folder/'common_bridge_candidates.json')
    if c in ('N01','N02'):
        motion_notes={'geometric_candidates_at_confirmed_same_room_video_scope':sum(not x['old_hsv_pass'] for x in cand['nonlocal_wide_candidate_frames']),
                      'prior_recovered_SIFT_rows_in_source': read(PRE/'C'/c/'freeze_B1.json')['B1']['counts'].get('restored_geometry_outside_hsv_prefilter',0),
                      'prior_recovered_motion_rows_in_source':read(PRE/'C'/c/'freeze_B1.json')['B1']['counts'].get('restored_registered_motion_observations_behind_legacy_guard',0),
                      'manually_verified_anchor_pairs':None,'note':'Photographer confirms room, not each point identity/geometry/motion.'}
    else:motion_notes=None
    return base|{'B1_scene_band_roles':dict(q1),'B2_scene_band_roles':dict(qc),'B2_review_bridges':audited,
            'max_B2_span':max((x['span_s'] for x in audited),default=None),
            'historical_route_probes':ro,'D_scored_before':scored(labels,d['before']),'D_scored_after':scored(labels,d['after']),
            'D_hypotheses':{'before':len(d['before']['scene_hypotheses']),'after':len(d['after']['scene_hypotheses']),
                       'before_occurrences':len(d['before']['occurrences']),'after_occurrences':len(d['after']['occurrences'])},
            'D_status':d['status'],'visits':vi,'frozen_time':t,'same_room_confirmation_scope':motion_notes,
            'fresh_query':query(folder/'resample/resample_trace.json')}


def p_case(o,c,labels):
    d=o/'P2_03'/c;old=PRE/'P2'/c
    z=read(d/'Pchain_trace.json');meta=read(PRE/'C'/c/'frames.json')
    rule=labels.get(c)
    with np.load(ROOT/'v06_first_run_20261117'/c/'shared/gray_pixels.npz') as npz: gr=npz['gray'].astype(np.float32)/255
    arms={}
    for a in ('P0','P1','Psmall','Pquery','V','uniform','Pchain'):
        path=d/'Pchain_keyframes.json' if a=='Pchain' else old/(a if a not in ('V','uniform') else 'P0')/'keyframes.json'
        data=read(path); picks=data['controls'][a.lower()] if a in ('V','uniform') else data['frames']
        role={k:{'available_in_coarse':sum(within(f['t'],b) for f in meta['frames']),
                 'selected':[f['t'] for f in picks if within(f['t'],b)]} for k,b in rule['roles'].items()} if rule else {}
        ns=[f['sample'] for f in picks]
        arms[a]={'indices':ns,'times':[f['t'] for f in picks],'roles':role,
            'paired':bool(all(role[r]['selected'] for r in rule['pair'])) if rule else None,
            'near_duplicate_pairs':int(sum(np.abs(gr[i]-gr[j]).mean()<.02 for i,j in combinations(ns,2)))}
    return {'case':c,'arms':arms,'trace':z}


def evaluate(o):
    manifest=read(o/'run_manifest.json')
    if manifest['status']=='running':raise SystemExit('Predictions still running')
    prev=ROOT/'if06_checkpoint_20261118'
    lv=read(prev/'labels_B_sealed.json');p2=read(prev/'labels_P2_sealed.json')
    creator=read(o/'photographer_review_sealed.json')
    for n,a in creator['source_expectations'].items():
        actual=read(PRE/'C'/n/'frames.json')['source']['sha256']
        if actual!=a['sha256']:raise ValueError('Photographer note source hash changed')
    event={x['id']:x for x in read(ROOT/'artifacts/videos/manifest.json')['videos']}
    annot=read(ROOT/'validation_full_scene_20261116/annotations.json')['clips'][0]['events'][0]
    points=read(ROOT/'p14_scene_prototypes/frame_labels.json')['cases']
    result={'A':[read(f) for f in sorted((o/'A_four').glob('*.json'))],
         'spatial':[spatial_case(o,c,lv,event,annot,points) for c in manifest['completed']['IF08']],
         'P2':[p_case(o,c,p2) for c in manifest['completed']['P2_03']],
         'version':'same-range seen development; bridge validity ≠ true scene grouping',
         'photographer_evidence':creator,'Qchain':'deferred rather than combine or change historical budgets'}
    dump(o/'evaluation.json',result)
    print('Evaluated',len(result['A']),len(result['spatial']),len(result['P2']),flush=True)


def audit(o):
    m=read(o/'run_manifest.json');l=read(o/'history_lock.json');rules=read(o/'protocol_code_lock.json')
    files=history(o)
    diff={'new_files':sorted(files.keys()-l['history'].keys()),'changed_or_missing':sorted(x for x,c in l['history'].items() if files.get(x)!=c)}
    source={c['sha256']:c['path'] for c in l['media']+list(l['historical_sources'].values())}
    media=all(f['path'] and sha(f['path'])==f['sha256'] for f in l['media']+list(l['historical_sources'].values()))
    code=all(sha(HERE/n)==h for n,h in rules['prediction_code_sha256'].items()) and sha(o/'protocol_locked.json')==rules['prediction_code_sha256']['protocol.json']
    rev=all(sha(n)==h for n,h in rules['label_hashes_only'].items())
    timing=[];images=[]
    for c in m['completed']['IF08']:
        d=o/'IF08'/c;q=d/'resample'
        for framespath,keys in [(d/'frames.json',[d/'keyframes.json']), (q/'frames.json',[q/'keyframes.json'])]:
            local=framespath.parent==q
            meta=read(framespath);src=meta['source'];path=source[src['sha256']]
            controls=[f for p in keys for f in selected(p)]
            if not local and c in m['completed']['P2_03']:
                controls+=read(o/'P2_03'/c/'Pchain_keyframes.json')['frames']
            fresh,probe=read_video(path,10 if local else 5,320)
            triples={(f['source_frame'],f['t'],f['pts']) for f in fresh}
            timing.append({'case':c,'fps':10 if local else 5,'pass':probe['time_base']==src['time_base'] and probe['sha256']==src['sha256'] and
                all((r['source_frame'],r['t'],r['pts']) in triples and r['time_base']==src['time_base'] for r in meta['frames']+controls)})
            images.extend(audit_images(src|{'path':path},controls))
        print('Source verified',c,flush=True)
    uniq=list({r['path']:r for r in images}.values())
    result={'pass':not any(diff.values()) and media and code and rev and all(x['pass'] for x in timing) and
            all(x['exact_source_rgb_equal'] and x['file_hash_equal'] for x in uniq),'history':diff,'historical_file_count':len(files),
            'source_hashes':media,'code_and_protocol':code,'evaluation_and_provenance_labels':rev,
            'PTS':timing,'unique_native_files':len(uniq),'native_checks':uniq}
    dump(o/'source_audit.json',result)
    print('AUDIT',result['pass'],len(files),len(timing),len(uniq),flush=True)
    if not result['pass']:raise SystemExit(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--audit',action='store_true')
    a=p.parse_args();o=a.out.resolve()
    if a.audit:audit(o)
    else:evaluate(o)

if __name__=='__main__':main()
