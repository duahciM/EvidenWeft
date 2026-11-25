"""Inspect limitations of finished predictions; does not generate or change candidates."""
from collections import Counter
from pathlib import Path
import argparse

import cv2
import numpy as np
from PIL import Image,ImageDraw

from p1.io import dump
from .run import PRE,read,read_ledger
from .bridges import project
from .freeze import ROOT
from .chain import patch_distance


def bridge_diagnostics(o):
    e=read(o/'evaluation.json');clips=[]
    for z in e['spatial']:
        n=z['case']; d=o/'IF08'/n
        prev=read(PRE/'C'/n/'proposal_gate.json')['wide_pairs']
        prior={x+':SIFT' for x in prev}
        cand=read(d/'common_bridge_candidates.json')
        root={p['pair']:p for p in cand['nonlocal_wide_candidate_frames']}
        models={x['id']:x for x in read(d/'local_models.json')['models']}
        chosen=[x for x in read(d/'B2_model_bridges.json') if x['status']=='independently_checked_image_bridge']
        b0=[i for i in z['historical_route_probes'] if i['legacy']['status']=='no_bounded_multi_edge_route']
        only=[p for p in root.values() if p['pair'] not in prior]
        mismatch=[p for p in only if abs((p['t'][1]-p['t'][0])-2.4)>1e-8]
        probes=[]
        for x in b0:
            label=str(x['pair']).split(':')
            i,j=map(int,label[1:])
            m=read(d/'local_models.json')['primary_model_by_sample']
            probes.append({'root':x['pair'],'windows':[models[f'M{m[k]:03d}']['status'] for k in (i,j)],
                'B0_short_same_budget':x['B0_same_budget_graph']['status'],'B1':x['B1']['status'],
                'B1_is_original_direct_edge_only':x['B1'].get('path')==[x['pair']+':SIFT'],'B2':x['B2']['state']})
        clips.append({'case':n,'legacy_wide_root_count':len(prior),'common_root_count':len(root),
            'common_exact_overlap_with_legacy':len(prior&root.keys()),'nominal_2_4s_rounding_added_count':len(only),
            'nonrounding_disagreement':mismatch, 'checked_with_both_roots_in_legacy_exact':sum(x['fit_root'] in prior and x['validation_root'] in prior for x in chosen),
            'checked_fit_or_test_on_legacy_different_supported_IDs':sum(any(all(root[q]['legacy_appearance_ids']) and
              len(set(root[q]['legacy_appearance_ids']))>1 for q in (x['fit_root'],x['validation_root'])) for x in chosen),
            'checked_fit_or_test_across_unexplained_link':sum(any(root[q]['crosses_suspected_unexplained_link'] for q in (x['fit_root'],x['validation_root'])) for x in chosen),
            'unreachable_route_probes':probes})
    dump(o/'bridge_diagnostics.json',clips)


def patch_diagnostics(o):
    target=o/'P2_03/local_object'; c=read(target/'Pchain_trace.json')['sources'][0]
    k=read(PRE/'P2/local_object/Psmall/role_trace.json')['reserved'][0]; frame=read(PRE/'C/local_object/frames.json')['frames']
    with np.load(ROOT/'v06_first_run_20261117/local_object/shared/gray_pixels.npz') as p:g=p['gray'].astype(np.float32)/255
    x,y,r,z=c['roi_xyxy']; readout={a['sample']:a for a in c['trace']}
    first,last=c['source_proposal']; h,w=g[0].shape
    point=[]
    def show(fr,box):
        a=g[fr];dx,dy=readout[fr]['shift_px']
        curr=cv2.warpAffine(a,np.float32([[1,0,dx],[0,1,dy]]),(w,h)) if dx or dy else a
        return patch_distance(g[first, y:z,x:r], curr[y:z,x:r],k['area'])
    for f in frame:
        s=f['sample']; rr=readout[s]
        if rr['state'] in ('registered_patch','stable_image_coordinates'):
            point.append({'sample':s,'t':f['t'],'pts':f['pts'],'registered_patch_distance_to_existing_before':show(s,c['roi_xyxy'])})
    dump(target/'after_run_patch_diagnostics.json',{'anchor':first,'roi_xyxy':c['roi_xyxy'],'source_delta_gate':.04,
                                   'note':'post-prediction measurement only, does not add another candidate or change score','per_frame':point})
    indices=[5,first,last]
    pen=Image.new('RGB',(w*len(indices),h+28),'#162835');ink=ImageDraw.Draw(pen)
    for j,s in enumerate(indices):
        source=frame[s]
        with Image.open(source['image']) as img:pen.paste(img,(j*w,0))
        ink.rectangle((j*w+x,y,j*w+r,z),outline='#ffbb58',width=2)
        ink.text((j*w+6,h+5),f"{'earlier review' if j==0 else 'frozen Psmall'} {source['t']:.3f}s PTS {source['pts']}",fill='white')
    pen.save(target/'roi_failure_comparison.png')


def anchor_budget_diagnostic(o):
    folder=PRE/'C/N03'
    raw,_,_=read_ledger(folder/'evidence_ledger.jsonl',read(folder/'freeze_B1.json')['ledger_sha256'])
    old=read(folder/'anchor_pose_graph.json')
    home=o/'IF08/N03'
    charts=read(home/'local_models.json')['models'];counts=[]
    for m in (charts[0],charts[-1]):
        ref=m['reference_sample']; track={}
        for v in m['observed']:
            i=v['sample']
            if i==ref:continue
            pair=raw[min(i,ref),max(i,ref)]
            rp,sp=pair['xy'] if ref<i else pair['xy'][::-1]
            warped=project(np.asarray(v['to_reference']).reshape(3,3),sp)
            if warped is None:continue
            for u,z in zip(rp,warped):track.setdefault(tuple(u),{})[i]=z
        n=0;agree=0
        for xy,others in track.items():
            if len(others)<2:continue
            n+=1
            points=np.array([xy,*others.values()]); med=np.median(points,axis=0)
            agree+=int(float(np.median(np.linalg.norm(points-med,axis=1)))<=3)
        counts.append({'model':m['id'],'reference':ref,'historic_selected_local_tracks':len(m['candidate_landmark_tracks']),
                       'three_view_reference_points':n,'three_view_reference_points_with_median_le_3px':agree})
    dump(home/'after_run_anchor_budget_diagnostic.json',{'checks':counts,'historic_track_budget':len(old['landmarks']),
        'historic_eligible_tracks':old['eligible_tracks_before_budget'],
        'limitation':'Post-run local exact-keypoint star tally; not substituted into B2 and not rerun; shared keypoints do not verify static instances.'})

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    bridge_diagnostics(a.out);patch_diagnostics(a.out);anchor_budget_diagnostic(a.out);print('Read-only diagnostics rendered',flush=True)

if __name__=='__main__':main()
