"""Trace frozen local change proposals to persistent coarse-pool states."""
from itertools import combinations
from collections import Counter

import cv2
import numpy as np


def region(box,shape,halo):
    x,y,w,h=box
    return max(0,x-round(w*halo)),max(0,y-round(h*halo)),min(shape[1],x+w+round(w*halo)),min(shape[0],y+h+round(h*halo))


def patch_distance(a,b,n):
    d=np.abs(a-b).ravel()
    n=min(len(d),max(1,n))
    return float(np.partition(d,len(d)-n)[-n:].mean())


def trace(frames,motion,proposal,cfg,remaining):
    a,b=proposal['samples']; im=np.stack([f['gray'] for f in frames]).astype(np.float32)/255
    x,y,r,z=region(proposal['bbox_xywh'],im[0].shape,cfg['halo'])
    roi=(slice(y,z),slice(x,r))
    bg=np.ones(im[0].shape,bool);bg[roi]=False
    anchor=im[a]; area=proposal['area']; logging=[]; views={}; n=len(frames)
    if int(bg.sum())<.65*bg.size:return {'status':'not_evaluable_little_surrounding_support','samples':[a,b],'examined':0}
    stable_region=bg.astype(np.float32); reference=anchor*stable_region; operators=motion['pair_metrics']
    far=sorted(range(n),key=lambda i:(0 if a<=i<=b else min(abs(i-a),abs(i-b)),i))[:remaining]
    for i in sorted(far):
        f=frames[i];status=None; shift=[0.,0.]; response=None
        current=im[i]; bgmad=None
        if f['sharpness']<8:status='low_quality'
        k=operators[i] if i<len(operators) else {}
        if not status and k.get('brightness',{}).get('supported') and k.get('raw_mad',0)>.035:
            status='photometric_explained'
        if not status and k.get('registration_valid') and k.get('motion_supported') and max(abs(k.get('dx',0)),abs(k.get('dy',0)))>cfg['max_phase_shift']:
            status='previous_measured_motion_outside_domain'
        if not status:
            bgmad=float(np.mean(np.abs(current[bg]-anchor[bg])))
            if bgmad>.008:
                delta,response=cv2.phaseCorrelate(reference,current*stable_region)
                if not all(np.isfinite(delta)) or max(map(abs,delta))>cfg['max_phase_shift'] or response<.15:
                    status='unreliable_local_registration'
                else:
                    shift=[float(-delta[0]),float(-delta[1])]
                    warp=np.float32([[1,0,shift[0]],[0,1,shift[1]]])
                    current=cv2.warpAffine(current,warp,(anchor.shape[1],anchor.shape[0]))
                    support=cv2.warpAffine(np.ones_like(anchor),warp,(anchor.shape[1],anchor.shape[0]))>=.99
                    if float(support[roi].mean())<cfg['min_coverage']:status='patch_not_visible'
                    bgmad=float(np.mean(np.abs(current[support&bg]-anchor[support&bg])))
            if not status and bgmad>cfg['max_bg_mad']:status='surrounding_structure_changed'
        if not status:
            views[i]=current[roi]
            status='registered_patch' if response is not None else 'stable_image_coordinates'
        logging.append({'sample':i,'pts':f['pts'],'t':f['t'],'state':status,'bg_mad':bgmad,
                        'phase_response':float(response) if response is not None and np.isfinite(response) else None,'shift_px':shift})
    eligible=set(views)
    deb=['earlier','middle','later']
    answer={'source_proposal':[a,b],'examined':len(far),'max_reads':remaining,'roi_xyxy':[x,y,r,z],'trace':logging,'unassessed':sorted(set(range(n))-set(far)),
            'samples':[a,b],'intermediate':[],'status':'frozen_pair_no_reliable_extension','physical_object_and_cause':'unknown'}
    if a not in eligible or b not in eligible:return answer|{'status':'not_evaluable_reference_patch'}
    left=a
    while left-1 in eligible and frames[left]['t']-frames[left-1]['t']<=.61:left-=1
    right=b
    while right+1 in eligible and frames[right+1]['t']-frames[right]['t']<=.61:right+=1
    step=cfg['support_frames']
    states=[]
    for i in range(left,right-step+2):
        block=list(range(i,i+step))
        if not all(k in eligible for k in block):continue
        if frames[block[-1]]['t']-frames[i]['t']<.25:continue
        median=np.median([views[f] for f in block],axis=0)
        pairs=[patch_distance(v,median,area) for v in [views[f] for f in block]]
        medoid=min(zip(pairs,block),key=lambda x:(round(x[0],6),abs(x[1]-np.median(block)),x[1]))[1]
        if max(pairs)<=cfg['max_patch_jitter']:
            states.append((block,median,medoid,max(pairs)))
    changes=[]
    for side,seed,pool in [('earlier',a,[s for s in states if s[0][-1]<a]),('later',b,list(reversed([s for s in states if s[0][0]>b])))]:
        for block,median,medoid,jitter in pool:
            delta=patch_distance(median,views[seed],area)
            if delta<cfg['state_delta']:continue
            relative=patch_distance(median,views[b if side=='earlier' else a],area)
            # Related visual chain remains ambiguous for A->B->A; no collapse of original A/B evidence.
            changes.append({'role':side,'support':block,'sample':medoid,'delta':delta,'jitter':jitter,
                            'recurrent_appearance_vs_other_seed':relative<cfg['state_delta']})
            break
    if changes:
        answer['samples']=[next((x['sample'] for x in changes if x['role']=='earlier'),a),
                           next((x['sample'] for x in changes if x['role']=='later'),b)]
        answer['intermediate']=[i for i in (a,b) if i not in answer['samples']]
        answer.update(status='extended_registered_state_chain',extensions=changes)
    answer.update(connected_visible_sample_range=[left,right],plateau_windows=len(states))
    return answer


def select(frames,motion,small,budget,cfg):
    # The only proposals are the cached Psmall reservations. Reuse their existing states if tracking abstains.
    roots=small['reserved'][:cfg['max_role_proposals']]
    traces=[]; base=small['frozen_indices']; keep=list(dict.fromkeys(i for s in small['broader_P1_roles'] for i in s['samples']))
    work=0
    for p in roots:
        limit=min(len(frames),cfg['max_patch_reads_per_original_proposal'],max(0,budget-work))
        t=trace(frames,motion,p,cfg,limit);traces.append(t);work+=t['examined']
    desired=list(dict.fromkeys(keep+[i for t in traces for i in t['samples']]+[i for t in traces for i in t['intermediate']]+base))
    selected=sorted(desired[:cfg['budget']])
    return {'frozen_proposal_count':len(roots),'same_proposals':True,'patch_visits':work,'max_patch_visits_from_frozen_detection':budget,
            'sources':traces,'samples':selected,'added':sorted(set(selected)-set(base)),'removed':sorted(set(base)-set(selected)),
            'unrepresented_intermediate_due_budget':sorted({i for t in traces for i in t['intermediate']}-set(selected)),
            'unrepresented_proven_roles_due_budget':sorted({i for t in traces for i in t['samples']}-set(selected))}
