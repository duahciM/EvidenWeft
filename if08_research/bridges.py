"""12e: immutable IF07 observations -> local charts -> independently checked sparse bridges."""
from collections import Counter,defaultdict,deque
from itertools import combinations
import math
from time import perf_counter

import cv2
import numpy as np

from if07_research.inference import project,supported_pose
from if06_checkpoint.spatial import space_interface
from p14_scene_prototypes.features import hull_coverage


def unpack(record):
    s=record['measurement']; m=s.get('inlier_pairs',[])
    a=np.float32([x['query_xy'] for x in m]).reshape(-1,2) if s['verified'] else np.empty((0,2),np.float32)
    b=np.float32([x['reference_xy'] for x in m]).reshape(-1,2) if s['verified'] else np.empty((0,2),np.float32)
    return {'key':tuple(record['subject_ids']),'id':record['record_id'],'verified':s['verified'],
        'count':s.get('inlier_count',0),'fraction':s.get('inlier_fraction',0.),
        'hull':[s.get('query_hull_coverage',0.),s.get('reference_hull_coverage',0.)],
        'xy':(a,b),'quality':record['quality_eligible'],'distribution':record['distribution_state'],
        'applies':record['coordinates_are_complete'],'old_color_pass':record['old_hsv_prefilter_pass']}


def gate(p,shape,r):
    if not p['quality']:return 'low_quality'
    if p['distribution']=='old_computation_failed':return 'not_evaluable_computation'
    if not p['verified']:return 'no_verified_geometry'
    if p['count']<r['wide_min_inliers'] or p['fraction']<r['wide_inlier_fraction']:return 'insufficient_strength'
    if min(p['hull'])<r['wide_min_hull']:return 'localized_hull'
    if not p['applies'] or len(p['xy'][0])!=p['count']:return 'budget_unassessed' if p['distribution']=='budget_unassessed' else 'distribution_unavailable'
    h,w=shape
    if min(len({(x>=w/2,y>=h/2) for x,y in z}) for z in p['xy'])<r['wide_quadrants']:return 'insufficient_full_quadrants'
    return 'wide_direct'


def legacy_raw(p):
    return {'id':p['id'].removesuffix(':SIFT'),'sift':{'inlier_pairs':[{'query_xy':a.tolist(),'reference_xy':b.tolist()} for a,b in zip(*p['xy'])]}}


def bounds(frames,cfg):
    s=frames[0]['t']; last=frames[-1]['t']; stride=cfg['stride_s']; half=cfg['window_s']/2
    labels=[int(math.floor((x['t']-s)/stride+1e-8)) for x in frames]
    windows=[{'id':f'M{k:03d}','range':[s+(k+.5)*stride-half,s+(k+.5)*stride+half],
              'primary':[i for i,j in enumerate(labels) if j==k]} for k in range(max(labels)+1)]
    return labels,windows


def models(frames,poses,tracks,cfg):
    labels,windows=bounds(frames,cfg); edges={tuple(p['samples']):p for p in poses if p['state']=='image_relative_pose'}
    data={}; result=[]
    for w in windows:
        lo,hi=w['range']; pool=[f['sample'] for f in frames if lo<=f['t']<=hi and f['sharpness']>=8]
        scores=[(sum(tuple(sorted((i,j))) in edges for j in pool if i!=j),i) for i in pool]
        if not scores:
            result.append(w|{'status':'unknown_no_clear_views','observed':[]});continue
        _,ref=max(scores,key=lambda v:(v[0],v[1] in w['primary'],-abs(frames[v[1]]['t']-(lo+hi)/2),-v[1]))
        neighbors=[]
        for i in pool:
            p=edges.get(tuple(sorted((i,ref))))
            if p:neighbors.append((i,p))
        neighbors=sorted(neighbors,key=lambda k:(k[0] not in w['primary'],k[1]['held_position_error_px'],abs(k[0]-ref),k[0]))[:cfg['max_views']-1]
        views={ref:(np.eye(3),None,0.)}
        for i,p in neighbors:
            h=np.array(p['homography']).reshape(3,3)
            try:target=h if i<ref else np.linalg.inv(h)
            except np.linalg.LinAlgError:continue
            views[i]=(target,p['dependency_ids'][0],p['held_position_error_px'])
        own=[]
        for t in tracks:
            obs=[o for o in t['observations'] if o['sample'] in views]
            if len(obs)<3:continue
            pts=[]
            for o in obs:
                v=project(views[o['sample']][0],np.array([o['xy']]))
                if v is not None:pts.append(v[0])
            if len(pts)<3:continue
            mid=np.median(pts,axis=0)
            scat=float(np.median(np.linalg.norm(pts-mid,axis=1)))
            if scat>3:continue
            own.append({'parent_track':t['landmark_instance_hypothesis'],'xyz_image_only':mid.tolist(),
                'observations':obs,'median_scatter_px':scat,'root_ids':sorted({v for o in obs if (v:=views[o['sample']][1])})})
        valid=len(views)>=cfg['minimum_views'] and len(own)>=cfg['minimum_locally_tracked_points']
        if valid:data[w['id']]=views
        result.append(w|{'status':'local_image_chart' if valid else 'insufficient_local_multiview_support','reference_sample':ref,
             'observed':[{'sample':i,'pts':frames[i]['pts'],'time_base':frames[i]['time_base'],'t':frames[i]['t'],'source_frame':frames[i]['source_frame'],
                          'image':frames[i]['image'],'to_reference':h.ravel().tolist(),'root_id':d,'local_held_point_error_px':v}
                         for i,(h,d,v) in sorted(views.items())],
             'unseen_or_unaligned':[f['sample'] for f in frames if lo<=f['t']<=hi and f['sample'] not in views],
             'candidate_landmark_tracks':own[:cfg['max_tracks_per_model']],'additional_tracks_budget_unassessed':max(0,len(own)-cfg['max_tracks_per_model']),
             'physical_depth_angle_dynamic_instance':'unknown; local projective chart only'})
    return labels,result,data,edges


def affinity(p,frames):
    a,b=p['xy']
    displacement=np.linalg.norm(b-a,axis=1) if len(a) else None
    bins=[]; h,w=frames[0]['gray'].shape
    for z in (a,b):
        n=Counter((x>=w/2,y>=h/2) for x,y in z)
        bins.append({'occupied':len(n),'max_quadrant_fraction':max(n.values(),default=0)/max(1,len(z))})
    return {'inliers':p['count'],'inlier_fraction':p['fraction'],'hulls':p['hull'],'quadrants':bins,
            'median_image_displacement_px':float(np.median(displacement)) if displacement is not None else None,
            'parallax_dynamic_repeated_texture':'unresolved; coordinate concentration is a proxy, not an object count'}


def proposals(frames,pairs,labels,chart,occ,edits,r,cfg):
    bins=defaultdict(list); hist=Counter(); candidates=[]
    masks={i:o['scene_id'] for o in occ for i in o['samples']}
    for (i,j),p in pairs.items():
        if frames[j]['t']-frames[i]['t']<cfg['long_min_s']-1e-8:continue
        g=gate(p,frames[0]['gray'].shape,r);hist[g]+=1
        if g!='wide_direct':continue
        a,b=labels[i],labels[j]
        if a==b:continue
        bins[a,b].append(p)
        candidates.append({'pair':p['id'],'samples':[i,j],'source_pts':[frames[i]['pts'],frames[j]['pts']],
            'source_frames':[frames[i]['source_frame'],frames[j]['source_frame']],'t':[frames[i]['t'],frames[j]['t']],
            'time_base':frames[i]['time_base'],'windows':[chart[a]['id'],chart[b]['id']],
            'legacy_appearance_ids':[masks.get(i),masks.get(j)],
            'crosses_suspected_unexplained_link':any(i<k<=j for k in edits),
            'legacy_span_overrun':frames[j]['t']-frames[i]['t']>r['local_gap_s']*r['max_path_hops'],
            'old_hsv_pass':p['old_color_pass'],'geometry':affinity(p,frames)})
    ordered=[];by_span=defaultdict(deque)
    # Spread visits across different source times and spans; not just earliest/last endpoints.
    for k,v in sorted(bins.items(),key=lambda kv:(-len(kv[1]),kv[0])):
        by_span[(k[0]//4,(k[1]-k[0])//4)].append(k)
    group_order=sorted(by_span)
    while group_order:
        remaining=[]
        for k in group_order:
            ordered.append(by_span[k].popleft())
            if by_span[k]:remaining.append(k)
        group_order=remaining
    out=[]
    for idx,(a,b) in enumerate(ordered):
        allp=sorted(bins[a,b],key=lambda p:(-min(p['hull']),-p['count'],p['key']))
        admitted=[]
        if idx<cfg['max_groups_per_clip']:
            roots=list(allp)
            used=set()
            while roots and len(admitted)<cfg['max_roots_per_group']:
                k=min(range(len(roots)),key=lambda j:(sum(t in used for t in roots[j]['key']),j))
                p=roots.pop(k);admitted.append(p);used.update(p['key'])
        out.append({'windows':[chart[a]['id'],chart[b]['id']],'all_root_count':len(allp),
            'root_ids': [p['id'] for p in admitted], 'not_evaluated_roots': [p['id'] for p in allp if p['id'] not in {x['id'] for x in admitted}],
            'status':'common_candidate' if admitted else 'group_budget_unassessed'})
    return out,candidates,dict(hist)


def root_pose(p,views_a,views_b):
    i,j=p['key']; a,ar,_=views_a[i]; b,br,_=views_b[j]
    x=project(a,p['xy'][0]);y=project(b,p['xy'][1])
    return x,y,{k for k in (ar,br) if k}


def validate_transform(h,x,y):
    if h is None or x is None or y is None:return None
    try: inv=np.linalg.inv(h)
    except np.linalg.LinAlgError:return None
    p,q=project(h,x),project(inv,y)
    if p is None or q is None:return None
    e=np.maximum(np.linalg.norm(p-y,axis=1),np.linalg.norm(q-x,axis=1))
    return float(np.median(e)),float(np.quantile(e,.9))


def model_bridges(groups,pairs,models,frames,root_cfg,cfg):
    source={p['id']:p for p in pairs.values() if p['verified']}
    b1=[]; b2=[]
    for g in groups:
        a,b=g['windows']; ids=g['root_ids']
        if not ids:
            record={'windows':[a,b],'status':'group_budget_unassessed','considered':[]}
            b1.append(record); b2.append(record);continue
        train1=source[ids[0]]
        b1.append({'windows':[a,b],'status':'direct_frame_bridge_only','pair':train1['id'],'samples':train1['key'],
             'gap_s':frames[train1['key'][1]]['t']-frames[train1['key'][0]]['t'],'independent_chart_validation':False,'considered':ids})
        rec={'windows':[a,b],'considered':ids,'evaluations':[],'status':'unknown_unaligned_models'}
        b2.append(rec)
        if a not in models or b not in models:continue
        ma,mb=models[a],models[b]
        vals=[source[k] for k in ids if source[k]['key'][0] in ma and source[k]['key'][1] in mb]
        if not vals:rec['status']='no_root_aligned_to_both_models';continue
        first=vals[0]; x,y,deps=root_pose(first,ma,mb)
        rec['fit_root']=first['id'];rec['fit_local_dependencies']=sorted(deps)
        if x is None or y is None:rec['status']='unstable_projected_fit_points';continue
        h,_=cv2.findHomography(x,y,0)
        if h is None:rec['status']='degenerate_bridge_transform';continue
        rec['transform']=h.ravel().tolist();rec['status']='no_independent_holdout_pair'
        for test in vals[1:]:
            if len(rec['evaluations'])>=cfg['max_transfer_evaluations_per_group']:
                rec['transfer_cap_reached']=True;break
            if set(first['key'])&set(test['key']):continue
            tx,ty,tdeps=root_pose(test,ma,mb)
            if deps&tdeps or first['id'] in tdeps or test['id'] in deps:continue
            result=validate_transform(h,tx,ty)
            if result is None:
                rec['evaluations'].append({'root':test['id'],'status':'nonfinite_transfer'});continue
            med,p90=result
            # Test points must retain independent spatial spread even after projection into the charts.
            sh=frames[0]['gray'].shape
            cov=[hull_coverage(v,sh) for v in (tx,ty)]
            ok=med<=root_cfg['max_holdout_error_px'] and p90<=cfg['p90_error_px'] and min(cov)>=root_cfg['wide_min_hull']
            rec['evaluations'].append({'root':test['id'],'local_dependencies':sorted(tdeps),
                   'holdout_samples':test['key'],'median_px':med,'p90_px':p90,'projected_hulls':cov,
                   'status':'passed' if ok else 'holdout_prediction_disagrees_or_narrow'})
            if ok:
                rec['status']='independently_checked_image_bridge'
                rec['validation_root']=test['id'];rec['validation_samples']=test['key']
                rec['fit_samples']=first['key'];rec['holdout_median_px']=med;rec['holdout_p90_px']=p90
                rec['source_observations']=[{'role':s,'pair_id':p['id'],'samples':p['key'],'coordinates_first_frame':p['xy'][0].tolist(),
                                             'coordinates_second_frame':p['xy'][1].tolist()}
                                            for s,p in [('fit',first),('independent_view_pair',test)]]
                rec['scene_semantics']='not determined'
                break
            rec['status']='holdout_not_supported'
    return b1,b2


def reach(start,stop,edges,long,budget,max_hops):
    adj=defaultdict(list)
    for i,j in edges:adj[i].append((j,False,f'pair:{i}:{j}:SIFT'))
    for row in long:
        if row['status']!='direct_frame_bridge_only':continue
        i,j=row['samples'];adj[i].append((j,True,row['pair']))
    q=deque([(start,[])]);seen={start};visited=set()
    while q:
        f,path=q.popleft()
        if len(path)>=max_hops:continue
        for j,nonlocal_edge,d in sorted(adj[f],key=lambda x:(x[0]!=stop,not x[1],-x[0],x[2])):
            if j>stop:continue
            if len(visited)>=budget:return {'status':'edge_budget_unassessed','visited':len(visited)}
            visited.add(d)
            if j==stop:
                linked=path+[(d,nonlocal_edge)]
                return {'status':'reachable_via_sparse_direct' if any(flag for d,flag in linked) else 'reachable_via_short_graph',
                        'visited':len(visited),'path':[d for d,flag in linked]}
            if j not in seen:seen.add(j);q.append((j,path+[(d,nonlocal_edge)]))
    return {'status':'unreachable_in_sampled_sparse_graph','visited':len(visited)}


def probe_replay(fs,pairs,graph,edges,frozen,b1,b2,labels,r,cfg):
    by_start=defaultdict(list);by_end=defaultdict(list)
    for (i,j),p in edges.items():by_start[i].append(p);by_end[j].append(p)
    mapping={tuple(p['windows']):p for p in b2}
    out=[]
    for p in frozen:
        i,j=map(int,p['pair'].split(':')[1:])
        b0=supported_pose(i,j,legacy_raw(pairs[i,j]),fs,graph,by_start,by_end,True,'forward',r)
        assert b0==p['relative_loop'],p['pair']
        f=reach(i,j,edges,b1,cfg['probe_latent_graph_edges'],cfg['probe_max_hops'])
        same=reach(i,j,edges,[],cfg['probe_latent_graph_edges'],cfg['probe_max_hops'])
        model=mapping.get((f'M{labels[i]:03d}',f'M{labels[j]:03d}'))
        if model and model['status']=='independently_checked_image_bridge':
            proof={'state':model['status'],'fit':model['fit_root'],'holdout':model['validation_root'],
                   'probe_reused_as_fit':model['fit_root']==pairs[i,j]['id'],
                   'scope':'endpoint windows linked; not composition along intervening frames'}
        else:proof={'state':model['status'] if model else 'no_common_window_candidate'}
        out.append({'pair':p['pair'],'t':p['sources'],'legacy':b0,'B0_same_budget_graph':same,'B1':f,'B2':proof})
    return out


def judge(frames,coarse,refined,rows,b2,abrupt,motion,params):
    valid=[b for b in b2 if b['status']=='independently_checked_image_bridge']
    history=space_interface(coarse,refined,rows,'frozen',None)
    if not valid:return {'status':'not_run_no_qualified_bridge','before':history,'after':history,'revision':None,'decisions':[]}
    occ=refined['occurrences']; by={i:o for o in occ for i in o['samples']}; decisions=[]; candidate=None
    for b in valid:
        fit=b['fit_samples'];hold=b['validation_samples']; a,z=[by.get(i) for i in fit]
        q={'bridge':b['windows'],'roots':[b['fit_root'],b['validation_root']],'result':'unknown'}
        decisions.append(q)
        if not a or not z or by.get(hold[0])!=a or by.get(hold[1])!=z:
            q['reason']='no_two_stable_occurrence_sides_for_both_view_pairs';continue
        if a['scene_id']==z['scene_id']:
            q['reason']='already_same_fallback_hypothesis_no_revision';continue
        lo=min(a['samples']);hi=max(z['samples'])
        if any(lo<t<=hi for t in abrupt) or any(d['transition_accepted'] and d['left_sample']<hi and lo<d['right_sample'] for d in refined['diagnostics']):
            q['reason']='timing_or_discontinuity_conflict_keeps_unknown';continue
        if any(a['end']<d['start']<z['start'] and d['scene_id'] not in (a['scene_id'],z['scene_id']) for d in occ):
            q['reason']='other_intermediate_appearance_protected';continue
        between=[m for m in motion if a['samples'][-1]<m['sample']<=z['samples'][0]]
        if len(between)<2 or not all(m.get('motion_supported') and m.get('registration_valid') for m in between):
            q['reason']='missing_independent_continuous_motion_support';continue
        if candidate is not None:q['reason']='scene_revision_budget';continue
        q.update(result='tentative_same_scene_development_only',reason='wide_independent_bridge_and_separate_registered_motion_support')
        candidate={'occurrences':[a['occurrence_id'],z['occurrence_id']],'source_roots':q['roots'],
            'rollback':'remove this one alias if the geometry or independent continuity is contradicted',
            'before_scene_ids':[a['scene_id'],z['scene_id']],'temporal_occurrences_unchanged':True}
    return {'status':'evaluated','before':history,'after':space_interface(coarse,refined,rows,'frozen',candidate),'revision':candidate,'decisions':decisions}
