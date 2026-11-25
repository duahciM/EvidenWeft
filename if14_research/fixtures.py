"""Sealed logical fixtures; these are supplied premises, never detector measurements."""
from pathlib import Path
import argparse

from identity_catalog_matching.common import read,sha
from p1.io import dump
from .judgement import ARMS,case_claim,evidence,identifiers


def definitions():
    def e(k,v,*r):
        return evidence(k,v,r or (f"fixture:{k}",))
    g=e("G","slot_verified_in_comparable_coordinates")
    n=e("N","context_compatible")
    x=e("X","individually_compatible")
    bad=e("X","individual_conflict")
    scope=[
      ("lookalike_across_gap", [x,g,n], [[1,4]], "appearance_compatible_identity_unresolved","unknown",
       "latent treatment is indistinguishable from an unchanged object"),
      ("same_slot_visible_discriminator_conflict",[bad,g,n],[[1,4]],"different_instance_candidate",
       "same_slot_replacement_candidate","weak context cannot reverse strong individual contradiction"),
      ("partial_with_direct_points",[e("X","partially_compatible"),n,e("T","observed_sampled_continuation")],[[1,1.2]],
       "appearance_compatible_identity_unresolved","unknown","partial appearance may be supported while physical identity remains unresolved"),
      ("same_look_changed_neighbors",[x,e("N","context_conflict")],[[1,4]],"appearance_compatible_identity_unresolved","unknown",
       "surroundings difference is separate from scene difference or instance identity"),
      ("no_view_with_compatible_slot",[g],[[1,4]],"unknown","unknown","missing appearance does not vote against identity"),
      ("visible_handoff_with_distinguishing_details",[bad,g,n,e("T","observed_distinct_handoff")],[],
       "different_instance_candidate","observed_distinct_handoff_at_verified_slot","only directly supplied distinct entry/exit supports handoff"),
      ("similar_look_at_different_slots",[e("X","shared_appearance"),e("G","slot_conflict"),n],[[1,4]],
       "appearance_compatible_identity_unresolved","unknown","different slot alone does not split similar/possibly moved object"),
      ("tracked_visible_relative_displacement",[e("X","partially_compatible"),e("G","slot_conflict"),n,e("T","observed_displacement")],[[1,1.1]],
       "appearance_compatible_identity_unresolved","unknown","support observed motion, do not infer identity while between two pictures"),
      ("independent_primary_track",[x,g,n,e("T","observed_sampled_continuation")],[[1,1.1]],
       "individual_association_supported_at_observed_points","unknown","strong individual evidence plus sampled track supports a reversible link at observed times"),
      ("shared_pair_dedup",[e("X","individually_compatible","pair:ab"),e("G","image_slot_compatible","pair:ab"),
         e("N","context_compatible","pair:ab")],[[1,4]],"appearance_compatible_identity_unresolved","unknown","same source alignment cannot vote three times"),
      ("unrooted_conflict",[evidence("X","individual_conflict"),g,n],[[1,4]],"unknown","unknown",
       "state without source proof cannot support contradiction"),
      ("visual_handoff_unverified_location",[bad,e("T","observed_distinct_handoff")],[],
       "different_instance_candidate","unknown","handoff does not prove same slot without comparable location"),
    ]
    return scope


def make():
    packs=[]
    expected={}
    stub=[{"source_frame":100,"pts":100,"time_base":"1/100","t":1.0},
          {"source_frame":400,"pts":400,"time_base":"1/100","t":4.0}]
    for name,obs,gaps,inst,repl,note in definitions():
        packs.append({"comparison_id":name,"origin":"logical_fixture","observations":identifiers(name,stub),
                      "unobserved_open_intervals":gaps,"evidence":obs,"replacement_truth":"unknown"})
        expected[name]={"instance":inst,"replacement":repl,"why":note}
    expected["two_latent_worlds"]={"identical_observation_fixture":"lookalike_across_gap",
       "physical_alternatives":["unchanged","same_appearance_swap_in_gap"],"evaluation":"non-identifiability"}
    return packs,expected


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    folder=a.out/"R_logic_fixtures"
    folder.mkdir(exist_ok=False)
    fixtures,truth=make()
    dump(folder/"input.json",fixtures)
    dump(folder/"evaluation_sealed.json",truth)
    dump(folder/"lock.json",{"protocol":sha(Path(__file__).with_name("protocol.json")),
         "evidence":sha(folder/"input.json"),"sealed_rules":sha(folder/"evaluation_sealed.json"),
         "predictor":sha(Path(__file__).with_name("judgement.py"))})
    outputs={x["comparison_id"]:{k:case_claim(x,k) for k in ARMS} for x in read(folder/"input.json")}
    dump(folder/"results.json",outputs)
    # Only the evaluator opens the expected fixture states.
    truth=read(folder/"evaluation_sealed.json")
    audit=[]
    for k,res in outputs.items():
        r=res["R3"]
        exp=truth[k]
        ok=(r["instance"]["state"]==exp["instance"] and r["replacement"]==exp["replacement"]
            and r["episode"]["absence_of_replacement_between_samples"] is None
            and not r["instance"]["physical_id_merge"] and r["scene"]["scene_assignment"] is None)
        audit.append({"case":k,"pass":ok,"R3":r["instance"]["state"],"replacement":r["replacement"]})
    dup=outputs["shared_pair_dedup"]["R3"]["ablation"]["fixed_4_4_2_1"]
    veto=outputs["same_slot_visible_discriminator_conflict"]["R3"]["ablation"]["fixed_4_4_2_1"]
    safety={"duplicate_score_is_4":dup["usable_association_rank"]==4 and len(dup["de_duplicated"])==2,
            "strong_conflict_veto":veto["usable_association_rank"] is None,
            "same_observation_leaves_both_worlds_open":
                outputs["lookalike_across_gap"]["R3"]["hypotheses"]["same_instance"] ==
                outputs["lookalike_across_gap"]["R3"]["hypotheses"]["same_slot_replacement"]=="open",
            "environment_ablation_preserves_verdict":
                all(r["R3"]["ablation"]["ranking_cannot_change_verdict"] for r in outputs.values())}
    dump(folder/"score.json",{"logic_only":True,"checks":audit,"invariants":safety,
                             "passed":sum(x["pass"] for x in audit),"total":len(audit),
                             "real_replacement_denominator":0})
    print("fixtures",sum(x["pass"] for x in audit),len(audit),safety,flush=True)
    assert all(x["pass"] for x in audit) and all(safety.values())


if __name__=="__main__":
    main()
