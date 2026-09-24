"""Production-call regressions and explicit legacy-mutant counterexamples."""
import sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/f095_campaign'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from d10_frontier import (reachable_ladder,policy_outcomes,fit_isotonic,apply_isotonic,
                         paired_own_quality_ci)
from d06_roles import select_frame,audit_roles,quartet_search,labels_batch,motion_cap
from d06_oracle import channel_label
from frontier_repair import frontier
from frontier_repair import (budget_threshold, deployment_bootstrap_ci,
                             deployment_metrics, grouped_outcomes,
                             oracle_label_curve, paired_deployment_ci,
                             pareto_envelope, rows_for, scene_row_matrix)
from fair_continuation import state_metrics

class Regression(unittest.TestCase):
 def test_complete_action_ladder_and_legacy_mutant(self):
  scores=np.array([.4,.9]);cost=np.array([1.,2.]);truth=np.array([True,False]);scene=np.zeros(2,int)
  def witness(ts):return any(policy_outcomes(scores,scene,cost,truth,[0],t)[1][0] for t in ts)
  self.assertTrue(witness(reachable_ladder(scores)))
  self.assertFalse(witness([scores.max()])) # legacy max-only mutant misses cheap success
  outcomes=[policy_outcomes(scores,scene,cost,truth,[0],t) for t in reachable_ladder(scores)]
  self.assertTrue(any(not o[0][0] for o in outcomes))
 def test_pava_training_plateau_and_legacy_mutant(self):
  bx,by=fit_isotonic([0,1,2],[1,0,1]);fit=apply_isotonic(bx,by,[-1,0,1,2,3])
  np.testing.assert_allclose(fit,[.5,.5,.5,1,1])
  legacy=np.interp([0,1,2],[.5,2],[.5,1])
  self.assertFalse(np.allclose(legacy,[.5,.5,1]))
 def test_duplicate_x_pava(self):
  bx,by=fit_isotonic([0,0,1,2],[1,0,0,1])
  np.testing.assert_allclose(apply_isotonic(bx,by,[0,1,2]),[1/3,1/3,1])
 def test_frontier_distinct_actions_and_largest_threshold(self):
  score=np.array([.4,.9]);t,sel,*_=frontier(score,np.array([0,0]),np.array([1,2]),np.array([1,0]),np.array([0]))
  self.assertEqual(len(sel),3)
  self.assertEqual(float(t[np.flatnonzero(sel[:,0]==0)[0]]),.4)
 def test_period_unique_frame_and_tie(self):
  d=pd.DataFrame({'source_match_id':['m']*4,'game_section':['first','first','second','first'],'timestamp_ms':[90,110,100,90],'person_id':['a','a','a','b']})
  x=select_frame(d,'m','first',100)
  self.assertEqual(set(x.timestamp_ms),{90});self.assertEqual(set(x.person_id),{'a','b'})
  legacy=d[np.abs(d.timestamp_ms-100)==np.abs(d.timestamp_ms-100).min()]
  self.assertEqual(set(legacy.game_section),{'second'}) # old global nearest selects wrong period
  with self.assertRaises(ValueError):select_frame(pd.concat([d,d.iloc[[0]]]),'m','first',90)
 def test_true_E_and_production_label_consistency(self):
  p=np.array([0.,0.]);r=np.array([10.,0.]);D=np.array([[5.,.7]])
  result=quartet_search(p,r,D,.5,np.random.default_rng(1))
  self.assertTrue(result['E_found']);w=result['witness'];rr=r+w['receiver_delta'];dd=D.copy();dd[result['defender_index']]+=w['defender_delta']
  labels=[channel_label(p,a,b,.5,1.)['label'] for a,b in [(r,D),(rr,D),(r,dd),(rr,dd)]]
  self.assertEqual(labels,w['labels_0_A_B_AB']);self.assertEqual(labels[:3],[labels[0]]*3);self.assertNotEqual(labels[0],labels[3])
  np.testing.assert_array_equal(labels_batch(p,np.array([r,rr,r,rr]),np.array([D,D,dd,dd])),labels)
 def test_keep_and_flip_does_not_imply_E(self):
  p=[0,0];r=[10,0];d=np.array([[5.,.7]])
  # One variable can keep and flip; pairing two blockers is not an endpoint interaction.
  keep=channel_label(p,r,d,.5,1.)['label'];flipped=channel_label(p,r,d+[[0,-.3]],.5,1.)['label']
  self.assertNotEqual(keep,flipped)
  self.assertFalse(keep==keep==flipped) # explicit former predicate insufficient
 def test_nonplayers_not_defenders(self):
  rows=[]
  for i in range(11):rows.append(dict(person_id='d'+str(i),entity_type='player',team_id='b',x=i,y=2.))
  rows.extend([dict(person_id='p',entity_type='player',team_id='a',x=0,y=0),dict(person_id='r',entity_type='player',team_id='a',x=10,y=0),dict(person_id='ball',entity_type='ball',team_id='ball',x=5,y=0)])
  p,r,D,ids,n=audit_roles(pd.DataFrame(rows),'p','r','a');self.assertEqual(len(D),11);self.assertEqual(n,1)
  with self.assertRaises(ValueError):audit_roles(pd.DataFrame(rows),'p','d0','a')
 def test_unseen_regression_denominator_and_joint_transitions(self):
  labels=np.array([[1,1,0],[1,1,0]]);before=np.array([[1,1,1],[0,1,0]]);after=np.array([[1,0,0],[1,1,0]])
  r=state_metrics(before,after,labels,np.array([0,1]));self.assertEqual(r['old_correct_atomic_n'],3);self.assertEqual(r['old_correct_atomic_regressed'],1)
  self.assertEqual(np.array(r['transition_rows_old_cols_new_000_to_111']).sum(),2)
 # ---- R09 all-scene deployment (no oracle-feasibility filter) ----
 def test_grouped_outcomes_matches_production_and_cheapest_rule(self):
  score=np.array([[.9,.8]]).reshape(-1);cost=np.array([[2.,1.]]).reshape(-1);rows=np.array([[0,1]]);feas=np.array([True,True])
  a,s,c,sel=grouped_outcomes(score,rows,cost,feas,.5)
  self.assertEqual(int(sel[0]),1);self.assertEqual(float(c[0]),1.0)
  a2,s2,c2=policy_outcomes(score,np.zeros(2,int),cost,feas,np.array([0]),.5)
  self.assertTrue(np.array_equal(a,a2) and np.array_equal(s,s2) and np.allclose(c,c2))
  dearer=np.where(np.array([True,True]),-cost,np.inf).argmin()
  self.assertEqual(int(dearer),0) # legacy mutant would buy the dearer candidate
  tie=np.array([[1.,1.]]).reshape(-1)
  self.assertEqual(int(grouped_outcomes(score,rows,tie,feas,.5)[3][0]),0) # equal cost -> original row order
  self.assertEqual(int(grouped_outcomes(np.array([.4,.4]),np.array([[0,1]]),cost,feas,.5)[3][0]),-1) # below threshold -> reject
 def test_deployment_denominator_keeps_oracle_infeasible_scenes(self):
  score=np.array([[.9,.1],[.1,.1]]).reshape(-1);cost=np.array([[2.,1.],[2.,1.]]).reshape(-1)
  rows=np.array([[0,1],[2,3]]);feas=np.array([True,False,False,False])
  acted,succ,c,sel=grouped_outcomes(score,rows,cost,feas,.5)
  m=deployment_metrics(acted,succ,c,np.array([True,False]))
  self.assertEqual(m['n_scenes'],2);self.assertEqual(m['coverage'],0.5);self.assertEqual(m['own_quality'],1.0)
  self.assertEqual(m['net_benefit_per_scene'],-0.5)
  self.assertEqual(m['oracle_infeasible_scenes']['n_scenes'],1)
  self.assertEqual(m['oracle_infeasible_scenes']['n_acted'],0)
  self.assertEqual(m['oracle_infeasible_scenes']['n_success'],0)
  self.assertEqual(m['oracle_feasible_scenes']['coverage'],1.0)
  self.assertNotEqual(m['coverage'],m['oracle_feasible_scenes']['coverage']) # feasible-only denominator mutant
 def test_scene_groups_and_rows_for_are_label_free(self):
  so=np.array([5,5,5,7,7,7]);scenes,rows=scene_row_matrix(so)
  np.testing.assert_array_equal(scenes,[5,7]);np.testing.assert_array_equal(rows,[[0,1,2],[3,4,5]])
  np.testing.assert_array_equal(rows_for(np.array([5]),scenes,rows),[[0,1,2]])
  with self.assertRaises(ValueError):scene_row_matrix(np.array([5,5,7]))
  with self.assertRaises(ValueError):rows_for(np.array([7,5]),scenes,rows)
 def test_budget_threshold_dev_only_tie_takes_largest(self):
  act=np.array([[True,False],[True,True],[True,True]]);ts=np.array([1.,2.,3.])
  k,tau,cov=budget_threshold(act,ts,1.0);self.assertEqual((k,tau,cov),(2,3.0,1.0))
  self.assertEqual(budget_threshold(act,ts,0.5)[0],0)
 def test_paired_bootstrap_matches_archived_own_quality_ci(self):
  a1=np.array([True,True,False]);s1=np.array([True,False,False]);c1=np.array([1.,0.,0.])
  a2=np.array([True,False,False]);s2=np.array([False,False,False]);c2=np.array([1.,0.,0.])
  mine=paired_deployment_ci(a1,s1,c1,a2,s2,c2,np.random.default_rng(7))['own_quality']
  arch=paired_own_quality_ci(s1,a1,s2,a2,np.arange(3),np.random.default_rng(7))
  self.assertEqual(mine,arch)
  ci=deployment_bootstrap_ci(a1,s1,c1,np.random.default_rng(3))
  self.assertLessEqual(ci['coverage'][0],1.0);self.assertGreaterEqual(ci['net_benefit_per_scene'][1],-1.0)
 def test_oracle_curve_dedup_and_envelope(self):
  score=np.array([[.9,.1],[.9,.1]]).reshape(-1);cost=np.array([[2.,1.],[2.,1.]]).reshape(-1)
  rows=np.array([[0,1],[2,3]]);feas=np.array([True,True,False,False])
  ts,curve,n_raw=oracle_label_curve(score,rows,cost,feas,np.array([True,False]))
  self.assertEqual(len(curve),len(ts));self.assertLessEqual(len(ts),n_raw)
  env=pareto_envelope(curve);self.assertTrue(all(r in curve for r in env));self.assertTrue(len(env)>=1)
  for r in env: # no envelope point is dominated on (own quality, net benefit, -cost)
   q=lambda z: z[2] if z[2] is not None else 0.0
   self.assertFalse(any((q(o)>=q(r) and o[5]>=r[5] and o[4]<=r[4]) and (q(o)>q(r) or o[5]>r[5] or o[4]<r[4]) for o in curve))
if __name__=='__main__':unittest.main(verbosity=2)
