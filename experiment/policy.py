"""Assumed masses mapped to activation distance; alpha is always one for added cost."""
import math
ROBOT_MASS_KG=5.0
REFERENCE_HAND_MASS_KG=0.6
REFERENCE_ETA_M=0.025
HAND_MASSES={'ETA06':0.6,'ETA23':2.3}
def reduced_mass(hand_mass):return ROBOT_MASS_KG*hand_mass/(ROBOT_MASS_KG+hand_mass)
def mass_policy(condition):
 hand=HAND_MASSES.get(condition)
 if hand is None and condition!='R':raise ValueError(condition)
 mu=reduced_mass(hand) if hand is not None else None
 eta=REFERENCE_ETA_M*math.sqrt(mu/reduced_mass(REFERENCE_HAND_MASS_KG)) if mu is not None else REFERENCE_ETA_M
 return {'condition':condition,'assumed_robot_effective_mass_kg':ROBOT_MASS_KG,'assumed_hand_effective_mass_kg':hand,'reduced_mass_kg':mu,'reference_reduced_mass_kg':reduced_mass(REFERENCE_HAND_MASS_KG),'eta_calculated_m':eta,'alpha':1.0 if condition!='R' else None,'mu_formula':'m_r*m_h/(m_r+m_h)','eta_formula':'0.025*sqrt(mu(m_h)/mu(0.6))','mass_values_are_assumptions':True,'interpretation':'force-ratio-inspired heuristic activation policy; not physically derived safety distance or actual force prediction','robot_mass_source':'fixed assumption; not calculated or validated for Franka'}
