{
  "version": "5.6",
  "source": "zenodo.org/records/17171112",
  "h4m10": {
    "beta_max": 0.3,
    "sharpness": 10.0,
    "comm_scale": 0.1,
    "formula": "beta=0.30*(1-near_tie)*(1-0.5*cw)",
    "l_comm_current": 0.0,
    "comm_weight": 0.0
  },
  "ablation": {
    "0.002": {
      "v0_fb": 198,
      "v0_fg": 0,
      "h4_fb": 0,
      "h4_fg": 0,
      "rho_v0": 0.9532814238042269,
      "rho_h4": 0.9995550611790878
    },
    "0.005": {
      "v0_fb": 192,
      "v0_fg": 0,
      "h4_fb": 0,
      "h4_fg": 0,
      "rho_v0": 0.9532814238042269,
      "rho_h4": 0.9995550611790878
    },
    "0.01": {
      "v0_fb": 185,
      "v0_fg": 0,
      "h4_fb": 0,
      "h4_fg": 0,
      "rho_v0": 0.9532814238042269,
      "rho_h4": 0.9995550611790878
    },
    "0.02": {
      "v0_fb": 161,
      "v0_fg": 0,
      "h4_fb": 0,
      "h4_fg": 0,
      "rho_v0": 0.9532814238042269,
      "rho_h4": 0.9995550611790878
    },
    "0.05": {
      "v0_fb": 118,
      "v0_fg": 0,
      "h4_fb": 0,
      "h4_fg": 0,
      "rho_v0": 0.9532814238042269,
      "rho_h4": 0.9995550611790878
    },
    "0.1": {
      "v0_fb": 52,
      "v0_fg": 0,
      "h4_fb": 0,
      "h4_fg": 0,
      "rho_v0": 0.9532814238042269,
      "rho_h4": 0.9995550611790878
    }
  },
  "all_gaps_flp_bad_zero": true,
  "rho_gain": 0.0463,
  "benchmark": {
    "n1": {
      "rnd": 208,
      "bm25": 1,
      "hmc_pi": 1,
      "psi_v0": 1,
      "psi_h4": 1
    },
    "n2": {
      "rnd": 81,
      "bm25": 1,
      "hmc_pi": 1,
      "psi_v0": 1,
      "psi_h4": 1
    },
    "n3": {
      "rnd": 2470,
      "bm25": 1,
      "hmc_pi": 1,
      "psi_v0": 1,
      "psi_h4": 1
    },
    "n4": {
      "rnd": 1970,
      "bm25": 1,
      "hmc_pi": 1,
      "psi_v0": 2,
      "psi_h4": 1
    }
  },
  "mrr": {
    "rnd": 0.0045,
    "bm25": 1.0,
    "hmc_pi": 1.0,
    "psi_v0": 0.875,
    "psi_h4": 1.0
  },
  "timing_ms": {
    "bm25": 2.41,
    "hmc_pi": 73.72,
    "psi_v0": 5.7,
    "psi_h4": 6.64
  },
  "open_problems": [
    "causal_On_logn",
    "n4_sbert_real",
    "m10_needs_nonzero_angles_for_cw_effect"
  ]
}
