"""Beam scraper utilities."""

def IsScraper(mc : Data, beam_scraper_args : dict) -> ak.Array:
    beam_inst_KE = KE(mc.recoParticles.beam_inst_P, Particle.from_pdgid(211).mass) # get kinetic energy from beam instrumentation
    true_ffKE = mc.trueParticles.beam_KE_front_face

    delta_KE_upstream = beam_inst_KE - true_ffKE

    scraper_ids = {}
    for k, v in beam_scraper_args.items():
        scraper_ids[k] = (beam_inst_KE > min(v["bins"])) & (beam_inst_KE < max(v["bins"]))
        threshold = v["mu_e_res"] + 3 * abs(v["sigma_e_res"])
        
        scraper_ids[k] = scraper_ids[k] & (delta_KE_upstream > threshold)
    scraper_ids = SelectionTools.CombineMasks(scraper_ids, operator = "or")
    return scraper_ids
