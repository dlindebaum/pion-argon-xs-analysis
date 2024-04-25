"""
Created on: 11/05/2023 17:07

Author: Shyam Bhuller

Description: Module for creating tags for events.
"""
from dataclasses import dataclass

import awkward as ak
from particle import Particle

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from python.analysis.Master import Data

@dataclass(slots = True)
class Tag:
    """ Tag data class, stores the information a tag can have.
    """
    name : str = ""
    name_simple : str = ""
    colour : str = None
    mask : ak.Array = ""
    number : int = -1


class Tags(dict):
    """ A collection of tags, it is a fancy dictionary with the capability of accessing tag information using the TagIterator.
        i.e. one can do tags.mask.values to get the mask for every tag in the collection,
        or you can iterate over the items in the tag rather than the dictionary key i.e.
        for t in tags.colour:
            print(t) # this will print the colour of the tag rather than the key.
    """
    def __init__(self, *arg, **kw):
        for member in list(Tag.__annotations__.keys()):
            setattr(self, member, TagIterator(self, member)) # assign a TagIterator per item in Tag

        self.name = TagIterator(self, "name")
        super(Tags, self).__init__(*arg, **kw) 


class TagIterator:
    """ Iterator which allows enumeration of a collection of tags based on the value of the tag.
        See how this is used in Tags. 
    """
    def __init__(self, parent : Tags, value : str):
        self.__parent = parent # which collection this belongs to
        self.__value = value # which value does this iterate over

    def __getitem__(self, i):
        for _, v in self.__parent.items():
            if getattr(v, self.__value) == i:
                return v

    @property
    def values(self) -> list:
        """ returns a list of values for all tags.

        Returns:
            list: list of values for each tag
        """
        return [getattr(v, self.__value) for _, v in self.__parent.items()]

    def __str__(self) -> str:
        return str(self.values)


def ParticleMasks(pdgs : ak.Array, to_tag : list) -> dict:
    """ produces a dictionry of masks based on particle pdg codes to tag specified by the user.

    Args:
        pdgs (ak.Array): array of pdg codes
        to_tag (list): particle pgd codes to tag

    Returns:
        dict: particle tags
    """
    masks = {}
    for t in to_tag:
        masks["$" + Particle.from_pdgid(t).latex_name + "$"] = pdgs == t
    return masks


def OtherMask(masks : dict) -> ak.Array:
    """ Creates a mask which selects indices not already tagged by a set of masks.

    Args:
        masks (dict): masks which tag data

    Returns:
        ak.Array: mask which tags any untagged data.
    """
    other = None
    for m in masks:
        if other is None:
            other = masks[m]
        else:
            other = other | masks[m]
    return ~other


def GenerateTrueParticleTags(events):# : Data) -> Tags:
    """ Creates true particle tags with boolean masks. Does this for all PFOs.

    Args:
        events (Master.Data): events to look at

    Returns:
        Tags: tags
    """
    particles_to_tag = [
        211, -211, 13, -13, 11, -11, 22, 2212, 321
    ] # anything not in this list is tagged as other

    if ak.count(events.trueParticlesBT.pdg) == 0: # the ntuple has no MC, so provide some null data base off recoParticles array shape
        pdg = ak.where(events.recoParticles.number, -1, 0)
    else:
        pdg = events.trueParticlesBT.pdg
    masks = ParticleMasks(pdg, particles_to_tag)
    masks["other"] = OtherMask(masks)

    tags = Tags()
    for i, m in enumerate(masks):
        tags[m] = Tag(m, m, "C" + str(i), masks[m], i)

    return tags


def GenerateTrueBeamParticleTags(events):# : Data) -> Tags:
    """ Creates true particle tags with boolean masks for beam particles.

    Args:
        events (Master.Data): events to look at

    Returns:
        Tags: tags
    """
    particles_to_tag = [
        211, -13, -11, 2212, 321
    ] # anything not in this list is tagged as other

    masks = ParticleMasks(events.trueParticlesBT.beam_pdg, particles_to_tag)
    masks["other"] = OtherMask(masks)

    inel = events.trueParticlesBT.beam_endProcess == "pi+Inelastic"
    cosmic = events.trueParticlesBT.beam_origin == 2

    new_mask = {"$\\pi^{+}$:inel" : masks["$\\pi^{+}$"] & inel, "$\\pi^{+}$:decay" : masks["$\\pi^{+}$"] & ~inel, }

    masks.pop("$\\pi^{+}$")

    new_mask.update(masks)
    masks = new_mask

    tags = Tags()
    for i, m in enumerate(masks):
        tags[m] = Tag(m, m, "C" + str(i), masks[m] & ~cosmic, i)

    tags["cosmics"] = Tag("cosmics", "cosmics", "C" + str(i + 1), cosmic, i+1)

    return tags


def GenerateTrueParticleTagsPiPlus(events):# : Data) -> Tags:
    """ Creates true particle tags with boolean masks, with specific tags related to pi+ particles. Does this for all PFOs.

    Args:
        events (Master.Data): events to look at

    Returns:
        Tags: tags
    """
    particles_to_tag = [
        211, -211, 13, -13, -11, 22, 2212
    ] # anything not in this list is tagged as other
    # particles_to_tag = [
    #     211, -211, 13, -13, 11, -11, 22, 2212, 321
    # ]

    if ak.count(events.trueParticlesBT.pdg) == 0: # the ntuple has no MC, so provide some null data base off recoParticles array shape
        pdg = ak.where(events.recoParticles.number, -1, 0)
        beam_daughter = pdg

    else:
        pdg = events.trueParticlesBT.pdg
        beam_daughter = events.trueParticlesBT.mother == 1

    masks = ParticleMasks(pdg, particles_to_tag)
    masks["other"] = OtherMask(masks)

    masks["$\\pi^{\pm}$"] = masks["$\\pi^{+}$"] | masks["$\\pi^{-}$"]
    for p in ["$\\pi^{+}$", "$\\pi^{-}$"]:
        masks.pop(p) 

    masks["$\mu^{\pm}$"] = masks["$\mu^{+}$"] | masks["$\mu^{-}$"]
    for p in ["$\mu^{+}$", "$\mu^{-}$"]:
        masks.pop(p) 

    for p in ["$\\pi^{\pm}$"]:
        new_mask = {p : masks[p] & beam_daughter, f"{p}:2nd" : masks[p] & (~beam_daughter)}
        masks.pop(p)
        new_mask.update(masks)
        masks = new_mask

    tags = Tags()
    for i, m in enumerate(masks):
        tags[m] = Tag(m, m, "C" + str(i), masks[m], i)

    return tags


def GenerateTrueParticleTagsInterestingPFOs(events) -> Tags:
    particles_to_tag = [
        211, -211, 13, -13, 11, -11, 22, 2212
    ]
    if ak.count(events.trueParticlesBT.pdg) == 0: # the ntuple has no MC, so provide some null data base off recoParticles array shape
        pdg = ak.where(events.recoParticles.number, -1, 0)
        beam_pi0 = pdg
        beam_daughter = pdg
        other_pi0 = pdg

    else:
        pdg = events.trueParticlesBT.pdg
        beam_pi0 = events.trueParticlesBT.is_beam_pi0
        other_pi0 = (events.trueParticlesBT.motherPdg == 111) & (~events.trueParticlesBT.is_beam_pi0)
        beam_daughter = events.trueParticlesBT.mother == 1

    masks = ParticleMasks(pdg, particles_to_tag)
    masks["other"] = OtherMask(masks)

    for p in ["$\\gamma$"]:
        new_mask = {p+":2nd" : masks[p] & ~beam_pi0 & ~other_pi0, f"{p}:beam $\pi^{0}$" : masks[p] & beam_pi0, f"{p}:other $\pi^{0}$" : masks[p] & other_pi0}
        masks.pop(p)
        new_mask.update(masks)
        masks = new_mask

    masks["$\\pi^{\pm}$"] = masks["$\\pi^{+}$"] | masks["$\\pi^{-}$"]
    for p in ["$\\pi^{+}$", "$\\pi^{-}$"]:
        masks.pop(p) 

    masks["$\mu^{\pm}$"] = masks["$\mu^{+}$"] | masks["$\mu^{-}$"]
    for p in ["$\mu^{+}$", "$\mu^{-}$"]:
        masks.pop(p) 

    for p in ["$\\pi^{\pm}$"]:
        new_mask = {p : masks[p] & beam_daughter, f"{p}:2nd" : masks[p] & (~beam_daughter)}
        masks.pop(p)
        new_mask.update(masks)
        masks = new_mask

    tags = Tags()
    for i, m in enumerate(masks):
        tags[m] = Tag(m, m, "C" + str(i), masks[m], i)

    return tags

def GenerateTrueParticleTagsPi0Shower(events) -> Tags:# : Data):
    """ Creates true particle tags with boolean masks with specific tags for pi0 photon showers. Does this for all PFOs.

    Args:
        events (Master.Data): events to look at

    Returns:
        Tags: tags
    """
    particles_to_tag = [
        211, -211, 13, -13, 11, -11, 22, 2212
    ]
    if ak.count(events.trueParticlesBT.pdg) == 0: # the ntuple has no MC, so provide some null data base off recoParticles array shape
        pdg = ak.where(events.recoParticles.number, -1, 0)
        beam_pi0 = pdg
        other_pi0 = pdg
    else:
        pdg = events.trueParticlesBT.pdg
        beam_pi0 = events.trueParticlesBT.is_beam_pi0
        other_pi0 = (events.trueParticlesBT.motherPdg == 111) & (~events.trueParticlesBT.is_beam_pi0)
    masks = ParticleMasks(pdg, particles_to_tag)
    masks["other"] = OtherMask(masks)

    for p in ["$\\gamma$"]:
        new_mask = {p+":2nd" : masks[p] & ~beam_pi0 & ~other_pi0, f"{p}:beam $\pi^{0}$" : masks[p] & beam_pi0, f"{p}:other $\pi^{0}$" : masks[p] & other_pi0}
        masks.pop(p)
        new_mask.update(masks)
        masks = new_mask

    tags = Tags()
    for i, m in enumerate(masks):
        tags[m] = Tag(m, m, "C" + str(i), masks[m], i)

    return tags


def GeneratePi0Tags(events, photon_PFOs : ak.Array) -> Tags:# : Data, photon_PFOs : ak.Array) -> Tags:
    """ Truth tags for pi0s.
        Categories are:
            two photons from the same pi0
            two photons from two different pi0s
            one photon from a pi0
            no photons from a pi0

    Args:
        events (Data): events to look at.
        photon_PFOs (ak.Array): mask which select PFOs that are photon showers

    Returns:
        Tags: tags
    """
    pi0_photon = events.trueParticlesBT.motherPdg == 111

    correctly_matched_photons = ak.sum(pi0_photon & photon_PFOs, -1)
    photon_mothers = events.trueParticlesBT.mother[pi0_photon & photon_PFOs]
    photon_mothers = ak.pad_none(photon_mothers, 2, -1)
    # same_mother = photon_mothers[correctly_matched_photons == 2][:, 0] == photon_mothers[correctly_matched_photons == 2][:, 1]
    same_mother = ak.where(correctly_matched_photons == 2, photon_mothers[:, 0] == photon_mothers[:, 1], False)

    pi0_tags = Tags()
    pi0_tags["2 $\gamma$'s, same $\pi^{0}$"]      = Tag("2 $\gamma$'s, same $\pi^{0}$" , "pi0s"               , mask = same_mother, number = 0) # both PFOs are photons from the same pi0
    pi0_tags["2 $\gamma$'s, different $\pi^{0}$"] = Tag("2 $\gamma$s, different $\pi^{0}$", "different mother", mask = (~same_mother) & (correctly_matched_photons == 2), number = 1) # both PFOs are pi0 photons, but not from the same pi0
    pi0_tags["1 $\gamma$"]                        = Tag("1 $\gamma$"                   , "one photon"         , mask = correctly_matched_photons == 1, number = 2) # one PFO is a pi0 photon
    pi0_tags["0 $\gamma$"]                       = Tag("0 $\gamma$"                   , "no photons"          , mask = correctly_matched_photons == 0, number = 3) # no PFO is a pi0 photon
    return pi0_tags


def ExclusiveProcessTags(true_masks):
    tags = Tags()
    colours = {
        "charge_exchange" : "#8EBA42",
        "absorption"      : "#777777",
        "single_pion_production" : "#E24A33",
        "pion_production" : "#988ED5",
    }
    name_simple = {
        "charge_exchange" : "cex",
        "absorption" :"abs",
        "single_pion_production" : "spip",
        "pion_production" : "pip"
    }

    if true_masks is None:
        true_masks = {k : None for k in name_simple}
    for i, t in enumerate(true_masks):
        tags[t] = Tag(t, name_simple[t], colours[t], true_masks[t], i)
    return tags


def StoppingMuonTag(events : "Data"):
    masks = ParticleMasks(events.trueParticlesBT.beam_pdg, [-13])
    masks["other"] = OtherMask(masks)

    decay = events.trueParticlesBT.beam_endProcess == "Decay"

    new_mask = {"$\\mu^{+}$:inel" : masks["$\\mu^{+}$"] & ~decay, "$\\mu^{+}$:decay" : masks["$\\mu^{+}$"] & decay}

    masks.pop("$\\mu^{+}$")

    new_mask.update(masks)
    masks = new_mask

    tags = Tags()
    for i, m in enumerate(masks):
        tags[m] = Tag(m, m, "C" + str(i), masks[m], i)
    return tags


def BeamScraperTag(scraper_ids : ak.Array):
    tags = Tags()
    tags["scraper"] = Tag("scraper", "scraper", "C6", scraper_ids, 0)
    tags["non scraper"] = Tag("non scraper", "non scraper", "C0", ~scraper_ids, 1)
    return tags