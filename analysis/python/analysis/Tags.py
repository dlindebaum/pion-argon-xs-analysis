"""
Created on: 11/05/2023 17:07

Author: Shyam Bhuller

Description: Module for creating tags for events.
"""
from dataclasses import dataclass

import awkward as ak
from particle import Particle

from python.analysis.Master import Data
from python.analysis.EventSelection import generate_truth_tags

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


def GenerateTrueParticleTags(events : Data) -> Tags:
    """ Creates true particle tags with boolean masks. Does this for all PFOs.

    Args:
        events (Master.Data): events to look at

    Returns:
        Tags: tags
    """
    particles_to_tag = [
        211, -211, 13, -13, 11, -11, 22, 2212, 321
    ] # anything not in this list is tagged as other

    masks = ParticleMasks(events.trueParticlesBT.pdg, particles_to_tag)
    masks["other"] = OtherMask(masks)

    tags = Tags()
    for i, m in enumerate(masks):
        tags[m] = Tag(m, m, "C" + str(i), masks[m], i)

    return tags


def GenerateTrueBeamParticleTags(events : Data) -> Tags:
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

    tags = Tags()
    for i, m in enumerate(masks):
        tags[m] = Tag(m, m, "C" + str(i), masks[m], i)

    return tags


def GeneratePi0Tags(events : Data) -> Tags:
    """ Truth tags for pi0s.
        Categories are:
            two photons from the same pi0
            two photons from two different pi0s
            one photon from a pi0
            no photons from a pi0

    Args:
        events (Data): events to look at

    Returns:
        Tags: tags
    """
    #! This needs to be modified so that it doesn't rely on events only containing photon candidates.
    pi0_photon = events.trueParticlesBT.motherPdg == 111

    two_photons = ak.all(pi0_photon, 1)
    same_mother = (events.trueParticlesBT.mother[:, 0] == events.trueParticlesBT.mother[:, 1])

    pi0_tags = Tags()
    pi0_tags["2 $\gamma$'s, same $\pi^{0}$"]      = Tag("2 $\gamma$'s, same $\pi^{0}$" , "pi0s"            , mask = two_photons & same_mother  , number = 0) # both PFOs are photons from the same pi0
    pi0_tags["2 $\gamma$'s, different $\pi^{0}$"] = Tag("2 $\gamma$s, different $\pi0$", "different mother", mask = two_photons & ~same_mother , number = 1) # both PFOs are pi0 photons, but not from the same pi0
    pi0_tags["1 $\gamma$"]                        = Tag("one photon"                   , "one photon"      , mask = ak.sum(pi0_photon, -1) == 1, number = 2) # one PFO is a pi0 photon
    pi0_tags["no $\gamma$"]                       = Tag("no photons"                   , "no photons"      , mask = ak.sum(pi0_photon, -1) == 0, number = 3) # no PFO is a pi0 photon
    return pi0_tags


def GenerateTrueFinalStateTags(events : Data = None) -> Tags:
    """ Generate truth tags for final state of the beam interaction.

    Args:
        events (Data, optional): events to look at. Defaults to None.

    Returns:
        Tags: tags
    """
    tags = Tags()
    tags["$1\pi^{0} + 0\pi^{+}$"     ]          = Tag("$1\pi^{0} + 0\pi^{+}$"              , "exclusive signal", "#8EBA42", generate_truth_tags(events, 1, 0)    if events is not None else None, 0)
    tags["$0\pi^{0} + 0\pi^{+}$"     ]          = Tag("$0\pi^{0} + 0\pi^{+}$"              , "background",       "#777777", generate_truth_tags(events, 0, 0) if events is not None else None, 1)
    tags["$1\pi^{0} + \geq 1\pi^{+}$"]          = Tag("$1\pi^{0} + \geq 1\pi^{+}$"         , "sideband",         "#E24A33", generate_truth_tags(events, 1, (1,)) if events is not None else None, 2)
    tags["$0\pi^{0} + \geq 1\pi^{+}$"]          = Tag("$0\pi^{0} + \geq 1\pi^{+}$"         , "sideband",         "#988ED5", generate_truth_tags(events, 0, (1,)) if events is not None else None, 3)
    tags["$\greater 1\pi^{0} + \geq 0\pi^{+}$"] = Tag("$> 1\pi^{0} + \geq 0\pi^{+}$"       , "sideband",         "#348ABD", generate_truth_tags(events, (2,), (0,)) if events is not None else None, 4)
    return tags
