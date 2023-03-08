#!/usr/bin/env python

# Imports
from python.analysis import vector
import sys
sys.path.insert(1, '/users/wx21978/projects/pion-phys/pi0-analysis/analysis/')


def get_impact_parameter(direction, start_pos, beam_vertex):
    """
    Finds the impact parameter between a PFO and beam vertex.

    Parameters
    ----------
    direction : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Direction of PFO.
    start_pos : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Initial point of the PFO.
    beam_vertex : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Beam interaction point.

    Returns
    -------
    m : ak.Array
        Impact parameter between the PFO and beam vertex.
    """
    rel_pos = vector.sub(beam_vertex, start_pos)
    cross = vector.cross(rel_pos, direction)
    return vector.magnitude(cross)


def get_separation(pos1, pos2):
    """
    Finds the separation between positions `pos1` and `pos2`.

    Parameters
    ----------
    spos1 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Postion 1.
    spos1 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Postion 2.

    Returns
    -------
    separation : ak.Array
        Separation between positions 1 and 2.
    """
    return vector.magnitude(vector.sub(pos1, pos2))


def closest_approach(dir1, dir2, start1, start2):
    """
    Finds the closest approach between two showers.

    Parameters
    ----------
    dir1 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Direction of first PFO.
    dir2 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Direction of second PFO.
    start1 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Initial point of first PFO.
    start2 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Initial point of second PFO.

    Returns
    -------
    d : ak.Array
        Magnitude of the sortest direction between the two PFOs.
    """
    # x_1 - x_2 + lambda_1 v_1 - lambda_2 v_2 = d/sin(theta) v_1 x v_2
    cross = vector.normalize(vector.cross(dir1, dir2))
    rel_start = vector.sub(start1, start2)
    # Separation between the lines
    d = vector.dot(rel_start, cross)
    return d


def get_shared_vertex(mom1, mom2, start1, start2):
    """
    Estimates a shared vertex for two vectors and starting positions
    by taking the midpoint of the line of closest approach.

    This is a projected point, not the difference between positions.

    Momenta are used instead of directions to allow for potential
    updates which weight the position of the vertex along the line
    of closest approach based on the relative momenta.

    Parameters
    ----------
    mom1 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Momentum of first PFO.
    mom2 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Momentum of second PFO.
    start1 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Initial point of first PFO.
    start2 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Initial point of second PFO.

    Returns
    -------
    vertex : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Shared vertex between the PFOs.
    """
    # We estimate the shared vertex by taking the midpoint of the line of closest approch
    # This is a projected point, NOT the difference between the reconstructed starts
    joining_dir = vector.normalize(vector.cross(mom1, mom2))
    separation = vector.dot(vector.sub(start1, start2), joining_dir)
    dir1_selector = vector.cross(joining_dir, mom2)
    # We don't use the normalised momentum, because we later multiply by the momentum, so it cancels
    start1_offset = vector.dot(vector.sub(
        start2, start1), dir1_selector) / vector.dot(mom1, dir1_selector)
    pos1 = vector.add(start1, vector.prod(start1_offset, mom1))
    return vector.add(pos1, vector.prod(separation/2, joining_dir))


def get_midpoints(x1, x2):
    """
    Returns the midpoint of the starting positions:
        ``mp = (x1 + x2)/2``

    Parameters
    ----------
    x1 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Position of first point.
    x2 : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Position of second point.

    Returns
    -------
    mp : ak.zip({'x':ak.Array, 'y':ak.Array, 'z':ak.Array})
        Midpoint of `x1` and `x2`.
    """
    return vector.prod(0.5, vector.add(x1, x2))
