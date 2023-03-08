"""
Created on: 26/01/2022 15:45

Author: Shyam Bhuller

Description: Contains vector operations, based of awkward array records.
"""
import awkward as ak
import numpy as np


def vector(x, y, z) -> ak.Record:
    """ Creates a vector like record.

    Args:
        x (Any): x component
        y (Any): y component
        z (Any): z component

    Returns:
        ak.Record: record structurd like a 3-vector
    """
    return ak.zip({"x" : x, "y" : y, "z" : z})


def magnitude(vec : ak.Record) -> ak.Array:
    """ Magnitude of 3-vector.

    Args:
        vec (ak.Record created by vector): vector

    Returns:
        ak.Array: array of magnitudes
    """
    return (vec.x**2 + vec.y**2 + vec.z**2)**0.5


def normalize(vec : ak.Record) -> ak.Record:
    """ Normalize a vector (get direction).

    Args:
        vec (ak.Record): a vector

    Returns:
        ak.Record: norm of vector
    """
    m = magnitude(vec)
    return vector( vec.x / m, vec.y / m, vec.z / m )


def dot(a : ak.Record, b : ak.Record) -> ak.Array:
    """dot product of 3-vectors.

    Args:
        a (ak.Record): first vector
        b (ak.Record): second vector

    Returns:
        ak.Array: array of dot products
    """
    return (a.x * b.x) + (a.y * b.y) + (a.z * b.z)


def cross(a : ak.Record, b : ak.Record) -> ak.Array:
    """cross product of 3-vectors.

    Args:
        a (ak.Record): first vector
        b (ak.Record): second vector

    Returns:
        ak.Record: array of cross products
    """
    x = a.y*b.z - b.y*a.z
    y = a.z*b.x - b.z*a.x
    z = a.x*b.y - b.x*a.y
    return ak.zip({"x": x, "y": y, "z": z})


def prod(s, v : ak.Record) -> ak.Record:
    """ Product of scalar and vector.

    Args:
        s (ak.Array or single number): scalar
        v (ak.Record): vector

    Returns:
        ak.Record created by vector: s * v
    """
    return vector(s * v.x, s * v.y, s * v.z)


def angle(a : ak.Record, b : ak.Record) -> ak.Array:
    """ Compute angle between two vectors.

    Args:
        a (ak.Record): a vector
        b (ak.Record): another vector

    Returns:
        ak.Array: angle between a and b
    """
    return np.arccos(dot(a, b) / (magnitude(a) * magnitude(b)))


def dist(a : ak.Record, b : ak.Record):
    """ Compute cartesian distance between two vectors.

    Args:
        a (ak.Record): a vector
        b (ak.Record): another vector

    Returns:
        ak.Array: distance between a and b
    """
    return magnitude(ak.zip({"x": a.x - b.x, "y": a.y - b.y, "z": a.z - b.z}))


def add(a : ak.Record, b : ak.Record):
    """ Compute vector addition of two vectors.

    Args:
        a (ak.Record): a vector
        b (ak.Record): another vector

    Returns:
        ak.Array: addition of a and b
    """
    return ak.zip({"x": a.x + b.x, "y": a.y + b.y, "z": a.z + b.z})


def sub(a : ak.Record, b : ak.Record):
    """ Compute vector subtraction (displacement) of two vectors.

    Args:
        a (ak.Record): a vector
        b (ak.Record): another vector

    Returns:
        ak.Array: displacement between a and b
    """
    return ak.zip({"x": a.x - b.x, "y": a.y - b.y, "z": a.z - b.z})