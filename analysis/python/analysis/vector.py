"""
Created on: 26/01/2022 15:45

Author: Shyam Bhuller

Description: Contains vector operations, based of awkward array records.
"""
import awkward as ak
import numpy as np


null_vector = ak.Array([{"x": -999, "y": -999, "z": -999}])

def vector(x, y, z) -> ak.Record:
    """ Creates a vector like record.

        Args:
            x (Any): X component.
            y (Any): Y component.
            z (Any): Z component.

        Returns:
            ak.Record: Record structurd like a 3-vector.
    """
    return ak.zip({"x" : x, "y" : y, "z" : z})


def magnitude(vec : ak.Record) -> ak.Array:
    """ Magnitude of 3-vector.

        Args:
            vec (ak.Record): Vector.

        Returns:
            ak.Array: Array of magnitudes.
    """
    return (vec.x**2 + vec.y**2 + vec.z**2)**0.5


def normalize(vec : ak.Record) -> ak.Record:
    """ Normalize a vector (get direction).

        Args:
            vec (ak.Record): A vector.

        Returns:
            ak.Record: Norm of vector.
    """
    m = magnitude(vec)
    return vector( vec.x / m, vec.y / m, vec.z / m )


def dot(a : ak.Record, b : ak.Record) -> ak.Array:
    """ Dot product of 3-vectors.

        Args:
            a (ak.Record): First vector.
            b (ak.Record): Second vector.

        Returns:
            ak.Array: Array of dot products.
    """
    return (a.x * b.x) + (a.y * b.y) + (a.z * b.z)


def cross(a : ak.Record, b : ak.Record) -> ak.Array:
    """ Cross product of 3-vectors.

        Args:
            a (ak.Record): First vector.
            b (ak.Record): Second vector.

        Returns:
            ak.Array: Array of cross products.
    """
    x = a.y*b.z - b.y*a.z
    y = a.z*b.x - b.z*a.x
    z = a.x*b.y - b.x*a.y
    return ak.zip({"x": x, "y": y, "z": z})


def prod(s, v : ak.Record) -> ak.Record:
    """ Product of scalar and vector.

        Args:
            s (ak.Array or single number): Scalar.
            v (ak.Record): Vector.

        Returns:
            ak.Record: S * v.
    """
    return vector(s * v.x, s * v.y, s * v.z)


def angle(a : ak.Record, b : ak.Record) -> ak.Array:
    """ Compute angle between two vectors.

        Args:
            a (ak.Record): A vector.
            b (ak.Record): Another vector.

        Returns:
            ak.Array: Angle between a and b.
    """
    return np.arccos(dot(a, b) / (magnitude(a) * magnitude(b)))


def add(a : ak.Record, b : ak.Record):
    """ Compute vector addition of two vectors.

        Args:
            a (ak.Record): A vector.
            b (ak.Record): Another vector.

        Returns:
            ak.Array: Addition of a and b.
    """
    return ak.zip({"x": a.x + b.x, "y": a.y + b.y, "z": a.z + b.z})


def sub(a : ak.Record, b : ak.Record):
    """ Compute vector subtraction (displacement) of two vectors.

        Args:
            a (ak.Record): A vector.
            b (ak.Record): Another vector.

        Returns:
            ak.Array: Displacement between a and b.
    """
    return ak.zip({"x": a.x - b.x, "y": a.y - b.y, "z": a.z - b.z})


def dist(a : ak.Record, b : ak.Record):
    """ Compute cartesian distance between two vectors.

        Args:
            a (ak.Record): A vector.
            b (ak.Record): Another vector.

        Returns:
            ak.Array: Distance between a and b.
    """
    return magnitude(sub(a, b))


def reciprocal(v : ak.Record) -> ak.Record:
    """ Compute the reciprocal of each component for a vector.

        Args:
            v (ak.Record): A vector.

        Returns:
            ak.Record: Vector of reciprocal values.
    """
    return ak.zip({"x" : 1/v.x, "y" : 1/v.y, "z" : 1/v.z})


def multiply(a : ak.Record, b : ak.Record) -> ak.Record:
    """Multiply corresponding x, y, and z components elementwise.

        Args:
            a (ak.Record): First vector with x, y, and z components.
            b (ak.Record): Second vector with corresponding components.

        Returns:
            ak.Record: Vector whose components are the pairwise products.
    """
    return ak.zip({"x" : a.x * b.x, "y" : a.y * b.y, "z" : a.z * b.z})


def power(a : ak.Record, e : float) -> ak.Record:
    """ Raise each component by an exponent.

        Args:
            a (ak.Record): A vector.
            e (float): Exponent.

        Returns:
            ak.Record: Vector of values raised to a power.
    """
    return ak.zip({"x" : a.x ** e, "y" : a.y ** e, "z" : a.z ** e})