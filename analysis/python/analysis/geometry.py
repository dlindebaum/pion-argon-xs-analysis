"""
Created on: 24/09/2026 14:23

Author: Shyam Bhuller

Description:  Module that contains detector geometry classes.
"""

class TPCGeometry:
    """Define x, y, and z bounds and provide coordinate and position outside-volume masks.
    """
    x : tuple[float, float] # bounds of the TPC in the x axis.
    y : tuple[float, float] # bounds of the TPC in the y axis.
    z : tuple[float, float] # bounds of the TPC in the z axis.


    def __check_bounds__(self, v : float | list, v_bounds : tuple[float, float]) -> bool | list[bool]:
        """Return whether each value lies strictly outside the supplied inclusive bounds.

            Args:
                v (float | list): Coordinate value or values to test.
                v_bounds (tuple[float, float]): Lower and upper coordinate bounds.

            Returns:
                bool | list[bool]: ``True`` where a value is below the lower or above the upper bound.
        """
        return (min(v_bounds) > v) | (v > max(v_bounds))


    def outside_tpc_x(self, x : float | list) -> bool | list[bool]:
        """ Check is value is outside x bounds of the TPC.

            Args:
                x (float | list): X values.

            Returns:
                bool | list[bool]: True if the value is outside the bounds, False otherwise.
        """
        return self.__check_bounds__(x, self.x)

    
    def outside_tpc_y(self, y : float | list) -> bool | list[bool]:
        """ Check is value is outside y bounds of the TPC.

            Args:
                y (float | list): Y values.

            Returns:
                bool | list[bool]: True if the value is outside the bounds, False otherwise.
        """
        return self.__check_bounds__(y, self.y)

    
    def outside_tpc_z(self, z : float | list) -> bool | list[bool]:
        """ Check is value is outside z bounds of the TPC.

            Args:
                z (float | list): Z values.

            Returns:
                bool | list[bool]: True if the value is outside the bounds, False otherwise.
        """
        return self.__check_bounds__(z, self.z)


    def outside_tpc(self, x : float | list, y : float | list, z : float | list) -> bool | list[bool]:
        """ Check if positions are outside bounds of the TPC.

            Args:
                x (float | list): X values.
                y (float | list): Y values.
                z (float | list): Z values.

            Returns:
                bool | list[bool]: True if the value is outside the bounds, False otherwise.
        """
        return self.outside_tpc_x(x) | self.outside_tpc_y(y) | self.outside_tpc_z(z)


class ProtoDUNESPGeometry(TPCGeometry):
    """TPC geometry with ProtoDUNE-SP bounds: X=[-350, 350], y=[0, 600], and z=[0, 700].
    """
    x = [-350, 350]
    y = [0, 600]
    z = [0, 700]
