"""
Created on: 24/09/2026 14:23

Author: Shyam Bhuller

Description:  module that contains detector geometry classes.
"""

class TPCGeometry:
    x : tuple[float, float] # bounds of the TPC in the x axis.
    y : tuple[float, float] # bounds of the TPC in the y axis.
    z : tuple[float, float] # bounds of the TPC in the z axis.


    def __check_bounds__(self, v : float | list, v_bounds : tuple[float, float]) -> bool | list[bool]:
        """ Checks if a value is within the bounds.

        Args:
            v (float | list): Value/s.
            v_bounds (tuple[float, float]): Bounds for the value.

        Returns:
            bool | list[bool]: True is the value is outside the bounds, False otherwise.
        """
        return (min(v_bounds) > v) | (v > max(v_bounds))


    def outside_tpc_x(self, x : float | list) -> bool | list[bool]:
        """ Check is value is outside x bounds of the TPC.

        Args:
            x (float | list): x values.

        Returns:
            bool | list[bool]: True if the value is outside the bounds, False otherwise.
        """
        return self.__check_bounds__(x, self.x)

    
    def outside_tpc_y(self, y : float | list) -> bool | list[bool]:
        """ Check is value is outside y bounds of the TPC.

        Args:
            y (float | list): y values.

        Returns:
            bool | list[bool]: True if the value is outside the bounds, False otherwise.
        """
        return self.__check_bounds__(y, self.y)

    
    def outside_tpc_z(self, z : float | list) -> bool | list[bool]:
        """ Check is value is outside z bounds of the TPC.

        Args:
            z (float | list): z values.

        Returns:
            bool | list[bool]: True if the value is outside the bounds, False otherwise.
        """
        return self.__check_bounds__(z, self.z)


    def outside_tpc(self, x : float | list, y : float | list, z : float | list) -> bool | list[bool]:
        """ Check if positions are outside bounds of the TPC.

        Args:
            x (float | list): x values.
            y (float | list): y values.
            z (float | list): z values.

        Returns:
            bool | list[bool]: True if the value is outside the bounds, False otherwise.
        """
        return self.outside_tpc_x(x) | self.outside_tpc_y(y) | self.outside_tpc_z(z)


class ProtoDUNESPGeometry(TPCGeometry):
    x = [-350, 350]
    y = [0, 600]
    z = [0, 700]
