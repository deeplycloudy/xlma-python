# Coordinates
---
The coords module handles conversions between coordinate systems. The `CoordinateSystem` class represents a generic transform. Subclasses of `CoordinateSystem` are specific coordinate systems that have implemented transformations. These subclasses are documented on the "Transforms" page, see the sidebar.

There are a few useful tools related to coordinate transforms included as well.

::: pyxlma.coords
    options:
        members:
        - centers_to_edges
        - centers_to_edges_2d
        - semiaxes_to_invflattening 
        - CoordinateSystem