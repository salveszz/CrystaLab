# HOW THIS FILE WORKS
# Stores the crystal as four boundary positions inside the 2D tank.
# The engine moves left, right, bottom, and top independently to represent
# different growth rates on different faces. Width, height, and area are derived
# from those positions rather than stored separately.
# Area times the configured density gives crystal mass, assuming unit depth.
# defects is an optional synthetic event count, not a measured quality score.
# This rectangle is a simple geometry approximation, not a full KDP crystal shape.

"""A rectangular 2D seed with four independently advancing faces."""
from dataclasses import dataclass


@dataclass
class Crystal:
    left: float
    right: float
    bottom: float
    top: float
    defects: int = 0

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.top - self.bottom

    @property
    def area(self):
        return self.width * self.height
