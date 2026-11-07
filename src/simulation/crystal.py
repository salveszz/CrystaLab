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
    # The four positions locate the rectangular faces in tank coordinates.
    # They are independent, so one side can grow faster without moving the other.
    left: float
    right: float
    bottom: float
    top: float
    # Synthetic event counter only; it does not encode defect position or quality.
    defects: int = 0

    @property
    def width(self):
        # Derive the current span from the faces instead of storing duplicate state.
        return self.right - self.left

    @property
    def height(self):
        # Vertical span changes when either the bottom or top face advances.
        return self.top - self.bottom

    @property
    def area(self):
        # Unit-depth crystal volume equals this 2D area in the starter model.
        return self.width * self.height
