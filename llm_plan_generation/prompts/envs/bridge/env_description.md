The environment description is:

In this setting, you are in a environment with three objects: a moveable box, a low platform, and a high platform. Following are the information of these objects.There is a gap between low platform and high platform, which is too wide for a quadruped robot to cross but is large enough to fit a movable box. These three objects are placed independently on the ground, each with a different height, which means that you need to climb from one to another. The height difference between the movable box and the low platforms, the movable box and the high platforms is within your climbing capability. Note that the shape tuple (a, b, c) correspond to the length of the object along axis x, y, z respectively. You can obtain the center of the three objects by calling get_position()

- Moveable Box: Shape is (box_size_x, box_size_y, box_size_z). The box can be pushed or moved only on the ground, and cannot be moved upon the low or high platforms. You can obtain the center of the box by calling get_position('moveable box'). You can only push the boxes on the ground. The height of the movable box is too high so that you can't climbed onto the box directly from the ground.

- Low Platform: Shape is (low_platform_size_x, low_platform_size_y, low_platform_size_z). The platform is fixed to the ground and cannot be moved. The height of the lower platform is within your climbing capability.  

- High Platform: Shape is (high_platform_size_x, high_platform_box_size_y, high_platform_size_z). A fixed platform is too high to be climbed directly from the ground.


The coordinate system: z is pointing upward, x is along the center of two platforms, pointing from the low platform to higher platform, y can be determined by right-hand rule.

Initially you are on the ground. Your goal is to get to the higher platform.

Constraints you must follow:

- You can only climb up or climb down a platform with a height no larger than 0.24m in one step.
- You can only push objects which are on the same plane with you.


