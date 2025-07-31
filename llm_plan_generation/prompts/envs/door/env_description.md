The environment description is:

In this setting, you are in a environment with a door and a moveable package. Following are the information of these objects. Note that the shape tuple (a, b, c) correspond to the length of the object along axis x, y, z respectively.

- Package: Shape is (package_size_x, package_size_y, package_size_z). It stays on the ground initially, and it's movable only on the ground. You can obtain the center of the package by calling get_position('package'). 
- Door: It is initially closed, and its shape is (door_size_x, door_size_y, door_size_z). You can obtain the center of the door when it's closed by calling get_position('door'). The door is currently closed and cannot be pushed open by you.
- Doorbell: It is always on the wall, and it's next to the door. You can obtain the center of the doorbell by calling get_position('doorbell'). By pressing it, you could notify human to open the door. Note that you can only ring it when you are in front of it.

The coordinate system: z is pointing upward, x is perpendicular to the door when it's closed, pointing from outside the door to inside the door, and y can be determined by right-hand rule. Note that x is also perpendicular to the wall on which hangs the door bell.

Initially you are on plain ground. Your goal is to move the package inside the door.

Constraints you must follow:

- You can only climb up or climb down a platform with a height no larger than 0.24m in one step.
- When you stand up on two legs, your left hand can reach target with height in range [0.55m, 0.70m].
- You can assume that the robot size is (robot_size[0], robot_size[1], robot_size[2]).