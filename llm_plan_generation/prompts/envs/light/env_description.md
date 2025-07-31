The environment description is:

In this setting, you are in a environment with a stair case, a moveable box, and a button. Following are the information of these objects. Note that the shape tuple (a, b, c) correspond to the length of the object along axis x, y, z respectively.

- Stair Case: 2 steps. We can view the staircase as two consecutive boxes with different heights. The shape of the lower box is (lowerstair_size_x, lowerstair_size_y, lowerstair_size_z), and the shape of the higher box is (higherstair_size_x, higherstair_size_y, higherstair_size_z). So the height of each step in the stair case is lowerstair_size_z, higherstair_size_z - lowerstair_size_z respectively. The stair case is unmovable. You can obtain the center of the lower and higher boxes by calling get_position('lower stair') and get_position('higher stair'). In order to climb to the top of the stairs, you need to climb the two steps separately.
- Box: Shape is (box_size_x, box_size_y, box_size_z). Movable only on the ground, and cannot move up the stairs. You can obtain the center of the box by calling get_position('moveable box'). Its only usage is that you can move it in front of the lower stair and form as an additional stair for climbing the lower stair if the box height is within your climbing ability and if the height of lowerstair minus box height is within your climbing ability. It's preferable to construct new steps in the direction of the original stair case. Note that it cannot help climbing the higher stair as it cannot move up the lower stair. Note that you don't need to use it if not necessary.
- Button: Unmovable. You can obtain the position of the button by calling get_position('button'), and it would stay the same. The button is on top of the stairs, and you can only touch it if your standing platform is the higher stair in the stair case. The function of the button is that once pressed, you can close the light of the room.

The coordinate system: z is pointing upward, x is along the center of two boxes of the stair case, pointing from the low step to higher step, y can be determined by right-hand rule.

Initially you are on plain ground. Your goal is to close the light of the room.

Constraints you must follow:

- You can only climb up or climb down a platform with a height no larger than 0.24m in one step.
- When you stand up on two legs, your left hand can reach target with height in range [0.55m, 0.70m].
- You can assume that the robot size is (robot_size[0], robot_size[1], robot_size[2]).
- At the end of the episode, you must sit with four legs on the floor.