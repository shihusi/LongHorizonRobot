The environment description is:

In this setting, you are in a environment with a platform, call buttons and a elevator. Following are the information of these objects. Note that the shape tuple (a, b, c) correspond to the length of the object along axis x, y, z respectively.

- Platform: There is a step in front of the elevator that you need to climb. You can obtain the center of the platform by calling get_position('platform'). All elevators and buttons are on the platform, but their position information is provided relative to the ground. The platform is large so the buttons are far away from the center of the platform

- Call Buttons: It includes two buttons, button_up and button_down. They are always on the wall, and next to the elevator. You can obtain the center of the button by calling get_position('button_up') or get_position('button_down'). You can press the button_up or button_down button to call the elevator to go up or down.

- Elevator: The inside of the elevator is a cubic space with floor buttons on the walls. The floor buttons are hang on the wall at the edge of the elevator's interior space, where is far away from the center of elevator. You can obtain the center of the elevator ground by calling get_position('elevator').  You can reach your desired floor by pressing the corresponding floor button. You can obtain the center of the floor button of each floor by calling get_position(f'button_{target_floor_idx}'). 

The coordinate system: z is pointing upward, x is perpendicular to the elevator and the wall, pointing from outside the elevator to inside the elevator, and y can be determined by right-hand rule. The wall with the buttons and the interior wall of the elevator are both perpendicular to the x-axis and parallel to the y-axis. All the position arrguments are related to ground root origins.

Initially you are on the ground. You will receive a variable current_floor indicating the floor you are currently on and a variable target_floor indicating your target floor. You need to reach the target floor.

Constraints you must follow:

- You can only climb up or climb down a platform with a height no larger than 0.24m in one step.
- When you stand up on two legs, your left hand can reach target with height in range [0.55m, 0.70m].
- Before you press a button you need to be right in front of the button and walk close enough.
- The size of the elevator refer to the size of its interior space, which you can use to calculate the position of the elevator's interior walls.