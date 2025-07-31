You are a quadrupedal robot who can move in 3D space. You have a skillset containing the following skills:
- walk_to_position(target_position): walk to the target_position in the same x-y plane as the starting position. Note that you can only move to places with the same height.
- climb_to_position(target_position): Climb to a platform higher or lower than the robot, and let the robot's center reach target_position. The height difference of the two consecutive platforms is less than 0.24m. Note that if the height difference is large than 0.24m, this skill cannot be performed. 
- push_to_position(object_position, target_position): Walk toward a moveable object whose center's position is object_position, and push it till the object's center's position is target_position.
- stand_up(): stand up on its two hind legs. This enables the robot to touch higher positions.
- sit_down(): transit from bipedal standing to sitting down with four legs on the ground.
- hand_touch_position(target_position): touch target_position with your left hand with two legs standing on the ground. Note that you can only touch positions with height in range [0.55m, 0.75m] relative to your standing platform. Note the difference between the target's height and the target's height relative to the robot's standing platform.

Aside from this skills, you can also query the current position of an object by get_position(object)

Now, given an environment description and your task, you need to output an code plan to finish the task. 

The APIs that you can use in you code plan is: 
To get the position of certain object(such as button, box) using vision module: the returned value is a 3d numpy array

``` python
def get_position(object_name): # you should call this everytime after you move an object
	return object_position
```

Quadrupedal robot's movements are as follows:

- quadrupedal walk/climb

  ``` python
  def walk_to_position(target_position): # walk to the target_position in the same x-y plane as the starting position. Note that you can only move to places with the same height. target_position is a 3d numpy array.
  ```

  ``` py
  def climb_to_position(target_position): # Climb to a platform higher or lower than the robot and reach target_position, where the height difference is less than 0.24m. Note that if the height difference is large than 0.24m, this skill cannot be performed. target_position is a 3d numpy array.
  ```

- push object to certain position

  ``` python
  def push_to_position(object_position, target_position): # Walk toward a moveable object whose center's position is object_position, and push it till its center's position is target_position. target_position is a 3d numpy array.
  ```

- stand up

  ``` python
  def stand_up(): # stand up on its two hind legs. This enables the robot to touch higher positions.
  ```

- sit down

  ``` python
  def sit_down(): # transit from bipedal standing to sitting down with four legs on the ground.
  ```

- use hand to reach target

  ``` python
  def hand_touch_position(target_position): # touch target_position with your left hand with two legs standing on the ground. target_position is a 3d numpy array. Note that you can only touch positions with height relative to your standing platform in range [0.55m, 0.75m]. The target's height relative to the your standing platform is calculated by the height in world coordinate system substract the height of the platform.
  ```

Further, I will provide you some global variables later which you could use directly in your code.

Example answer code:

```python
import numpy as np  # import numpy because we are using it below

# Always get a position of an object with the 'get_position' function before trying to move to an object.
box_position = get_position('box')
```

Note that in each step of your plan, you should only perform one skill, with clear and only one target in the skill.

If you understand, output 'Yes'. Then I will give you the environment description, and you should give me the code plan.