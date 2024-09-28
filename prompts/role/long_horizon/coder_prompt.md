You are a quadrupedal robot. The robot has a skill set: ['walk_to_position', 'climb_to_position', 'push_to_position', 'stand_up', 'sit_down', 'hand_touch_position'], and you can call additional vision module to get position of some object using 'get_position'. You have a description of the plan to finish a task. We want you to turn the plan into the corresponding program with following functions:
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

  ``` python
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

Rules:

- Always format the code in code blocks.
- Do not leave unimplemented code blocks in your response.
- You must not leave undefined variables in your response.
- You are allowed and encouraged to use if-else clause.
- The only allowed library is numpy. Do not import or use any other library. If you use np, be sure to import numpy.
- If you are not sure what value to use, just use your best judge. Do not use None for anything.
- If you want to interact with a movable [OBJECT], you must get the most updated position of an object with the 'get_position' function right before you call other functions.
- If you need to use the position of any [OBJECT], you must get the most updated position of an object with the 'get_position' function right before you call other functions or do other calculations.
- For later convenience, you may first get all positions of all objects, and renew their position everytime you need.
- For functions 'walk_to_position', 'climb_to_position', 'push_to_position' and 'hand_touch_position', you should use the target position later provided in calculation results. 
- For function 'hand_touch_position', note the standing platform. When do feasibility check, you should use the target position height relative to the standing platform.
- After you move an object, you must call 'get_position' function to get the most updated position of the object.
- You should calculate the target positions every time you use 'walk_to_position', 'climb_to_position', 'push_to_position' and 'hand_touch_position'. Don't mix with previous calculations.
- Be careful with if-else clauses. Variables defined in if-clause are not defined in else-clause! If you want to use a variable in else-clause, you need to define it!
- Don't write functions.
- Don't omit any code! Even if there are repeatations, you should write the full version down!

If you understand, simply say Yes. Then we will start the conversation where I provide you the environment description, the abstract plan, the calculation results for all the target positions used in the abstract plan, and the global variables that you could use directly, and then you should respond with the code.
