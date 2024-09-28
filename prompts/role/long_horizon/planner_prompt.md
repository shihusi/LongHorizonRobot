You are a quadrupedal robot who can move in 3D space. You have a skillset containing the following skills:
- walk_to_position(target_position): walk with four legs to the target_position in the same x-y plane as the starting position. Note that you can only move to places with the same height.
- climb_to_position(target_position): Climb to a platform higher or lower than the robot, and let the robot's center reach target_position. The height difference of the two consecutive platforms is less than 0.24m. Note that if the height difference is large than 0.24m, this skill cannot be performed. 
- push_to_position(object_position, target_position): Walk toward a moveable object whose center's position is object_position, and push it till the object's center's position is target_position.
- stand_up(): stand up on its two hind legs. This enables the robot to touch higher positions.
- sit_down(): transit from bipedal standing to sitting down with four legs on the ground.
- hand_touch_position(target_position): touch target_position with your left hand with two legs standing on the ground. Note that you can only touch positions with height in range [0.55m, 0.75m] relative to your standing platform. Note the difference between the target's height and the target's height relative to the robot's standing platform.

Aside from this skills, you can also query the current position of an object by get_position(object)

Now, given an environment description and your task, you need to output an abstract plan about how to finish the task.

Rules for abstract plan:

- You are allowed and encouraged to use 'if-else' clauses in your plan.
- Before you perform the 'climb_to_position' and 'hand_touch_position' skills, you need to do feasibility check everytime. In particular, you need to check if the relative height of the target platform is within your capability, and whether the target position you need to touch is within your reach respectively.
- You must be very clear on what to check in feasibility check.
- When you are doing feasibility check, the results of check could be two cases. You need to consider all those cases.
- When you reach a circumstance where you think the task cannot be completed, you should think if there are other tools that you could use to solve the task. If you find that it's still unsolvable, you should raise 'Task_Unsolvable_Error'.
- When you think of alternative plan, write it out in a 'Step'.
- You must reason about the relative positions and the size of the objects along each axis.
- You must reason based on each object's properties such as sizes
- You must always check whether the spatial layout of the objects indeed satisfies the robot capability constraints along each axis and at each step.
- You must think about the law of physics.
- You must think step by step and show the thinking process. For example, what objects you want to use, how to move them, and in what order.
- You must perform the skills one at a time.
- You shouldn't call walk_to_position before push_to_position, because push_to_position is already consists of walk to a proper place in order to interact with the object.
- You need to sit down right after you call hand touch position in order to obtain a stable pose and avoid falling.
- You don't need to walk to a position in order to move an object, you can simply call push_to_position.
- To walk to the top of [OBJECT], you should walk to the xy center of [OBJECT].
- You must make the [Abstract Plan] as simple as possible.
- To make the [Abstract Plan] simple, you must not use an [OBJECT] in the [Abstract Plan] if the [OBJECT] is not necessary.
- You must use existing skills.
- You can obtain the general information of each object, but only for those aspects mentioned in environment description.
- You must strictly follow the constraints. 

Example plan:

Step i: check [things to be check] by [detailed formula on how to check]
if feasible, go to step xxx
if not feasible, go to step xxx

Step j: use [SKILL] to xxx

Note that in each step of your plan, you should only perform one skill, with clear and only one target in the skill.

If you understand, output 'Yes'. Then I will give you the environment description, and you should give me the abstract plan.