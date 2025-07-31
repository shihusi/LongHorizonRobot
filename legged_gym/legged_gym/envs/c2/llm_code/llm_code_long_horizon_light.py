import torch
import numpy as np



def generate_llm_plan_main1(self, env_id):
    self.cur_step[env_id][0] = -1
    self.cur_num = 0
    self.box_target_place = None
    self.have_pushed = 0

    def get_position(object):  # TODO: use value from enviroment, change into numpy
        if object == "lower stair":
            return self.object_root_states[env_id, 1, :3].cpu().numpy()
            # return np.array([1.0, 1.0, 0.2])
        elif object == "higher stair":
            return self.object_root_states[env_id, 2, :3].cpu().numpy()
            # return np.array([1.8, 1.0, 0.3])
        elif object == "button":
            return self.button_root_states[env_id, :3].cpu().numpy()
            # return np.array([2.2, 1.2, 1.2])
        elif object == "moveable box":
            if self.box_target_place == None:
                return self.object_pos[env_id].cpu().numpy()
            return self.box_target_place
            # return np.array([0, 0, 0.1])
        return np.array([0, 0, 0])

    def walk_to_position(position):
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1

    def climb_to_position(position):
        if not self.have_pushed:
            self.llm_plan[env_id][self.cur_num][0] = 5
            ready_pos = torch.from_numpy(position).to(self.device)
            ready_pos[0] -= 0.8
            ready_pos[2] = 0.0
            self.llm_plan[env_id][self.cur_num][1:] = ready_pos
            self.cur_num += 1
            print("walk", object, "to", ready_pos)
        self.llm_plan[env_id][self.cur_num][0] = 1
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        print("climb to ", position)
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1

    def stand_up():
        # self.llm_plan.append([2, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 2
        self.llm_plan[env_id][self.cur_num][1:] = torch.tensor(
            [0, 0, 0], device=self.device
        )
        self.cur_num += 1
        print("stand up")

    def hand_touch_position(position):
        # self.llm_plan.append([3, torch.from_numpy(position).to(self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 3
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        print("hand touch", position)

    def sit_down():
        # self.llm_plan.append([6, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 6
        self.llm_plan[env_id][self.cur_num][1:] = torch.tensor(
            [0, 0, 0], device=self.device
        )
        self.cur_num += 1
        print("sit down")

    def push_to_position(object, position):
        # self.llm_plan.append([4, torch.from_numpy(position).to(self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 4
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        self.have_pushed = 1
        # self.llm_plan.append([5, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = (
            torch.tensor([0.1, -0.1, 0], device=self.device)
            + self.env_origins[env_id]
        )
        self.cur_num += 1
        self.box_target_place = position

    # Global variables
    robot_size = np.array([0.54, 0.40, 0.40])  # Example sizes, adjust as necessary
    lowerstair_size_x, lowerstair_size_y, lowerstair_size_z = (
        0.8,
        0.8,
        self.object_size[env_id, 1].item(),
    )
    higherstair_size_x, higherstair_size_y, higherstair_size_z = (
        0.8,
        0.8,
        self.object_size[env_id, 2].item(),
    )
    box_size_x, box_size_y, box_size_z = (
        0.8,
        0.8,
        self.object_size[env_id, 0].item(),
    )

    # Step 1: Check the height of the lower stair
    if lowerstair_size_z <= 0.24:
        # Step 2: Climb the lower stair
        lowerstair_position = get_position("lower stair")
        target_position_lower_stair = np.array(
            [
                lowerstair_position[0],
                lowerstair_position[1],
                lowerstair_size_z + robot_size[2] / 2,
            ]
        )
        climb_to_position(target_position_lower_stair)

        # Step 4: Check the height of the higher stair relative to the lower stair
        if (higherstair_size_z - lowerstair_size_z) <= 0.24:
            # Step 6: Climb the higher stair
            higherstair_position = get_position("higher stair")
            target_position_higher_stair = np.array(
                [
                    higherstair_position[0],
                    higherstair_position[1],
                    higherstair_size_z + robot_size[2] / 2,
                ]
            )
            climb_to_position(target_position_higher_stair)

            # Step 7: Check if the button is within hand reach
            button_position = get_position("button")
            if 0.55 <= (button_position[2] - higherstair_size_z) <= 0.70:
                # Step 8: Stand up to reach the button
                stand_up()

                # Step 9: Press the button
                hand_touch_position(button_position)

                # Step 10: Sit down
                sit_down()
            else:
                raise Exception("Task_Unsolvable_Error")
        else:
            raise Exception("Task_Unsolvable_Error")
    else:
        # Step 3: Check if the box can be used as an additional step
        if box_size_z <= 0.24 and (lowerstair_size_z - box_size_z) <= 0.24:
            # Step 5: Move the box to form an additional step
            lowerstair_position = get_position("lower stair")
            box_position = get_position("moveable box")
            target_position_box = np.array(
                [
                    lowerstair_position[0]
                    - (lowerstair_size_x / 2 + box_size_x / 2),
                    lowerstair_position[1],
                    box_position[2],
                ]
            )
            push_to_position(box_position, target_position_box)

            # After moving the box, attempt to climb the lower stair again
            lowerstair_position = get_position(
                "lower stair"
            )  # Update position after moving the box
            target_position_lower_stair = np.array(
                [
                    lowerstair_position[0],
                    lowerstair_position[1],
                    lowerstair_size_z + robot_size[2] / 2,
                ]
            )
            climb_to_position(target_position_lower_stair)

            # Repeat steps for climbing higher stair and pressing the button as above
            higherstair_position = get_position("higher stair")
            target_position_higher_stair = np.array(
                [
                    higherstair_position[0],
                    higherstair_position[1],
                    higherstair_size_z + robot_size[2] / 2,
                ]
            )
            climb_to_position(target_position_higher_stair)

            button_position = get_position("button")
            if 0.55 <= (button_position[2] - higherstair_size_z) <= 0.70:
                stand_up()
                hand_touch_position(button_position)
                sit_down()
            else:
                raise Exception("Task_Unsolvable_Error")
        else:
            raise Exception("Task_Unsolvable_Error")



def generate_llm_plan_no_calculator1(self, env_id):
    self.cur_step[env_id][0] = -1
    self.cur_num = 0
    self.box_target_place = None
    self.have_pushed = 0

    def get_position(object):  # TODO: use value from enviroment, change into numpy
        if object == "lower stair":
            return self.object_root_states[env_id, 1, :3].cpu().numpy()
            # return np.array([1.0, 1.0, 0.2])
        elif object == "higher stair":
            return self.object_root_states[env_id, 2, :3].cpu().numpy()
            # return np.array([1.8, 1.0, 0.3])
        elif object == "button":
            return self.button_root_states[env_id, :3].cpu().numpy()
            # return np.array([2.2, 1.2, 1.2])
        elif object == "moveable box":
            if self.box_target_place == None:
                return self.object_pos[env_id].cpu().numpy()
            return self.box_target_place
            # return np.array([0, 0, 0.1])
        return np.array([0, 0, 0])

    def walk_to_position(position):
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1

    def climb_to_position(position):
        if not self.have_pushed:
            self.llm_plan[env_id][self.cur_num][0] = 5
            ready_pos = torch.from_numpy(position).to(self.device)
            ready_pos[0] -= 0.8
            self.llm_plan[env_id][self.cur_num][1:] = ready_pos
            self.cur_num += 1
            print("walk", object, "to", position)
        self.llm_plan[env_id][self.cur_num][0] = 1
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        print("climb to ", position)
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1

    def stand_up():
        # self.llm_plan.append([2, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 2
        self.llm_plan[env_id][self.cur_num][1:] = torch.tensor(
            [0, 0, 0], device=self.device
        )
        self.cur_num += 1
        print("stand up")

    def hand_touch_position(position):
        # self.llm_plan.append([3, torch.from_numpy(position).to(self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 3
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        print("hand touch", position)

    def sit_down():
        # self.llm_plan.append([6, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 6
        self.llm_plan[env_id][self.cur_num][1:] = torch.tensor(
            [0, 0, 0], device=self.device
        )
        self.cur_num += 1
        print("sit down")

    def push_to_position(object, position):
        # self.llm_plan.append([4, torch.from_numpy(position).to(self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 4
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        self.have_pushed = 1
        # self.llm_plan.append([5, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = (
            torch.tensor([0.1, -0.1, 0], device=self.device)
            + self.env_origins[env_id]
        )
        self.cur_num += 1
        self.box_target_place = position

    # Global variables
    robot_size = np.array(
        [0.54, 0.40, 0.40]
    )  # Example values, replace with actual values
    lowerstair_size = np.array(
        [0.8, 0.8, self.object_size[env_id, 1].item()]
    )  # Example values, replace with actual values
    higherstair_size = np.array(
        [0.8, 0.8, self.object_size[env_id, 2].item()]
    )  # Example values, replace with actual values
    box_size = np.array(
        [0.8, 0.8, self.object_size[env_id, 0].item()]
    )  # Example values, replace with actual values

    # Step 1: Assess Climbing Feasibility for Lower Stair
    if lowerstair_size[2] <= 0.24:
        # Step 2: Climb Lower Stair
        lower_stair_position = get_position("lower stair")
        climb_to_position(lower_stair_position)
    else:
        # Check if the box can be used as an additional step
        if lowerstair_size[2] - box_size[2] <= 0.24:
            # Step 2b: Move the box in front of the lower stair
            box_position = get_position("moveable box")
            target_position_in_front_of_lower_stair = get_position(
                "lower stair"
            ) - np.array([0, 0, box_size[2]])
            push_to_position(box_position, target_position_in_front_of_lower_stair)
            box_position = get_position(
                "moveable box"
            )  # Update box position after moving it

            # Step 2c: Climb onto the box then the lower stair
            climb_to_position(box_position)
            lower_stair_position = get_position("lower stair")
            climb_to_position(lower_stair_position)
        else:
            raise Exception("Task_Unsolvable_Error")

    # Step 3: Assess Climbing Feasibility for Higher Stair
    if higherstair_size[2] - lowerstair_size[2] <= 0.24:
        # Step 4: Climb Higher Stair
        higher_stair_position = get_position("higher stair")
        climb_to_position(higher_stair_position)
    else:
        raise Exception("Task_Unsolvable_Error")

    # Step 5: Assess Button Pressing Feasibility
    button_position = get_position("button")
    if 0.55 <= (button_position[2] - higherstair_size[2]) <= 0.70:
        # Step 6: Press the Button
        stand_up()
        hand_touch_position(button_position)
        sit_down()
    else:
        raise Exception("Task_Unsolvable_Error")

def generate_llm_plan_no_calculator2(self, env_id):
    self.cur_step[env_id][0] = -1
    self.cur_num = 0
    self.box_target_place = None
    self.have_pushed = 0

    def get_position(object):  # TODO: use value from enviroment, change into numpy
        if object == "lower stair":
            return self.object_root_states[env_id, 1, :3].cpu().numpy()
            # return np.array([1.0, 1.0, 0.2])
        elif object == "higher stair":
            return self.object_root_states[env_id, 2, :3].cpu().numpy()
            # return np.array([1.8, 1.0, 0.3])
        elif object == "button":
            return self.button_root_states[env_id, :3].cpu().numpy()
            # return np.array([2.2, 1.2, 1.2])
        elif object == "moveable box":
            if self.box_target_place == None:
                return self.object_pos[env_id].cpu().numpy()
            return self.box_target_place
            # return np.array([0, 0, 0.1])
        return np.array([0, 0, 0])

    def walk_to_position(position):
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1

    def climb_to_position(position):
        if not self.have_pushed:
            self.llm_plan[env_id][self.cur_num][0] = 5
            ready_pos = torch.from_numpy(position).to(self.device)
            ready_pos[0] -= 0.8
            self.llm_plan[env_id][self.cur_num][1:] = ready_pos
            self.cur_num += 1
            print("walk", object, "to", position)
        self.llm_plan[env_id][self.cur_num][0] = 1
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        print("climb to ", position)
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1

    def stand_up():
        # self.llm_plan.append([2, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 2
        self.llm_plan[env_id][self.cur_num][1:] = torch.tensor(
            [0, 0, 0], device=self.device
        )
        self.cur_num += 1
        print("stand up")

    def hand_touch_position(position):
        # self.llm_plan.append([3, torch.from_numpy(position).to(self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 3
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        print("hand touch", position)

    def sit_down():
        # self.llm_plan.append([6, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 6
        self.llm_plan[env_id][self.cur_num][1:] = torch.tensor(
            [0, 0, 0], device=self.device
        )
        self.cur_num += 1
        print("sit down")

    def push_to_position(object, position):
        # self.llm_plan.append([4, torch.from_numpy(position).to(self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 4
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        self.have_pushed = 1
        # self.llm_plan.append([5, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = (
            torch.tensor([0.1, -0.1, 0], device=self.device)
            + self.env_origins[env_id]
        )
        self.cur_num += 1
        self.box_target_place = position

    # Global variables (sizes)
    robot_size = np.array([0.54, 0.40, 0.40])  # Example sizes, adjust as needed
    lowerstair_size_z = self.object_size[
        env_id, 1
    ].item()  # Example height, adjust as needed
    higherstair_size_z = self.object_size[
        env_id, 2
    ].item()  # Example height, adjust as needed
    box_size_z = self.object_size[
        env_id, 0
    ].item()  # Example height, adjust as needed

    # Step 1: Assess Climbing Feasibility for Lower Stair
    if lowerstair_size_z <= 0.24:
        # Step 2: Climb Lower Stair
        lower_stair_position = get_position("lower stair")
        climb_to_position(lower_stair_position)
        # Proceed to Step 4
    elif (
        lowerstair_size_z > 0.24
        and box_size_z <= 0.24
        and (lowerstair_size_z - box_size_z) <= 0.24
    ):
        # Step 3: Use Box to Assist Climbing Lower Stair
        moveable_box_position = get_position("moveable box")
        lower_stair_position = get_position("lower stair")
        position_in_front_of_lower_stair = lower_stair_position - np.array(
            [0, 0, box_size_z]
        )  # Adjust calculation as necessary
        push_to_position(moveable_box_position, position_in_front_of_lower_stair)
        position_on_top_of_box = get_position("moveable box") + np.array(
            [0, 0, box_size_z]
        )
        climb_to_position(position_on_top_of_box)
        lower_stair_position = get_position("lower stair")
        climb_to_position(lower_stair_position)
        # Proceed to Step 4
    else:
        raise Exception("Task_Unsolvable_Error")

    # Step 4: Assess Climbing Feasibility for Higher Stair
    if (higherstair_size_z - lowerstair_size_z) <= 0.24:
        # Step 5: Climb Higher Stair
        higher_stair_position = get_position("higher stair")
        climb_to_position(higher_stair_position)
        # Proceed to Step 6
    else:
        raise Exception("Task_Unsolvable_Error")

    # Step 6: Press the Button
    button_position = get_position("button")
    button_z = button_position[2]
    if 0.55 <= (button_z - higherstair_size_z) <= 0.70:
        # Step 7: Press Button and Sit Down
        stand_up()
        hand_touch_position(button_position)
        sit_down()
    else:
        raise Exception("Task_Unsolvable_Error")

def generate_llm_plan_no_calculator3(self, env_id):
    self.cur_step[env_id][0] = -1
    self.cur_num = 0
    self.box_target_place = None
    self.have_pushed = 0

    def get_position(object):  # TODO: use value from enviroment, change into numpy
        if object == "lower stair":
            return self.object_root_states[env_id, 1, :3].cpu().numpy()
            # return np.array([1.0, 1.0, 0.2])
        elif object == "higher stair":
            return self.object_root_states[env_id, 2, :3].cpu().numpy()
            # return np.array([1.8, 1.0, 0.3])
        elif object == "button":
            return self.button_root_states[env_id, :3].cpu().numpy()
            # return np.array([2.2, 1.2, 1.2])
        elif object == "moveable box":
            if self.box_target_place == None:
                return self.object_pos[env_id].cpu().numpy()
            return self.box_target_place
            # return np.array([0, 0, 0.1])
        return np.array([0, 0, 0])

    def walk_to_position(position):
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1

    def climb_to_position(position):
        if not self.have_pushed:
            self.llm_plan[env_id][self.cur_num][0] = 5
            ready_pos = torch.from_numpy(position).to(self.device)
            ready_pos[0] -= 0.8
            self.llm_plan[env_id][self.cur_num][1:] = ready_pos
            self.cur_num += 1
            print("walk", object, "to", position)
        self.llm_plan[env_id][self.cur_num][0] = 1
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        print("climb to ", position)
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1

    def stand_up():
        # self.llm_plan.append([2, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 2
        self.llm_plan[env_id][self.cur_num][1:] = torch.tensor(
            [0, 0, 0], device=self.device
        )
        self.cur_num += 1
        print("stand up")

    def hand_touch_position(position):
        # self.llm_plan.append([3, torch.from_numpy(position).to(self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 3
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        print("hand touch", position)

    def sit_down():
        # self.llm_plan.append([6, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 6
        self.llm_plan[env_id][self.cur_num][1:] = torch.tensor(
            [0, 0, 0], device=self.device
        )
        self.cur_num += 1
        print("sit down")

    def push_to_position(object, position):
        # self.llm_plan.append([4, torch.from_numpy(position).to(self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 4
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        self.have_pushed = 1
        # self.llm_plan.append([5, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = (
            torch.tensor([0.1, -0.1, 0], device=self.device)
            + self.env_origins[env_id]
        )
        self.cur_num += 1
        self.box_target_place = position

    # Global variables
    robot_size = np.array(
        [0.54, 0.40, 0.40]
    )  # Example values, replace with actual values
    lowerstair_size = np.array(
        [0.8, 0.8, self.object_size[env_id, 1].item()]
    )  # Example values, replace with actual values
    higherstair_size = np.array(
        [0.8, 0.8, self.object_size[env_id, 2].item()]
    )  # Example values, replace with actual values
    box_size = np.array(
        [0.8, 0.8, self.object_size[env_id, 0].item()]
    )  # Example values, replace with actual values

    # Step 1: Check the feasibility of climbing the lower stair
    if lowerstair_size[2] <= 0.24:
        # Step 2: Climb the lower stair
        climb_to_position(get_position("lower stair"))
        # Step 4: Check the feasibility of climbing the higher stair
        if higherstair_size[2] - lowerstair_size[2] <= 0.24:
            # Step 6: Climb the higher stair
            climb_to_position(get_position("higher stair"))
            # Step 7: Check the feasibility of touching the button
            button_position = get_position("button")
            if 0.55 <= button_position[2] - higherstair_size[2] <= 0.70:
                # Step 8: Press the button
                stand_up()
                hand_touch_position(button_position)
                # Step 9: Sit down
                sit_down()
            else:
                raise Exception("Task_Unsolvable_Error")
        else:
            raise Exception("Task_Unsolvable_Error")
    else:
        # Step 3: Check the feasibility of using the box as an additional step
        if box_size[2] <= 0.24 and lowerstair_size[2] - box_size[2] <= 0.24:
            # Step 5: Move the box and use it as an additional step
            box_position = get_position("moveable box")
            position_in_front_of_lower_stair = get_position(
                "lower stair"
            ) - np.array([0, 0, box_size[2]])
            push_to_position(box_position, position_in_front_of_lower_stair)
            # After moving the box, attempt to climb the lower stair again
            climb_to_position(get_position("lower stair"))
            # Then, check the feasibility of climbing the higher stair again
            if higherstair_size[2] - lowerstair_size[2] <= 0.24:
                # Climb the higher stair
                climb_to_position(get_position("higher stair"))
                # Check the feasibility of touching the button
                button_position = get_position("button")
                if 0.55 <= button_position[2] - higherstair_size[2] <= 0.70:
                    # Press the button
                    stand_up()
                    hand_touch_position(button_position)
                    # Sit down
                    sit_down()
                else:
                    raise Exception("Task_Unsolvable_Error")
            else:
                raise Exception("Task_Unsolvable_Error")
        else:
            raise Exception("Task_Unsolvable_Error")

def generate_llm_plan_coder_only1(self, env_id):
    self.cur_step[env_id][0] = -1
    self.cur_num = 0
    self.box_target_place = None
    self.have_pushed = 0

    def get_position(object):  # TODO: use value from enviroment, change into numpy
        if object == "lower stair":
            return self.object_root_states[env_id, 1, :3].cpu().numpy()
            # return np.array([1.0, 1.0, 0.2])
        elif object == "higher stair":
            return self.object_root_states[env_id, 2, :3].cpu().numpy()
            # return np.array([1.8, 1.0, 0.3])
        elif object == "button":
            return self.button_root_states[env_id, :3].cpu().numpy()
            # return np.array([2.2, 1.2, 1.2])
        elif object == "moveable box":
            if self.box_target_place == None:
                return self.object_pos[env_id].cpu().numpy()
            return self.box_target_place
            # return np.array([0, 0, 0.1])
        return np.array([0, 0, 0])

    def walk_to_position(position):
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1

    def climb_to_position(position):
        if not self.have_pushed:
            self.llm_plan[env_id][self.cur_num][0] = 5
            ready_pos = torch.from_numpy(position).to(self.device)
            ready_pos[0] -= 0.8
            self.llm_plan[env_id][self.cur_num][1:] = ready_pos
            self.cur_num += 1
            print("walk", object, "to", position)
        self.llm_plan[env_id][self.cur_num][0] = 1
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        print("climb to ", position)
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1

    def stand_up():
        # self.llm_plan.append([2, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 2
        self.llm_plan[env_id][self.cur_num][1:] = torch.tensor(
            [0, 0, 0], device=self.device
        )
        self.cur_num += 1
        print("stand up")

    def hand_touch_position(position):
        # self.llm_plan.append([3, torch.from_numpy(position).to(self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 3
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        print("hand touch", position)

    def sit_down():
        # self.llm_plan.append([6, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 6
        self.llm_plan[env_id][self.cur_num][1:] = torch.tensor(
            [0, 0, 0], device=self.device
        )
        self.cur_num += 1
        print("sit down")

    def push_to_position(object, position):
        # self.llm_plan.append([4, torch.from_numpy(position).to(self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 4
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        self.have_pushed = 1
        # self.llm_plan.append([5, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = (
            torch.tensor([0.1, -0.1, 0], device=self.device)
            + self.env_origins[env_id]
        )
        self.cur_num += 1
        self.box_target_place = position

    # Global variables
    robot_size = np.array(
        [0.54, 0.40, 0.40]
    )  # Example values, adjust as per actual
    lowerstair_size = np.array(
        [0.8, 0.8, self.object_size[env_id, 1].item()]
    )  # Example values, adjust as per actual
    higherstair_size = np.array(
        [0.8, 0.8, self.object_size[env_id, 2].item()]
    )  # Example values, adjust as per actual
    box_size = np.array(
        [0.8, 0.8, self.object_size[env_id, 0].item()]
    )  # Example values, adjust as per actual

    # Step 1: Assess if the robot can climb the lower stair without the box
    if lowerstair_size[2] <= 0.24:
        # Step 2: Climb the lower stair
        lower_stair_position = get_position("lower stair")
        climb_to_position(lower_stair_position)
    else:
        # Step 3: If the lower stair is too high, check if the box can be used as an additional step
        if box_size[2] <= 0.24 and (lowerstair_size[2] - box_size[2]) <= 0.24:
            # Step 4: Move the box to the front of the lower stair
            box_position = get_position("moveable box")
            lower_stair_position = get_position("lower stair")
            target_position_for_box = lower_stair_position - np.array(
                [0, 0, box_size[2]]
            )
            push_to_position(box_position, target_position_for_box)
            # Step 5: Climb onto the box
            climb_to_position(target_position_for_box)
            # Step 6: Climb onto the lower stair from the box
            climb_to_position(lower_stair_position)
        else:
            raise Exception("Task_Unsolvable_Error")

    # Step 7: Assess if the robot can climb the higher stair from the lower stair
    if (higherstair_size[2] - lowerstair_size[2]) <= 0.24:
        # Step 8: Climb the higher stair
        higher_stair_position = get_position("higher stair")
        climb_to_position(higher_stair_position)
    else:
        raise Exception("Task_Unsolvable_Error")

    # Step 9: Stand up to reach the button
    stand_up()

    # Step 10: Check if the button is within reach
    button_position = get_position("button")
    if 0.55 <= (button_position[2] - higherstair_size[2]) <= 0.70:
        # Step 11: Press the button
        hand_touch_position(button_position)
    else:
        raise Exception("Task_Unsolvable_Error")

    # Step 12: Sit down to complete the task
    sit_down()

def generate_llm_plan_coder_only2(self, env_id):
    self.cur_step[env_id][0] = -1
    self.cur_num = 0
    self.box_target_place = None
    self.have_pushed = 0

    def get_position(object):  # TODO: use value from enviroment, change into numpy
        if object == "lower stair":
            return self.object_root_states[env_id, 1, :3].cpu().numpy()
            # return np.array([1.0, 1.0, 0.2])
        elif object == "higher stair":
            return self.object_root_states[env_id, 2, :3].cpu().numpy()
            # return np.array([1.8, 1.0, 0.3])
        elif object == "button":
            return self.button_root_states[env_id, :3].cpu().numpy()
            # return np.array([2.2, 1.2, 1.2])
        elif object == "moveable box":
            if self.box_target_place == None:
                return self.object_pos[env_id].cpu().numpy()
            return self.box_target_place
            # return np.array([0, 0, 0.1])
        return np.array([0, 0, 0])

    def walk_to_position(position):
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1

    def climb_to_position(position):
        if not self.have_pushed:
            self.llm_plan[env_id][self.cur_num][0] = 5
            ready_pos = torch.from_numpy(position).to(self.device)
            ready_pos[0] -= 0.8
            self.llm_plan[env_id][self.cur_num][1:] = ready_pos
            self.cur_num += 1
            print("walk", object, "to", position)
        self.llm_plan[env_id][self.cur_num][0] = 1
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        print("climb to ", position)
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1

    def stand_up():
        # self.llm_plan.append([2, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 2
        self.llm_plan[env_id][self.cur_num][1:] = torch.tensor(
            [0, 0, 0], device=self.device
        )
        self.cur_num += 1
        print("stand up")

    def hand_touch_position(position):
        # self.llm_plan.append([3, torch.from_numpy(position).to(self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 3
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        print("hand touch", position)

    def sit_down():
        # self.llm_plan.append([6, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 6
        self.llm_plan[env_id][self.cur_num][1:] = torch.tensor(
            [0, 0, 0], device=self.device
        )
        self.cur_num += 1
        print("sit down")

    def push_to_position(object, position):
        # self.llm_plan.append([4, torch.from_numpy(position).to(self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 4
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        self.have_pushed = 1
        # self.llm_plan.append([5, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = (
            torch.tensor([0.1, -0.1, 0], device=self.device)
            + self.env_origins[env_id]
        )
        self.cur_num += 1
        self.box_target_place = position

    # Define global variables
    robot_size = np.array(
        [0.54, 0.40, 0.40]
    )  # Example values, replace with actual values
    lowerstair_size = np.array(
        [0.8, 0.8, self.object_size[env_id, 1].item()]
    )  # Example values, replace with actual values
    higherstair_size = np.array(
        [0.8, 0.8, self.object_size[env_id, 2].item()]
    )  # Example values, replace with actual values
    box_size = np.array(
        [0.8, 0.8, self.object_size[env_id, 0].item()]
    )  # Example values, replace with actual values

    # Start of the task
    try:
        # Step 1: Assess the height of the stairs and decide if the box is needed to climb the lower stair
        lower_stair_position = get_position("lower stair")
        higher_stair_position = get_position("higher stair")
        box_position = get_position("moveable box")
        button_position = get_position("button")

        # Check if the robot can climb the lower stair without the box
        if lowerstair_size[2] <= 0.24:
            # Climb the lower stair
            walk_to_position(lower_stair_position)
            climb_to_position(
                lower_stair_position + np.array([0, 0, lowerstair_size[2]])
            )
        elif box_size[2] <= 0.24 and (lowerstair_size[2] - box_size[2]) <= 0.24:
            # Move the box to the lower stair and use it as a step
            push_to_position(
                box_position,
                lower_stair_position - np.array([0, 0, box_size[2] / 2]),
            )
            climb_to_position(lower_stair_position - np.array([0, 0, box_size[2]]))
            climb_to_position(lower_stair_position)
        else:
            raise Exception("Task_Unsolvable_Error")

        # Step 2: Climb the higher stair
        if (higherstair_size[2] - lowerstair_size[2]) <= 0.24:
            climb_to_position(higher_stair_position)
        else:
            raise Exception("Task_Unsolvable_Error")

        # Step 3: Press the button
        # Check if the button is within reach
        if 0.55 <= (button_position[2] - higher_stair_position[2]) <= 0.70:
            stand_up()
            hand_touch_position(button_position)
            sit_down()
        else:
            raise Exception("Task_Unsolvable_Error")

    except Exception as e:
        print(e)


def generate_llm_plan_coder_only3(self, env_id):
    self.cur_step[env_id][0] = -1
    self.cur_num = 0
    self.box_target_place = None
    self.have_pushed = 0

    def get_position(object):  # TODO: use value from enviroment, change into numpy
        if object == "lower stair":
            return self.object_root_states[env_id, 1, :3].cpu().numpy()
            # return np.array([1.0, 1.0, 0.2])
        elif object == "higher stair":
            return self.object_root_states[env_id, 2, :3].cpu().numpy()
            # return np.array([1.8, 1.0, 0.3])
        elif object == "button":
            return self.button_root_states[env_id, :3].cpu().numpy()
            # return np.array([2.2, 1.2, 1.2])
        elif object == "moveable box":
            if self.box_target_place == None:
                return self.object_pos[env_id].cpu().numpy()
            return self.box_target_place
            # return np.array([0, 0, 0.1])
        return np.array([0, 0, 0])

    def walk_to_position(position):
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1

    def climb_to_position(position):
        if not self.have_pushed:
            self.llm_plan[env_id][self.cur_num][0] = 5
            ready_pos = torch.from_numpy(position).to(self.device)
            ready_pos[0] -= 0.8
            self.llm_plan[env_id][self.cur_num][1:] = ready_pos
            self.cur_num += 1
            print("walk", object, "to", position)
        self.llm_plan[env_id][self.cur_num][0] = 1
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        print("climb to ", position)
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1

    def stand_up():
        # self.llm_plan.append([2, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 2
        self.llm_plan[env_id][self.cur_num][1:] = torch.tensor(
            [0, 0, 0], device=self.device
        )
        self.cur_num += 1
        print("stand up")

    def hand_touch_position(position):
        # self.llm_plan.append([3, torch.from_numpy(position).to(self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 3
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        print("hand touch", position)

    def sit_down():
        # self.llm_plan.append([6, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 6
        self.llm_plan[env_id][self.cur_num][1:] = torch.tensor(
            [0, 0, 0], device=self.device
        )
        self.cur_num += 1
        print("sit down")

    def push_to_position(object, position):
        # self.llm_plan.append([4, torch.from_numpy(position).to(self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 4
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        self.have_pushed = 1
        # self.llm_plan.append([5, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = (
            torch.tensor([0.1, -0.1, 0], device=self.device)
            + self.env_origins[env_id]
        )
        self.cur_num += 1
        self.box_target_place = position

    # Define global variables for the sizes of objects and robot
    robot_size = np.array(
        [0.54, 0.40, 0.40]
    )  # Example values, adjust as per actual
    lowerstair_size = np.array(
        [0.8, 0.8, self.object_size[env_id, 1].item()]
    )  # Example values, adjust as per actual
    higherstair_size = np.array(
        [0.8, 0.8, self.object_size[env_id, 2].item()]
    )  # Example values, adjust as per actual
    box_size = np.array(
        [0.8, 0.8, self.object_size[env_id, 0].item()]
    )  # Example values, adjust as per actual

    # Function to check if climbing is possible
    def can_climb(current_height, target_height):
        return abs(target_height - current_height) <= 0.24

    # Function to check if the hand can touch the target position
    def can_touch(target_position, standing_platform_height):
        target_height_relative = target_position[2] - standing_platform_height
        return 0.55 <= target_height_relative <= 0.70

    # Main task function
    def close_room_light():
        # Get positions of objects
        lower_stair_position = get_position("lower stair")
        higher_stair_position = get_position("higher stair")
        box_position = get_position("moveable box")
        button_position = get_position("button")

        # Calculate the height of the stairs
        lower_stair_height = lower_stair_position[2] + lowerstair_size[2] / 2
        higher_stair_height = higher_stair_position[2] + higherstair_size[2] / 2

        # Check if we can climb the lower stair directly
        if can_climb(0, lower_stair_height):
            # Climb the lower stair
            climb_to_position(lower_stair_position)
        else:
            # Move the box to use as a step if needed
            if can_climb(0, box_size[2]) and can_climb(
                box_size[2], lower_stair_height
            ):
                # Calculate target position for the box to align with the lower stair
                target_box_position = lower_stair_position - np.array(
                    [0, 0, box_size[2] / 2 + lowerstair_size[2] / 2]
                )
                push_to_position(box_position, target_box_position)
                # Climb onto the box
                climb_to_position(target_box_position)
                # Climb onto the lower stair
                climb_to_position(lower_stair_position)
            else:
                pass
                # raise Exception('Task_Unsolvable_Error')

        # Check if we can climb the higher stair from the lower stair
        if can_climb(lower_stair_height, higher_stair_height):
            climb_to_position(higher_stair_position)
        else:
            pass
            # raise Exception('Task_Unsolvable_Error')

        # Stand up to reach the button
        stand_up()

        # Check if the button can be touched
        if can_touch(button_position, higher_stair_height):
            hand_touch_position(button_position)
        else:
            pass
            # raise Exception('Task_Unsolvable_Error')

        # Sit down at the end
        sit_down()

    # Execute the main task function
    close_room_light()

def generate_llm_plan_main3(self, env_id):
    self.cur_step[env_id][0] = -1
    self.cur_num = 0
    self.box_target_place = None
    self.have_pushed = 0

    def get_position(object):  # TODO: use value from enviroment, change into numpy
        if object == "lower stair":
            return self.object_root_states[env_id, 1, :3].cpu().numpy()
            # return np.array([1.0, 1.0, 0.2])
        elif object == "higher stair":
            return self.object_root_states[env_id, 2, :3].cpu().numpy()
            # return np.array([1.8, 1.0, 0.3])
        elif object == "button":
            return self.button_root_states[env_id, :3].cpu().numpy()
            # return np.array([2.2, 1.2, 1.2])
        elif object == "moveable box":
            if self.box_target_place == None:
                return self.object_pos[env_id].cpu().numpy()
            return self.box_target_place
            # return np.array([0, 0, 0.1])
        return np.array([0, 0, 0])

    def walk_to_position(position):
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1

    def climb_to_position(position):
        if not self.have_pushed:
            self.llm_plan[env_id][self.cur_num][0] = 5
            ready_pos = torch.from_numpy(position).to(self.device)
            ready_pos[0] -= 0.8
            self.llm_plan[env_id][self.cur_num][1:] = ready_pos
            self.cur_num += 1
            print("walk", object, "to", position)
        self.llm_plan[env_id][self.cur_num][0] = 1
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        print("climb to ", position)
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1

    def stand_up():
        # self.llm_plan.append([2, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 2
        self.llm_plan[env_id][self.cur_num][1:] = torch.tensor(
            [0, 0, 0], device=self.device
        )
        self.cur_num += 1
        print("stand up")

    def hand_touch_position(position):
        # self.llm_plan.append([3, torch.from_numpy(position).to(self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 3
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        print("hand touch", position)

    def sit_down():
        # self.llm_plan.append([6, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 6
        self.llm_plan[env_id][self.cur_num][1:] = torch.tensor(
            [0, 0, 0], device=self.device
        )
        self.cur_num += 1
        print("sit down")

    def push_to_position(object, position):
        # self.llm_plan.append([4, torch.from_numpy(position).to(self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 4
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        self.have_pushed = 1
        # self.llm_plan.append([5, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = (
            torch.tensor([0.1, -0.1, 0], device=self.device)
            + self.env_origins[env_id]
        )
        self.cur_num += 1
        self.box_target_place = position

    # Global variables
    robot_size = np.array(
        [0.54, 0.40, 0.40]
    )  # Example values, replace with actual values
    lowerstair_size = np.array(
        [0.8, 0.8, self.object_size[env_id, 1].item()]
    )  # Example values, replace with actual values
    higherstair_size = np.array(
        [0.8, 0.8, self.object_size[env_id, 2].item()]
    )  # Example values, replace with actual values
    box_size = np.array(
        [0.8, 0.8, self.object_size[env_id, 0].item()]
    )  # Example values, replace with actual values

    # Step 1: Check the height of the lower stair
    if lowerstair_size[2] <= 0.24:
        # Step 2: Climb the lower stair
        lowerstair_position = get_position("lower stair")
        target_position_climb_lower_stair = np.array(
            [
                lowerstair_position[0],
                lowerstair_position[1],
                lowerstair_size[2] + robot_size[2] / 2,
            ]
        )
        climb_to_position(target_position_climb_lower_stair)
    else:
        # Step 3: Check if the box can be used as an additional step
        if box_size[2] <= 0.24 and (lowerstair_size[2] - box_size[2]) <= 0.24:
            # Step 5: Move the box in front of the lower stair
            box_position = get_position("moveable box")
            lowerstair_position = get_position("lower stair")
            position_in_front_of_lower_stair = np.array(
                [
                    lowerstair_position[0]
                    - (lowerstair_size[0] / 2 + box_size[0] / 2),
                    lowerstair_position[1],
                    box_position[2],
                ]
            )
            push_to_position(box_position, position_in_front_of_lower_stair)
            # After moving the box, attempt to climb the lower stair again
            lowerstair_position = get_position(
                "lower stair"
            )  # Update position after potentially moving the box
            target_position_climb_lower_stair = np.array(
                [
                    lowerstair_position[0],
                    lowerstair_position[1],
                    lowerstair_size[2] + robot_size[2] / 2,
                ]
            )
            climb_to_position(target_position_climb_lower_stair)
        else:
            raise Exception("Task_Unsolvable_Error")

    # Step 4: Check the height of the higher stair relative to the lower stair
    if (higherstair_size[2] - lowerstair_size[2]) <= 0.24:
        # Step 6: Climb the higher stair
        higherstair_position = get_position("higher stair")
        target_position_climb_higher_stair = np.array(
            [
                higherstair_position[0],
                higherstair_position[1],
                higherstair_size[2] + robot_size[2] / 2,
            ]
        )
        climb_to_position(target_position_climb_higher_stair)
    else:
        raise Exception("Task_Unsolvable_Error")

    # Step 7: Check if the button is within hand touch range
    button_position = get_position("button")
    if 0.55 <= (button_position[2] - higherstair_size[2]) <= 0.70:
        # Step 8: Press the button
        stand_up()
        hand_touch_position(button_position)
        sit_down()
    else:
        raise Exception("Task_Unsolvable_Error")

def generate_llm_plan_main4(self, env_id):
    self.cur_step[env_id][0] = -1
    self.cur_num = 0
    self.box_target_place = None
    self.have_pushed = 0

    def get_position(object):  # TODO: use value from enviroment, change into numpy
        if object == "lower stair":
            return self.object_root_states[env_id, 1, :3].cpu().numpy()
            # return np.array([1.0, 1.0, 0.2])
        elif object == "higher stair":
            return self.object_root_states[env_id, 2, :3].cpu().numpy()
            # return np.array([1.8, 1.0, 0.3])
        elif object == "button":
            return self.button_root_states[env_id, :3].cpu().numpy()
            # return np.array([2.2, 1.2, 1.2])
        elif object == "moveable box":
            if type(self.box_target_place) == type(None):
                return self.object_pos[env_id].cpu().numpy()
            return self.box_target_place
            # return np.array([0, 0, 0.1])
        return np.array([0, 0, 0])

    def walk_to_position(position):
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1

    def climb_to_position(position):
        if not self.have_pushed:
            self.llm_plan[env_id][self.cur_num][0] = 5
            ready_pos = torch.from_numpy(position).to(self.device)
            ready_pos[0] -= 0.8
            self.llm_plan[env_id][self.cur_num][1:] = ready_pos
            self.cur_num += 1
            print("walk to", ready_pos)
        self.llm_plan[env_id][self.cur_num][0] = 1
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        print("climb to ", position)
        if abs(position[0] - get_position("moveable box")[0]) > 0.4:
            self.llm_plan[env_id][self.cur_num][0] = 5
            self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
                self.device
            )
            self.cur_num += 1
            print("walk to ", position)

    def stand_up():
        # self.llm_plan.append([2, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 2
        self.llm_plan[env_id][self.cur_num][1:] = torch.tensor(
            [0, 0, 0], device=self.device
        )
        self.cur_num += 1
        print("stand up")

    def hand_touch_position(position):
        # self.llm_plan.append([3, torch.from_numpy(position).to(self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 3
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        print("hand touch", position)

    def sit_down():
        # self.llm_plan.append([6, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 6
        self.llm_plan[env_id][self.cur_num][1:] = torch.tensor(
            [0, 0, 0], device=self.device
        )
        self.cur_num += 1
        print("sit down")

    def push_to_position(object, position):
        # self.llm_plan.append([4, torch.from_numpy(position).to(self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 4
        self.llm_plan[env_id][self.cur_num][1:] = torch.from_numpy(position).to(
            self.device
        )
        self.cur_num += 1
        self.have_pushed = 1
        print("push to ", position)
        # self.llm_plan.append([5, torch.tensor([0, 0, 0], device=self.device)])
        self.llm_plan[env_id][self.cur_num][0] = 5
        self.llm_plan[env_id][self.cur_num][1:] = (
            torch.tensor([0.1, -0.1, 0], device=self.device)
            + self.env_origins[env_id]
        )
        self.cur_num += 1
        self.box_target_place = position

    # Global variables
    robot_size = np.array(
        [0.54, 0.40, 0.40]
    )  # Example robot size, adjust as needed
    lowerstair_size = np.array(
        [0.8, 0.8, self.object_size[env_id, 1].item()]
    )  # Example sizes, adjust as needed
    higherstair_size = np.array(
        [0.8, 0.8, self.object_size[env_id, 2].item()]
    )  # Example sizes, adjust as needed
    box_size = np.array(
        [0.8, 0.8, self.object_size[env_id, 0].item()]
    )  # Example box size, adjust as needed

    # Step 1: Check the height of the lower stair
    if lowerstair_size[2] <= 0.24:
        # Step 2: Climb the lower stair
        lower_stair_position = get_position("lower stair")
        target_position = np.array(
            [
                lower_stair_position[0],
                lower_stair_position[1],
                lowerstair_size[2] + robot_size[2] / 2,
            ]
        )
        climb_to_position(target_position)
    else:
        # Step 3: Check if the box can be used as an additional step to climb the lower stair
        if box_size[2] <= 0.24 and (lowerstair_size[2] - box_size[2]) <= 0.24:
            # Step 5: Move the box in front of the lower stair
            box_position = get_position("moveable box")
            lower_stair_position = get_position("lower stair")
            position_in_front_of_lower_stair = np.array(
                [
                    lower_stair_position[0]
                    - (lowerstair_size[0] / 2 + box_size[0] / 2),
                    lower_stair_position[1],
                    0,
                ]
            )
            push_to_position(box_position, position_in_front_of_lower_stair)

            # Step 6: Climb onto the box
            box_position = get_position("moveable box")  # Update box position
            target_position = np.array(
                [box_position[0], box_position[1], box_size[2] + robot_size[2] / 2]
            )
            # print("climb_target_pos", target_position)
            climb_to_position(target_position)

            # Step 7: Climb the lower stair from the box
            lower_stair_position = get_position("lower stair")
            target_position = np.array(
                [
                    lower_stair_position[0],
                    lower_stair_position[1],
                    lowerstair_size[2] + robot_size[2] / 2,
                ]
            )
            climb_to_position(target_position)
        else:
            raise Exception("Task_Unsolvable_Error")

    # Step 4: Check the height of the higher stair relative to the lower stair
    if (higherstair_size[2] - lowerstair_size[2]) <= 0.24:
        # Step 8: Climb the higher stair
        higher_stair_position = get_position("higher stair")
        target_position = np.array(
            [
                higher_stair_position[0],
                higher_stair_position[1],
                higherstair_size[2] + robot_size[2] / 2,
            ]
        )
        climb_to_position(target_position)
    else:
        raise Exception("Task_Unsolvable_Error")

    # Step 9: Stand up to reach the button
    stand_up()

    # Step 10: Check if the button is within reach
    button_position = get_position("button")
    if 0.55 <= (button_position[2] - higherstair_size[2]) <= 0.70:
        # Step 11: Press the button
        hand_touch_position(button_position)
    else:
        raise Exception("Task_Unsolvable_Error")

    # Step 12: Sit down
    sit_down()