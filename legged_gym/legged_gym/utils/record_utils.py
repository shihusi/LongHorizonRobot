import json


def save_class_variables(obj, variables_to_store, filename):
    data = {}
    for var in variables_to_store:
        if hasattr(obj, var):
            data[var] = getattr(obj, var)
    
    with open(filename, 'w') as json_file:
        for key, value in data.items():
            json.dump({key: value}, json_file)
            json_file.write('\n')

def save_object_to_json(obj, filename):
    obj_dict = obj.__dict__
    print(">> Task Setup", obj.test_llm_code, obj_dict)
    
    with open(filename, 'w') as json_file:
        json.dump(obj_dict, json_file, indent=4)




