class Node:
    def __init__(self, task_id, task_name, result=None, state=None, related=None, children=None):
        self.task_id = int(task_id)
        self.task_name = task_name
        self.result = result
        self.state = state
        self.related = related or []
        self.children = children or []

def unpack_tree(root):
    tasks = []

    def serach_node(node):
        tasks.append(node)
        for child in node.children:
            serach_node(child)

    serach_node(root)
    return tasks
