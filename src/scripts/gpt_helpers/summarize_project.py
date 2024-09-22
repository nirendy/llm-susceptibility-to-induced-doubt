import os
import ast
from pathlib import Path


def get_type_annotation(arg):
    if arg.annotation:
        if isinstance(arg.annotation, ast.Name):
            return arg.annotation.id
        elif isinstance(arg.annotation, ast.Subscript):
            value_id = arg.annotation.value.id if isinstance(arg.annotation.value, ast.Name) else None
            slice_id = arg.annotation.slice.id if isinstance(arg.annotation.slice, ast.Name) else None
            return (value_id, slice_id)
    return None


def add_parent_info(tree):
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node


def get_init_function_body(node):
    for sub_node in node.body:
        if isinstance(sub_node, ast.FunctionDef) and sub_node.name == "__init__":
            return ast.unparse(sub_node)
    return None


def get_class_and_function_info(filepath):
    with open(filepath, 'r') as file:
        file_content = file.read()
    tree = ast.parse(file_content)
    add_parent_info(tree)

    class_info = {}
    module_functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            methods = []
            init_body = None
            for sub_node in node.body:
                if isinstance(sub_node, ast.FunctionDef):
                    args = [(arg.arg, get_type_annotation(arg)) for arg in sub_node.args.args]
                    methods.append({
                        'name': sub_node.name,
                        'args': args
                    })
                    if sub_node.name == "__init__":
                        init_body = ast.unparse(sub_node)
            class_info[class_name] = {
                'methods': methods,
                'init_body': init_body
            }
        elif isinstance(node, ast.FunctionDef) and isinstance(node.parent, ast.Module):
            args = [(arg.arg, get_type_annotation(arg)) for arg in node.args.args]
            module_functions.append({
                'name': node.name,
                'args': args
            })

    return class_info, module_functions


def summarize_directory(directory):
    summary = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                if file == '__init__.py':
                    continue
                filepath = os.path.join(root, file)
                class_info, module_functions = get_class_and_function_info(filepath)
                summary.append({
                    'file': filepath,
                    'classes': class_info,
                    'functions': module_functions,
                })
    return summary


def print_summary(summary):
    for item in summary:
        print(f"File: {item['file']}")
        if item['classes']:
            for class_name, class_details in item['classes'].items():
                print(f"  Class: {class_name}")
                for method in class_details['methods']:
                    args = ', '.join([f"{arg[0]}: {arg[1]}" if arg[1] else arg[0] for arg in method['args']])
                    print(f"    - Method: {method['name']}({args})")
                if class_details['init_body']:
                    print(f"    - __init__ implementation:\n{class_details['init_body']}")
        if item['functions']:
            print("  Functions:")
            for function in item['functions']:
                args = ', '.join([f"{arg[0]}: {arg[1]}" if arg[1] else arg[0] for arg in function['args']])
                print(f"    - {function['name']}({args})")
        print()


if __name__ == "__main__":
    project_dir = Path(__file__)
    while not (project_dir / 'src').exists():
        project_dir = project_dir.parent

    summary = summarize_directory(project_dir / 'src')
    print_summary(summary)
