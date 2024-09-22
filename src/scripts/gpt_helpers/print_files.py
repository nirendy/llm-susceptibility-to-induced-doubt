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
            for sub_node in node.body:
                if isinstance(sub_node, ast.FunctionDef):
                    args = [(arg.arg, get_type_annotation(arg)) for arg in sub_node.args.args]
                    method_code = ast.unparse(sub_node) if hasattr(ast, 'unparse') else ''
                    methods.append({
                        'name': sub_node.name,
                        'args': args,
                        'code': method_code
                    })
            class_info[class_name] = {
                'methods': methods,
            }
        elif isinstance(node, ast.FunctionDef) and isinstance(node.parent, ast.Module):
            args = [(arg.arg, get_type_annotation(arg)) for arg in node.args.args]
            function_code = ast.unparse(node) if hasattr(ast, 'unparse') else ''
            module_functions.append({
                'name': node.name,
                'args': args,
                'code': function_code
            })

    return class_info, module_functions


def summarize_directory(directory, project_dir):
    summary = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                if file == '__init__.py':
                    continue
                filepath = os.path.join(root, file)
                relative_path = os.path.relpath(filepath, start=project_dir)
                class_info, module_functions = get_class_and_function_info(filepath)
                summary.append({
                    'file': relative_path,
                    'classes': class_info,
                    'functions': module_functions,
                })
    return summary


def get_verbosity_for_file(filepath, verbosity_settings, default_verbosity):
    # Normalize paths to ensure consistent comparison
    filepath = os.path.normpath(filepath)
    max_match_length = -1
    verbosity = default_verbosity
    for path, level in verbosity_settings.items():
        path = os.path.normpath(path)
        if filepath == path or filepath.startswith(path + os.sep):
            # Longer paths are more specific
            if len(path) > max_match_length:
                max_match_length = len(path)
                verbosity = level
    return verbosity


def print_summary(summary, default_verbosity=1, verbosity_settings=None):
    if verbosity_settings is None:
        verbosity_settings = {}
    for item in summary:
        filepath = item['file']
        verbosity = get_verbosity_for_file(filepath, verbosity_settings, default_verbosity)

        if verbosity == 0:
            print(f"File: {filepath}")
            continue  # Only print the file name
        elif verbosity >= 1:
            print(f"File: {filepath}")
            if item['classes']:
                for class_name, class_details in item['classes'].items():
                    print(f"  Class: {class_name}")
                    for method in class_details['methods']:
                        args = ', '.join([f"{arg[0]}: {arg[1]}" if arg[1] else arg[0] for arg in method['args']])
                        print(f"    - Method: {method['name']}({args})")
                        if verbosity == 2:
                            print(f"      Code:\n{method['code']}")
            if item['functions']:
                print("  Functions:")
                for function in item['functions']:
                    args = ', '.join([f"{arg[0]}: {arg[1]}" if arg[1] else arg[0] for arg in function['args']])
                    print(f"    - {function['name']}({args})")
                    if verbosity == 2:
                        print(f"      Code:\n{function['code']}")
            print()


if __name__ == "__main__":
    # Set the default verbosity level (0, 1, or 2)
    default_verbosity = 0

    # Define verbosity levels for specific files or directories
    verbosity_settings = {
        'src/apps/chat': 2,  # Full implementations for 'chat' app
        'src/utils': 1,      # Signatures for 'utils' directory
    }

    project_dir = Path(__file__).resolve()
    while not (project_dir / 'src').exists():
        project_dir = project_dir.parent

    summary = summarize_directory(project_dir / 'src', project_dir)
    print_summary(summary, default_verbosity=default_verbosity, verbosity_settings=verbosity_settings)
