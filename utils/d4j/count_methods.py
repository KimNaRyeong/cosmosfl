import os
import javalang
import argparse

def count_methods(directory):
    method_count = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.java'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    try:
                        tree = javalang.parse.parse(content)
                        for _, node in tree.filter(javalang.tree.MethodDeclaration):
                            method_count += 1
                    except javalang.parser.JavaSyntaxError:
                        print(f"Syntax error in file: {file_path}")
    return method_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('project_dir', type=str)
    parser.add_argument('--output', '-o', type=str)
    args = parser.parse_args()
    methods = count_methods(args.project_dir)
    with open(args.output, 'w') as f:
        f.write(f"Number of methods: {methods}")
