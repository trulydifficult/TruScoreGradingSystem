import ast
import sys
import json
from pathlib import Path


class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.structure = []
        self.stack = [self.structure]

    def _push(self, item):
        self.stack[-1].append(item)
        if "children" in item:
            self.stack.append(item["children"])

    def _pop(self, item):
        if "children" in item:
            self.stack.pop()

    def visit_FunctionDef(self, node):
        obj = {
            "type": "function",
            "name": node.name,
            "line": node.lineno,
            "args": [a.arg for a in node.args.args],
            "children": []
        }
        self._push(obj)
        self.generic_visit(node)
        self._pop(obj)

    def visit_AsyncFunctionDef(self, node):
        obj = {
            "type": "async_function",
            "name": node.name,
            "line": node.lineno,
            "args": [a.arg for a in node.args.args],
            "children": []
        }
        self._push(obj)
        self.generic_visit(node)
        self._pop(obj)

    def visit_ClassDef(self, node):
        bases = [ast.unparse(b) for b in node.bases] if node.bases else []
        obj = {
            "type": "class",
            "name": node.name,
            "bases": bases,
            "line": node.lineno,
            "class_vars": [],
            "children": []
        }

        # class vars
        for body in node.body:
            if isinstance(body, ast.Assign):
                for target in body.targets:
                    if isinstance(target, ast.Name):
                        obj["class_vars"].append({
                            "name": target.id,
                            "line": body.lineno
                        })

        self._push(obj)
        self.generic_visit(node)
        self._pop(obj)

    def visit_Attribute(self, node):
        # instance vars: self.x
        if (isinstance(node.ctx, ast.Store)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"):
            self.stack[-1].append({
                "type": "instance_var",
                "name": f"self.{node.attr}",
                "line": node.lineno
            })

        self.generic_visit(node)


# ------------------------------
# Markdown Rendering
# ------------------------------
def to_markdown(items, depth=0):
    md = []
    indent = "    " * depth

    for item in items:
        if item["type"] == "class":
            md.append(f"{indent}- **Class `{item['name']}`** (line {item['line']})")
            if item["bases"]:
                md.append(f"{indent}    - Bases: {', '.join(item['bases'])}")
            if item["class_vars"]:
                md.append(f"{indent}    - Class Vars:")
                for var in item["class_vars"]:
                    md.append(f"{indent}        - `{var['name']}` (line {var['line']})")

        elif item["type"] in ("function", "async_function"):
            t = "Async Function" if item["type"] == "async_function" else "Function"
            md.append(f"{indent}- **{t} `{item['name']}`** (line {item['line']})")
            md.append(f"{indent}    - Args: {item['args']}")

        elif item["type"] == "instance_var":
            md.append(f"{indent}- Instance Var `{item['name']}` (line {item['line']})")

        if "children" in item and item["children"]:
            md.append(to_markdown(item["children"], depth + 1))

    return "\n".join(md)


# ------------------------------
# Analyze
# ------------------------------
def analyze_file(path):
    code = Path(path).read_text()
    tree = ast.parse(code)
    analyzer = Analyzer()
    analyzer.visit(tree)
    return analyzer.structure


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python analyze_structure.py script.py --json [outfile.json]")
        print("  python analyze_structure.py script.py --md   [outfile.md]")
        sys.exit(1)

    script_path = sys.argv[1]
    mode = sys.argv[2]

    # Optional custom output filename
    out_file = None
    if len(sys.argv) >= 4:
        out_file = sys.argv[3]

    structure = analyze_file(script_path)

    script_name = Path(script_path).stem

    if mode == "--json":
        if not out_file:
            out_file = f"{script_name}_structure.json"
        Path(out_file).write_text(json.dumps(structure, indent=4))
        print(f"JSON exported to: {out_file}")

    elif mode == "--md":
        if not out_file:
            out_file = f"{script_name}_structure.md"
        md = "# Code Structure\n\n" + to_markdown(structure)
        Path(out_file).write_text(md)
        print(f"Markdown exported to: {out_file}")

    else:
        print("Unknown mode. Use --json or --md")
