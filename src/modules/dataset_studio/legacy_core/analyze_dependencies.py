#!/usr/bin/env python3
"""
Dataset Creator Dependency Analysis Script
Analyzes all Python files in dataset_creator to map imports, functions, and dependencies
NO DELETIONS - Pure analysis only!
"""

import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict


class DependencyAnalyzer:
    """Analyze Python files for imports, functions, classes, and dependencies"""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.files_data = {}
        self.import_graph = defaultdict(set)
        self.reverse_import_graph = defaultdict(set)
        self.all_functions = defaultdict(list)
        self.all_classes = defaultdict(list)
        
    def analyze_all_files(self):
        """Analyze all Python files in the directory"""
        print(f"\n{'='*80}")
        print(f"DATASET CREATOR DEPENDENCY ANALYSIS")
        print(f"{'='*80}\n")
        print(f"Analyzing directory: {self.root_dir}\n")
        
        # Find all Python files
        py_files = list(self.root_dir.rglob("*.py"))
        print(f"Found {len(py_files)} Python files\n")
        
        # Analyze each file
        for py_file in sorted(py_files):
            if '__pycache__' in str(py_file):
                continue
            self.analyze_file(py_file)
        
        # Generate reports
        self.generate_reports()
    
    def analyze_file(self, file_path: Path):
        """Analyze a single Python file"""
        rel_path = file_path.relative_to(self.root_dir)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content, filename=str(file_path))
            
            # Extract information
            file_info = {
                'path': str(rel_path),
                'size': len(content),
                'lines': content.count('\n') + 1,
                'imports': [],
                'from_imports': [],
                'functions': [],
                'classes': [],
                'docstring': ast.get_docstring(tree) or "No docstring"
            }
            
            # Walk AST
            for node in ast.walk(tree):
                # Imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        file_info['imports'].append(alias.name)
                
                # From imports
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        import_str = f"{module}.{alias.name}" if module else alias.name
                        file_info['from_imports'].append(import_str)
                        
                        # Track internal dependencies
                        if 'dataset_creator' in module:
                            imported_file = module.replace('src.core.dataset_creator.', '')
                            self.import_graph[str(rel_path)].add(imported_file)
                            self.reverse_import_graph[imported_file].add(str(rel_path))
                
                # Functions
                elif isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node) or ""
                    }
                    file_info['functions'].append(func_info)
                    self.all_functions[node.name].append(str(rel_path))
                
                # Classes
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'line': node.lineno,
                        'bases': [self._get_name(base) for base in node.bases],
                        'methods': [],
                        'docstring': ast.get_docstring(node) or ""
                    }
                    
                    # Get methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            class_info['methods'].append(item.name)
                    
                    file_info['classes'].append(class_info)
                    self.all_classes[node.name].append(str(rel_path))
            
            self.files_data[str(rel_path)] = file_info
            
        except Exception as e:
            print(f"⚠️  Error analyzing {rel_path}: {e}")
    
    def _get_name(self, node):
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)
    
    def generate_reports(self):
        """Generate comprehensive analysis reports"""
        
        # Report 1: File Summary
        print(f"\n{'='*80}")
        print("FILE SUMMARY")
        print(f"{'='*80}\n")
        
        for file_path in sorted(self.files_data.keys()):
            info = self.files_data[file_path]
            print(f"📄 {file_path}")
            print(f"   Lines: {info['lines']} | Functions: {len(info['functions'])} | Classes: {len(info['classes'])}")
            print(f"   Imports: {len(info['imports']) + len(info['from_imports'])}")
            if info['docstring'] and info['docstring'] != "No docstring":
                doc_preview = info['docstring'][:80].replace('\n', ' ')
                print(f"   Doc: {doc_preview}...")
            print()
        
        # Report 2: Internal Dependencies
        print(f"\n{'='*80}")
        print("INTERNAL DEPENDENCIES (Who imports whom)")
        print(f"{'='*80}\n")
        
        for file_path in sorted(self.import_graph.keys()):
            imports = self.import_graph[file_path]
            if imports:
                print(f"📄 {file_path}")
                for imp in sorted(imports):
                    print(f"   → imports {imp}")
                print()
        
        # Report 3: Reverse Dependencies (Who uses this file)
        print(f"\n{'='*80}")
        print("REVERSE DEPENDENCIES (Who uses this file)")
        print(f"{'='*80}\n")
        
        for file_path in sorted(self.reverse_import_graph.keys()):
            used_by = self.reverse_import_graph[file_path]
            if used_by:
                print(f"📄 {file_path}")
                print(f"   Used by {len(used_by)} file(s):")
                for user in sorted(used_by):
                    print(f"   ← {user}")
                print()
        
        # Report 4: Orphaned Files (Not imported by anyone)
        print(f"\n{'='*80}")
        print("ORPHANED FILES (Not imported by any other file)")
        print(f"{'='*80}\n")
        
        all_files = set(self.files_data.keys())
        imported_files = set()
        for imports in self.import_graph.values():
            for imp in imports:
                # Try to match import to file
                for file_path in all_files:
                    if imp in file_path or file_path.replace('/', '.').replace('.py', '') in imp:
                        imported_files.add(file_path)
        
        orphaned = all_files - imported_files
        for file_path in sorted(orphaned):
            info = self.files_data[file_path]
            print(f"⚠️  {file_path}")
            print(f"    Lines: {info['lines']} | Classes: {len(info['classes'])} | Functions: {len(info['functions'])}")
            # Check if it's an entry point
            if 'run_' in file_path or 'main' in file_path or 'launch' in file_path:
                print(f"    ℹ️  Likely an ENTRY POINT (launcher/main script)")
            print()
        
        # Report 5: Duplicate Functions/Classes
        print(f"\n{'='*80}")
        print("DUPLICATE FUNCTIONS (Same name in multiple files)")
        print(f"{'='*80}\n")
        
        for func_name, files in sorted(self.all_functions.items()):
            if len(files) > 1:
                print(f"🔁 {func_name}")
                for file_path in sorted(files):
                    print(f"   → {file_path}")
                print()
        
        print(f"\n{'='*80}")
        print("DUPLICATE CLASSES (Same name in multiple files)")
        print(f"{'='*80}\n")
        
        for class_name, files in sorted(self.all_classes.items()):
            if len(files) > 1:
                print(f"🔁 {class_name}")
                for file_path in sorted(files):
                    print(f"   → {file_path}")
                print()
        
        # Report 6: External Dependencies
        print(f"\n{'='*80}")
        print("EXTERNAL DEPENDENCIES (Third-party imports)")
        print(f"{'='*80}\n")
        
        external_deps = defaultdict(list)
        for file_path, info in self.files_data.items():
            for imp in info['imports'] + info['from_imports']:
                # Skip standard library and internal imports
                if not imp.startswith('src.') and '.' in imp:
                    root_package = imp.split('.')[0]
                    if root_package not in ['os', 'sys', 'json', 'pathlib', 'typing', 
                                           'dataclasses', 'collections', 'time', 'datetime',
                                           'threading', 'logging', 'ast']:
                        external_deps[root_package].append(file_path)
        
        for package, files in sorted(external_deps.items()):
            print(f"📦 {package}")
            print(f"   Used by {len(set(files))} file(s)")
            for file_path in sorted(set(files))[:5]:  # Show first 5
                print(f"   → {file_path}")
            if len(set(files)) > 5:
                print(f"   ... and {len(set(files)) - 5} more")
            print()
        
        # Report 7: Key Classes and Their Files
        print(f"\n{'='*80}")
        print("KEY CLASSES DIRECTORY")
        print(f"{'='*80}\n")
        
        important_classes = ['DatasetStudioMain', 'TruScoreDatasetFrame', 'ImageCard', 
                            'DashboardView', 'ProjectManagerView', 'ConversionWorker',
                            'EnterpriseDatasetStudio', 'FlowLayout']
        
        for class_name in important_classes:
            if class_name in self.all_classes:
                print(f"🎯 {class_name}")
                for file_path in self.all_classes[class_name]:
                    info = self.files_data[file_path]
                    # Find the class info
                    for cls in info['classes']:
                        if cls['name'] == class_name:
                            print(f"   📄 {file_path}")
                            print(f"      Methods: {', '.join(cls['methods'][:10])}")
                            if len(cls['methods']) > 10:
                                print(f"      ... and {len(cls['methods']) - 10} more")
                            if cls['docstring']:
                                doc_preview = cls['docstring'][:80].replace('\n', ' ')
                                print(f"      Doc: {doc_preview}")
                            print()
        
        # Save JSON report
        self.save_json_report()
    
    def save_json_report(self):
        """Save complete analysis to JSON"""
        output_file = self.root_dir / "DEPENDENCY_ANALYSIS.json"
        
        report = {
            'files': self.files_data,
            'import_graph': {k: list(v) for k, v in self.import_graph.items()},
            'reverse_dependencies': {k: list(v) for k, v in self.reverse_import_graph.items()},
            'functions': {k: v for k, v in self.all_functions.items()},
            'classes': {k: v for k, v in self.all_classes.items()}
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✅ Full analysis saved to: {output_file}")
        print(f"{'='*80}\n")


def main():
    """Run the dependency analysis"""
    dataset_creator_dir = Path(__file__).parent
    
    analyzer = DependencyAnalyzer(dataset_creator_dir)
    analyzer.analyze_all_files()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nReview the reports above to understand:")
    print("  1. Which files are actually being used")
    print("  2. Which files import which other files")
    print("  3. Which files are orphaned (not imported by anyone)")
    print("  4. Which classes/functions are duplicated")
    print("  5. What external dependencies are needed")
    print("\n⚠️  NO FILES WERE DELETED - This is analysis only!\n")


if __name__ == "__main__":
    main()
