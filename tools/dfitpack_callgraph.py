#!/usr/bin/env python3
"""
Parse dfitpack.c and emit a list of functions and the functions they call.
"""
import sys
from pathlib import Path
from tree_sitter import Language, Parser
import tree_sitter_c

C_LANGUAGE = Language(tree_sitter_c.language())

def get_text(source, node):
    return source[node.start_byte:node.end_byte].decode()

def collect_calls(source, node, calls):
    if node.type == 'call_expression':
        func = node.child_by_field_name('function')
        if func is not None and func.type == 'identifier':
            calls.add(get_text(source, func))
    for child in node.children:
        collect_calls(source, child, calls)

def get_function_name(source, node):
    """Extract the function name from a function_definition node."""
    declarator = node.child_by_field_name('declarator')
    while declarator is not None:
        if declarator.type == 'function_declarator':
            inner = declarator.child_by_field_name('declarator')
            if inner is not None and inner.type == 'identifier':
                return get_text(source, inner)
            return None
        # pointer_declarator wraps a function_declarator
        next_decl = declarator.child_by_field_name('declarator')
        if next_decl is None:
            break
        declarator = next_decl
    return None

def main():
    path = Path(__file__).parent.parent / 'scipy/interpolate/src/dfitpack.c'
    source = path.read_bytes()

    parser = Parser(C_LANGUAGE)
    tree = parser.parse(source)

    results = {}  # func_name -> sorted list of called functions

    for node in tree.root_node.children:
        if node.type != 'function_definition':
            continue
        name = get_function_name(source, node)
        if name is None:
            continue
        body = node.child_by_field_name('body')
        if body is None:
            continue
        calls = set()
        collect_calls(source, body, calls)
        calls.discard(name)  # exclude self-calls from the list but note them
        self_recursive = name in {
            get_text(source, n)
            for n in body.children
            if n.type == 'call_expression'
        }
        results[name] = {
            'calls': sorted(calls),
            'recursive': name in calls or self_recursive,
        }

    out_path = Path(__file__).parent.parent / 'dfitpack_callgraph.txt'
    with out_path.open('w') as f:
        for func, info in sorted(results.items()):
            rec = ' [RECURSIVE]' if info['recursive'] else ''
            f.write(f'{func}{rec}:\n')
            for callee in info['calls']:
                f.write(f'    {callee}\n')
            f.write('\n')

    print(f'Wrote {out_path}')
    print(f'Total functions: {len(results)}')

if __name__ == '__main__':
    main()
