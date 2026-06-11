from __future__ import annotations

import ast
import inspect
import json
import re
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PUBLIC_DUNDERS = {
    "__call__",
    "__enter__",
    "__exit__",
    "__getitem__",
    "__iter__",
    "__len__",
    "__next__",
}


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    return compact(ast.unparse(node))


def clean_doc(docstring: str | None) -> str:
    return inspect.cleandoc(docstring or "")


def doc_summary(docstring: str | None) -> str:
    for line in clean_doc(docstring).splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def is_public_name(name: str) -> bool:
    return not name.startswith("_") or name in PUBLIC_DUNDERS


def relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def module_name(path: Path, package_root: Path) -> str:
    rel = path.relative_to(package_root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(["torch_volpy", *parts]) if parts else "torch_volpy"


def decorator_text(decorator: ast.AST) -> str:
    if isinstance(decorator, ast.Call):
        return unparse(decorator.func)
    return unparse(decorator)


def decorator_names(node: ast.AST) -> list[str]:
    return [decorator_text(decorator) for decorator in getattr(node, "decorator_list", [])]


def has_decorator(node: ast.AST, name: str) -> bool:
    return any(decorator.split(".")[-1] == name for decorator in decorator_names(node))


def assignment_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    if isinstance(node, ast.AnnAssign):
        return [node.target.id] if isinstance(node.target, ast.Name) else []

    names: list[str] = []
    for target in node.targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.extend(elt.id for elt in target.elts if isinstance(elt, ast.Name))
    return names


def literal_string_list(node: ast.AST | None) -> list[str]:
    if node is None:
        return []
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return []
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return list(value)
    return []


def format_arg(arg: ast.arg, default: ast.AST | None = None, prefix: str = "") -> str:
    out = f"{prefix}{arg.arg}"
    if arg.annotation is not None:
        out += f": {unparse(arg.annotation)}"
    if default is not None:
        out += f" = {unparse(default)}"
    return out


def format_signature(node: ast.FunctionDef | ast.AsyncFunctionDef, *, skip_first: bool = False) -> str:
    args = node.args
    pos_args = list(args.posonlyargs) + list(args.args)
    defaults = [None] * (len(pos_args) - len(args.defaults)) + list(args.defaults)
    if skip_first and pos_args:
        pos_args = pos_args[1:]
        defaults = defaults[1:]

    params: list[str] = []
    pos_only_count = max(0, len(args.posonlyargs) - (1 if skip_first and args.posonlyargs else 0))
    for index, (arg, default) in enumerate(zip(pos_args, defaults)):
        params.append(format_arg(arg, default))
        if pos_only_count and index == pos_only_count - 1:
            params.append("/")

    if args.vararg is not None:
        params.append(format_arg(args.vararg, prefix="*"))
    elif args.kwonlyargs:
        params.append("*")

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        params.append(format_arg(arg, default))

    if args.kwarg is not None:
        params.append(format_arg(args.kwarg, prefix="**"))

    signature = f"({', '.join(params)})"
    if node.returns is not None:
        signature += f" -> {unparse(node.returns)}"
    return signature


def node_source(node: ast.AST, path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": relpath(path, root),
        "line": getattr(node, "lineno", 1),
        "endLine": getattr(node, "end_lineno", getattr(node, "lineno", 1)),
    }


def make_search_text(item: dict[str, Any]) -> str:
    values = [
        item.get("kind", ""),
        item.get("name", ""),
        item.get("qualifiedName", ""),
        item.get("signature", ""),
        item.get("summary", ""),
        item.get("docstring", ""),
        " ".join(item.get("decorators", [])),
        " ".join(item.get("bases", [])),
        item.get("annotation", ""),
        item.get("default", ""),
    ]
    return "\n".join(str(value) for value in values if value)


def item_base(
    *,
    kind: str,
    name: str,
    qualified_name: str,
    module: str,
    source: dict[str, Any],
    public: bool,
    docstring: str | None = None,
    signature: str = "",
    decorators: list[str] | None = None,
    parent: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "kind": kind,
        "name": name,
        "qualifiedName": qualified_name,
        "module": module,
        "signature": signature,
        "summary": doc_summary(docstring),
        "docstring": clean_doc(docstring),
        "decorators": decorators or [],
        "source": source,
        "public": public,
    }
    if parent is not None:
        item["parent"] = parent
    item["searchText"] = make_search_text(item).lower()
    return item


def class_signature(node: ast.ClassDef) -> str:
    for child in node.body:
        if isinstance(child, ast.FunctionDef) and child.name == "__init__":
            return format_signature(child, skip_first=True)
    return "()"


def collect_dataclass_fields(
    node: ast.ClassDef,
    *,
    class_qname: str,
    class_public: bool,
    module: str,
    path: Path,
    root: Path,
) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    if not has_decorator(node, "dataclass"):
        return fields

    for child in node.body:
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            name = child.target.id
            annotation = unparse(child.annotation)
            field_doc = f"Dataclass field `{name}`."
            if annotation:
                field_doc += f" Type: `{annotation}`."
            item = item_base(
                kind="field",
                name=name,
                qualified_name=f"{class_qname}.{name}",
                module=module,
                parent=class_qname,
                signature=f": {annotation}",
                docstring=field_doc,
                source=node_source(child, path, root),
                public=class_public and is_public_name(name),
            )
            item["annotation"] = annotation
            if child.value is not None:
                item["default"] = unparse(child.value)
            item["searchText"] = make_search_text(item).lower()
            fields.append(item)
    return fields


def collect_class_items(
    node: ast.ClassDef,
    *,
    module: str,
    path: Path,
    root: Path,
    exports: set[str],
) -> list[dict[str, Any]]:
    qname = f"{module}.{node.name}"
    decorators = decorator_names(node)
    docstring = ast.get_docstring(node)
    public = is_public_name(node.name) or node.name in exports
    class_item = item_base(
        kind="class",
        name=node.name,
        qualified_name=qname,
        module=module,
        signature=class_signature(node),
        docstring=docstring,
        decorators=decorators,
        source=node_source(node, path, root),
        public=public,
    )
    class_item["bases"] = [unparse(base) for base in node.bases]
    class_item["dataclass"] = has_decorator(node, "dataclass")
    class_item["searchText"] = make_search_text(class_item).lower()

    items = [class_item]
    items.extend(
        collect_dataclass_fields(
            node,
            class_qname=qname,
            class_public=public,
            module=module,
            path=path,
            root=root,
        )
    )

    for child in node.body:
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorators = decorator_names(child)
        kind = "property" if has_decorator(child, "property") else "method"
        if has_decorator(child, "staticmethod"):
            kind = "staticmethod"
        elif has_decorator(child, "classmethod"):
            kind = "classmethod"

        item = item_base(
            kind=kind,
            name=child.name,
            qualified_name=f"{qname}.{child.name}",
            module=module,
            parent=qname,
            signature=format_signature(child),
            docstring=ast.get_docstring(child),
            decorators=decorators,
            source=node_source(child, path, root),
            public=public and is_public_name(child.name),
        )
        items.append(item)
    return items


def collect_module_items(path: Path, *, package_root: Path, root: Path) -> list[dict[str, Any]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    module = module_name(path, package_root)
    exports: set[str] = set()

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and "__all__" in assignment_names(node):
            exports.update(literal_string_list(getattr(node, "value", None)))

    items: list[dict[str, Any]] = [
        item_base(
            kind="module",
            name=module,
            qualified_name=module,
            module=module,
            docstring=ast.get_docstring(tree),
            source={"path": relpath(path, root), "line": 1, "endLine": 1},
            public=all(not part.startswith("_") for part in module.split(".")),
        )
    ]
    items[0]["exports"] = sorted(exports)
    items[0]["searchText"] = make_search_text(items[0]).lower()

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            items.extend(
                collect_class_items(
                    node,
                    module=module,
                    path=path,
                    root=root,
                    exports=exports,
                )
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = decorator_names(node)
            item = item_base(
                kind="function",
                name=node.name,
                qualified_name=f"{module}.{node.name}",
                module=module,
                signature=format_signature(node),
                docstring=ast.get_docstring(node),
                decorators=decorators,
                source=node_source(node, path, root),
                public=is_public_name(node.name) or node.name in exports,
            )
            items.append(item)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for name in assignment_names(node):
                if name == "__all__":
                    continue
                value = getattr(node, "value", None)
                annotation = unparse(node.annotation) if isinstance(node, ast.AnnAssign) else ""
                value_text = unparse(value)
                kind = "alias" if annotation or any(token in value_text for token in ("Union", "Tuple", "Optional", "|")) else "constant"
                item = item_base(
                    kind=kind,
                    name=name,
                    qualified_name=f"{module}.{name}",
                    module=module,
                    signature=f": {annotation}" if annotation else "",
                    source=node_source(node, path, root),
                    public=is_public_name(name) or name in exports,
                )
                if annotation:
                    item["annotation"] = annotation
                if value_text:
                    item["default"] = value_text
                item["searchText"] = make_search_text(item).lower()
                items.append(item)

    return items


def project_metadata(root: Path) -> dict[str, Any]:
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return {"name": "torch-volpy"}
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    return {
        "name": project.get("name", "torch-volpy"),
        "version": project.get("version", ""),
        "description": project.get("description", ""),
        "requiresPython": project.get("requires-python", ""),
        "dependencies": project.get("dependencies", []),
        "optionalDependencies": project.get("optional-dependencies", {}),
    }


def build_index(root: Path) -> dict[str, Any]:
    package_root = root / "src" / "torch_volpy"
    if not package_root.exists():
        raise FileNotFoundError(f"Package root not found: {package_root}")

    files = sorted(
        path
        for path in package_root.rglob("*.py")
        if "__pycache__" not in path.parts
    )

    items: list[dict[str, Any]] = []
    for path in files:
        items.extend(collect_module_items(path, package_root=package_root, root=root))

    stats: dict[str, int] = {}
    for item in items:
        stats[item["kind"]] = stats.get(item["kind"], 0) + 1

    return {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "project": project_metadata(root),
        "sourceRoot": "src",
        "stats": stats,
        "items": items,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_path = Path(__file__).resolve().with_name("api.json")
    index = build_index(root)
    out_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path.relative_to(root).as_posix()} with {len(index['items'])} API entries")


if __name__ == "__main__":
    main()
