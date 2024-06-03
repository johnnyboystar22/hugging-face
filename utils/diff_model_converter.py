# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import glob
import importlib
import re
from typing import Dict

import libcst as cst
from check_copies import run_ruff
from libcst import ClassDef, CSTTransformer, CSTVisitor
from libcst import matchers as m
from libcst.metadata import MetadataWrapper, ParentNodeProvider, PositionProvider, ScopeProvider

from transformers import logging


logger = logging.get_logger(__name__)


AUTO_GENERATED_MESSAGE = """#           🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#               This file was automatically generated from <path_to_diff_file.py>.
#         Do NOT edit this file manually as any edits will be overwritten by the generation of
#         the file from the diff. If any change should be done, please apply the change to the
#                                    diff.py file directly.
#           🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
"""


def get_module_source_from_name(module_name: str) -> str:
    # Extract the source code from the module name
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return f"Module {module_name} not found"

    with open(spec.origin, "r") as file:
        source_code = file.read()
    return source_code


class ClassFinder(CSTVisitor):
    """A visitor class which analyses a module, creating a mapping of dependencies between classes and functions.
    For example if the visited code has
    ```python3
    def init_value(): return 1

    class LlamaModel(PreTrainedModel):
        def __init__(self):
            super().__init__(self)
            self.value = init_value()
    ```
    then the `class_dependency_mapping` should be: `{"LlamaModel":["PreTrainedModel","init_value"], "init_value":[]}

    The dependency mapping is updated via the `visit_Name`, `visit_Arg` and `visit_Decorator`. This is very broad, and by
    checking the parent node, or the scope of a `cst.Name` or `cst.Arg` or `cst.Decorator` we are able to map the
    dependence parent -> child.

    When visiting such nodes, we update the dependency of the parent node, to take into account the visited node.

    All `visit_XXX` correspond to the code executed when vising the cst.Node of type XXX.
    """

    METADATA_DEPENDENCIES = (ParentNodeProvider, ScopeProvider, PositionProvider)

    def __init__(self, python_module: cst.Module):
        # fmt: off
        self.python_module: cst.Module = python_module  # original cst.Module being visited
        self.classes: Dict[str, cst.ClassDef] = {}      # stores a mapping from classname to the cst.Node
        self.imports = {}                               # stores all import statements
        self.function_def = {}                          # stores global scope function definition
        self.assignments = {}                           # LLAMA_DOCSTRING
        self.class_dependency_mapping = {}              # "LlamaModel":["LlamaDecoderLayer, "LlamaRMSNorm", "LlamaPreTrainedModel"], "LlamaDecoderLayer":["LlamaAttention","Llama"]
        # fmt: on

    def _update_class_dependency(self, name, value):
        """Update the dependency mapping for `name` with `value` by appending the previous
        dependencies to the new `value`.
        """
        dep = set(self.class_dependency_mapping.get(value, set()))
        dep |= set(self.class_dependency_mapping.get(name, {})) | set({value})
        self.class_dependency_mapping[name] = dep

    def visit_ClassDef(self, node: ClassDef) -> None:
        """We don't have non global scope class defs in transformers. Here we add the inheritance dependencies"""
        self.classes[node.name.value] = node
        for k in node.bases:  # deal with inheritance
            base_name = self.python_module.code_for_node(k)
            self._update_class_dependency(node.name.value, base_name)

    def visit_SimpleStatementLine(self, node):
        """
        Global Assigns like `GEMMA_INPUT_DOCSTRING = 'THIS IS THE INPUT' and all import statements
        are extracted and saved in their corresponding dict. They are then used when updating dependency mappings.
        """
        if m.matches(node, m.SimpleStatementLine(body=[m.Assign()])) and m.matches(
            self.get_metadata(cst.metadata.ParentNodeProvider, node), m.Module()
        ):
            self.assignments[node.body[0].targets[0].target.value] = node
        if m.matches(node, m.SimpleStatementLine(body=[m.Import() | m.ImportFrom()])):
            self.imports[node.body[0].names] = node

    def visit_FunctionDef(self, node):
        parent_node = self.get_metadata(cst.metadata.ParentNodeProvider, node)
        if m.matches(parent_node, m.Module()):
            self.function_def[node.name.value] = node

    def leave_If(self, node):
        for stmt in node.body.body:
            if m.matches(stmt, m.SimpleStatementLine(body=[m.ImportFrom() | m.Import()])):
                self.imports[stmt.body[0].names] = node

    def leave_Name(self, node):
        if node.value in self.classes.keys() | self.assignments.keys() | self.function_def.keys():
            parent = self.get_metadata(cst.metadata.ScopeProvider, node)
            if not isinstance(parent, cst.metadata.scope_provider.GlobalScope):
                self._update_class_dependency(parent._name_prefix.split(".")[0], node.value)

    def leave_Arg(self, node):
        if m.matches(node.value, m.Name()):
            parent = self.get_metadata(ParentNodeProvider, node)
            if m.matches(parent, m.ClassDef()) and parent.bases:
                self._update_class_dependency(parent.name.value, node.value.value)

    def leave_Dict(self, node):
        parent = self.get_metadata(cst.metadata.ParentNodeProvider, node)
        if m.matches(parent, m.Assign(targets=[m.AssignTarget()])):
            name = parent.targets[0].target.value
            if name in self.assignments:
                for k in node.elements:
                    dep_name = k.value.value
                    if dep_name in self.classes:
                        self._update_class_dependency(name, dep_name)

    def leave_Decorator(self, node):
        if hasattr(node.decorator, "args"):
            for k in node.decorator.args:
                if k.value.value in self.assignments:
                    parent = self.get_metadata(cst.metadata.ParentNodeProvider, node)
                    scope = self.get_metadata(cst.metadata.ScopeProvider, node)
                    name = scope._name_prefix.split(".")[0] if scope._name_prefix != "" else parent.name.value
                    self._update_class_dependency(name, k.value.value)

    def leave_Module(self, node):
        """When leaving the module, we store the position of each global scoped node (Assigns, function def and class def)
        to allow sorting the dependencies based on their position in the code. We use the PositionProvider metadata wrapper for this.
        """
        self.global_nodes = {**self.assignments, **self.classes, **self.function_def}
        # now sort the class dependency_mapping based on the position of the nodes
        self.class_start_line = {}
        for id, node in self.global_nodes.items():
            self.class_start_line[id] = self.get_metadata(cst.metadata.PositionProvider, node).start.line


class ReplaceNameTransformer(m.MatcherDecoratableTransformer):
    """A transformer that replaces `old_name` with `new_name` in comments, string and any references.
    It should take into account name like `MyNewModel`, or `my_new_model`. Without using the AUTO_MAPPING.
    Supported renaming patterns:
        - llama -> my_new_model     and     my_new_model    -> llama
        - Llama -> MyNewModel       and     MyNewModel      -> Llama
        - LLAMA -> MY_NEW_MODEL     and     MY_NEW_MODEL    -> LLAMA
        - LLaMa -> MyNewModel       abd     MyNewModel      -> Llama
    """

    def __init__(self, old_name, new_name):
        super().__init__()
        self.old_name = old_name
        self.new_name = new_name
        self.default_name = "".join(x.title() for x in new_name.split("_"))
        self.patterns = {
            old_name: new_name,
            old_name.upper(): new_name.upper(),
            "".join(x.title() for x in old_name.split("_")): self.default_name,
        }

    def preserve_case_replace(self, text):
        # Create a regex pattern to match all variations
        regex_pattern = "|".join(re.escape(key) for key in self.patterns.keys())
        compiled_regex = re.compile(regex_pattern, re.IGNORECASE)

        def replace(match):
            word = match.group(0)
            return self.patterns.get(word, self.default_name)

        return compiled_regex.sub(replace, text)

    @m.leave(m.Name() | m.SimpleString() | m.Comment())
    def replace_name(self, original_node, updated_node):
        update = self.preserve_case_replace(updated_node.value)
        return updated_node.with_changes(value=update)


def find_classes_in_file(module: cst.Module, old_id="llama", new_id="gemma"):
    """Helper function to rename and then parse a source file using the ClassFinder"""
    transformer = ReplaceNameTransformer(old_id, new_id)
    new_module = module.visit(transformer)

    wrapper = MetadataWrapper(new_module)

    class_finder = ClassFinder(new_module)
    wrapper.visit(class_finder)
    return class_finder


DOCSTRING_NODE = m.SimpleStatementLine(
    body=[
        m.Expr(
            value=m.SimpleString(
                # match anything between """ """
                value=m.MatchIfTrue(lambda value: re.search(r"\"\"\"[\s\S]*\"\"\"", value) is not None)
            )
        )
    ]
)


class SuperTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (ParentNodeProvider,)

    def __init__(self, python_module: cst.Module, original_methods, updated_methods):
        self.python_module = python_module
        self.original_methods = original_methods
        self.updated_methods = updated_methods

    def update_body(self, existing_body, new_statements):
        """
        Helper method to update the body by removing duplicates before adding new statements.
        """
        deduplicated_new_body = []
        existing_nodes = {
            self.python_module.code_for_node(node).strip() for node in new_statements if isinstance(node, cst.CSTNode)
        }
        for stmt in existing_body:
            if self.python_module.code_for_node(stmt).strip() not in existing_nodes:
                if m.matches(stmt, DOCSTRING_NODE) and self.has_docstring:
                    continue
                deduplicated_new_body.append(stmt)
                existing_nodes.add(stmt)
            else:
                logger.info(f"\nFound duplicate {self.python_module.code_for_node(stmt)}")
        return deduplicated_new_body

    def replace_super_calls(self, node: cst.IndentedBlock, func_name: str) -> cst.CSTNode:
        """Updates the body of the input `node`'s `func_name` function by replacing calls
        to super().func_name() with the source code of the parent class' `func_name`.
        It keeps everything that is defined before `super().func_name()`.
        """
        new_body = []
        self.has_docstring = False
        for expr in node.body:
            self.has_docstring = m.matches(node.body[0], DOCSTRING_NODE)
            if m.matches(
                expr,
                m.SimpleStatementLine(
                    body=[
                        m.Return(
                            value=m.Call(func=m.Attribute(value=m.Call(func=m.Name("super")), attr=m.Name(func_name)))
                        )
                        | m.Expr(
                            value=m.Call(func=m.Attribute(value=m.Call(func=m.Name("super")), attr=m.Name(func_name)))
                        )
                    ]
                ),
            ):
                new_body.extend(self.update_body(self.original_methods[func_name].body.body, node.body))
            else:
                new_body.append(expr)
        return node.with_changes(body=new_body)

    def leave_FunctionDef(self, original_node: cst.Call, updated_node: cst.Call) -> cst.CSTNode:
        if updated_node.name.value in self.updated_methods:
            name = updated_node.name.value
            new_body = self.replace_super_calls(updated_node.body, name)
            return updated_node.with_changes(body=new_body, params=updated_node.params)
        return updated_node

    def leave_Return(self, original_node: cst.Return, updated_node: cst.Return) -> cst.CSTNode:
        """ "When a return statement is reached, it is replaced with the unrolled super code"""
        if m.matches(updated_node.value, m.Call(func=m.Attribute(attr=m.Name("super")))):
            func_def = self.get_metadata(ParentNodeProvider, original_node)
            if m.matched(func_def, m.FunctionDef()) and func_def.name.value in self.original_methods:
                updated_return_value = updated_node.value.with_changes(
                    args=[
                        cst.Arg(
                            value=cst.Call(func=cst.Name("super"), args=[cst.Arg(value=cst.Name(func_def.name.value))])
                        )
                    ]
                )
                return updated_node.with_changes(value=updated_return_value)
        return updated_node


def replace_call_to_super(class_finder: ClassFinder, updated_node: cst.ClassDef, class_name: str):
    """
    Given the `class_name`, the `updated_node`'s call to super are unpacked.

                    |    ```python                          |               |    ```python
                    |    class GemmaModel(LlamaModel):      |               |       class GemmaModel(nn.Module):
                    |        def __init__(self):            |               |           def __init__(self):
    Going from:     |            self.dropout = 0.2         |       to:     |               self.dropout = 0.2
                    |            super().__init__()         |               |               super().__init__(config)
                    |     ```                               |               |               self.padding_idx = config.pad_token_id
                                                                            |               self.vocab_size = config.vocab_size
                                                                            |               self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
                                                                            |               self.layers = nn.ModuleList(
                                                                            |                   [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
                                                                            |               )
                                                                            |               self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                                                                            |               self.gradient_checkpointing = False
                                                                            |               # Initialize weights and apply final processing
                                                                            |               self.post_init()
                                                                            |     ```
    """
    original_node = class_finder.classes[class_name]
    original_methods = {f.name.value if hasattr(f, "name") else f: f for f in original_node.body.body}
    updated_methods = {f.name.value if hasattr(f, "name") else f: f for f in updated_node.body.body}
    end_meth = []
    for name, func in original_methods.items():
        if name in updated_methods and updated_methods[name] is not None:
            new_params = updated_methods[name].params
            # Replace the method in the replacement class, preserving decorators
            kwarg_name = getattr(updated_methods[name].params, "star_kwarg", None)
            if kwarg_name and kwarg_name.name.value == "super_kwargs":
                parent_params = {k.name.value: k for k in func.params.params}
                parent_params.update({k.name.value: k for k in new_params.params[1:]})
                new_params = new_params.with_changes(
                    params=list(parent_params.values()), star_kwarg=func.params.star_kwarg
                )
            func = func.with_changes(body=updated_methods[name].body, params=new_params)
        end_meth.append(func)

    result_node = original_node.with_changes(body=cst.IndentedBlock(body=end_meth))
    temp_module = cst.Module(body=[result_node])
    new_module = MetadataWrapper(temp_module)
    new_replacement_class = new_module.visit(SuperTransformer(temp_module, original_methods, updated_methods))
    new_replacement_body = new_replacement_class.body[0].body  # get the indented block
    return original_node.with_changes(body=new_replacement_body)


class DiffConverterTransformer(CSTTransformer):
    METADATA_DEPENDENCIES = (ParentNodeProvider, ScopeProvider, PositionProvider)

    def __init__(self, python_module, new_name):
        super().__init__()
        self.model_name = (
            new_name  # name of the model being defined. Should be in the format of `llama` or `layout_xlm` our `phi3`
        )
        # fmt: off
        self.python_module = python_module  # we store the original module to use `code_for_node`
        self.transformers_imports = {}      # maps the imports name like "from transformers.models.xxx" to the parsed AST module
        self.imported_mapping = {}          # stores the name of the imported classes, with their source {"LlamaModel":"transformers.model.llama.modeling_llama"}
        self.visited_module = {}            # modules visited like "transformers.models.llama.modeling_llama"
        self.new_body = {}                  # store the new body, all global scope nodes should be added here
        self.inserted_deps = []             # nodes inserted via super dependency
        self.all_imports = []               # just stores all of the imports
        self.global_scope_index = 0
        # fmt: on

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        """When visiting imports from `transformers.models.xxx` we need to:
        1. Get the original source code
        2. Parse it into an AST Tree
        3. Add this import to `self.transformers_imports` as visited to not parse it twice
        """
        import_statement = self.python_module.code_for_node(node.module)
        if m.matches(node.module, m.Attribute()):
            for imported_ in node.names:
                _import = re.search(r"transformers\.models\..*\.(modeling|configuration)_.*", import_statement)
                if _import:
                    source = _import.groups()[0]
                    if source == "modeling" and "Config" in self.python_module.code_for_node(imported_):
                        raise ValueError(
                            f"You are importing {self.python_module.code_for_node(imported_)} from the modeling file. Import from the `configuration_xxxx.py` file instead"
                        )
                    if import_statement not in self.transformers_imports:
                        source_code = get_module_source_from_name(import_statement)
                        tree = cst.parse_module(source_code)
                        self.transformers_imports[import_statement] = tree
                    imported_class = self.python_module.code_for_node(imported_.name)
                    self.imported_mapping[imported_class] = import_statement

    def leave_FunctionDef(self, original_node, node):
        parent_node = self.get_metadata(cst.metadata.ParentNodeProvider, original_node)
        if m.matches(parent_node, m.Module()):
            self.global_scope_index += 100
            self.new_body[node.name.value] = {"insert_idx": self.global_scope_index, "node": node}
        return node

    def leave_SimpleStatementLine(self, original_node, updated_node):
        parent_node = self.get_metadata(cst.metadata.ParentNodeProvider, original_node)
        if m.matches(parent_node, m.Module()):
            if m.matches(updated_node, m.SimpleStatementLine(body=[m.Import()])):
                if parent_node not in self.all_imports:
                    self.all_imports.append(updated_node)
                return updated_node
            elif m.matches(updated_node, m.SimpleStatementLine(body=[m.ImportFrom()])):
                full_statement = self.python_module.code_for_node(updated_node.body[0].module)
                if re.search(r"transformers\.models\..*\.(modeling|configuration)_.*", full_statement):
                    return cst.RemoveFromParent()
                if parent_node not in self.all_imports:
                    self.all_imports.append(updated_node)
                return updated_node
            self.global_scope_index += 100
            if m.matches(updated_node, m.SimpleStatementLine(body=[m.Assign()])):
                # TODO This only works for single target assigns!
                node_name = updated_node.body[0].targets[0].target.value
            else:
                node_name = self.python_module.code_for_node(updated_node.body[0])
            self.new_body[node_name] = {
                "insert_idx": self.global_scope_index,
                "node": updated_node,
            }
        return updated_node

    def leave_ClassDef(self, original_node, updated_node):
        """
        1. Filter the `base` classes of this class
        If they are from `transformers.models.xx` then:
        - take the AST tree of the module it comes from and parse it with a `ClassFinder`.
        - rename all every instance of `old_name` (llama) to `new_name` (gemma)
        2. We insert the modules which the inherited base depends on. This has to be done in
        the order of the dependencies. If on is already in the new_body (because it's defined in the diff file)
        then we remove it from the new body to add it again in the correct order.
        3. Replace the calls to `super().xxxx` merging parent code
        """
        class_name = original_node.name.value
        bases = [k.value.value for k in original_node.bases if k.value.value in self.imported_mapping]
        self.global_scope_index += 100
        for super_class in bases:
            if super_class not in self.imported_mapping:
                raise ImportError(
                    f"{super_class} was not imported using `from transformers.models.xxxxx.modeling_xxxx import {super_class}"
                )

            super_file_name = self.imported_mapping[super_class]  # we need to get the parsed tree
            model_name = re.search(r"_(\S*)", super_file_name)
            if model_name:
                model_name = model_name.groups()[0]
            else:
                raise ValueError(
                    f"Tried parsing the name of the imported package from {super_file_name}, could not extract the model name"
                )

            if super_file_name not in self.visited_module:  # only extract classes once
                class_finder = find_classes_in_file(
                    self.transformers_imports[super_file_name], model_name, self.model_name
                )
                self.visited_module[super_file_name] = class_finder
            else:  # we are re-using the previously parsed data
                class_finder = self.visited_module[super_file_name]

            list_dependencies = {
                dep: class_finder.class_start_line.get(dep, 1000)
                for dep in class_finder.class_dependency_mapping.get(class_name, [])
            }

            list_dependencies = sorted(list_dependencies.items(), key=lambda x: x[1], reverse=True)
            start_insert_idx = self.global_scope_index
            for dependency, _ in list_dependencies:
                node = class_finder.global_nodes.get(dependency, None)
                if node is not None:
                    if dependency not in self.new_body:
                        start_insert_idx -= 1
                        self.new_body[dependency] = {"insert_idx": start_insert_idx, "node": node}
                    elif dependency not in self.inserted_deps:
                        # make sure the node is written after it's dependencies
                        start_insert_idx = self.new_body[dependency]["insert_idx"] - 1
                    self.inserted_deps.append(dependency)
            if len(list_dependencies) > 0:
                updated_node = replace_call_to_super(class_finder, updated_node, class_name)
        if "Config" in class_name:
            self.config_body = [updated_node]
        else:
            self.new_body[class_name] = {"insert_idx": self.global_scope_index, "node": updated_node}
        return updated_node

    def leave_If(self, original_node, node):
        parent_node = self.get_metadata(cst.metadata.ParentNodeProvider, original_node)
        if m.matches(parent_node, m.Module()):
            full_statement = self.python_module.code_for_node(original_node.test)
            if re.search(r"[\s\S]*is_.*available", full_statement):
                self.all_imports.append(node)
            elif full_statement not in self.new_body:
                self.new_body[node] = {"insert_idx": self.global_scope_index, "node": node}
        return node

    def leave_Module(self, original_node: cst.Assign, node):
        imports = {self.python_module.code_for_node(k): k for k in self.all_imports}
        dependency_imports = {}
        for visiter in self.visited_module.values():
            dependency_imports.update({self.python_module.code_for_node(k): k for k in visiter.imports.values()})
        if hasattr(self, "config_body"):
            self.config_body = list(imports.values()) + self.config_body
        dependency_imports.update(imports)
        new_body = list(dependency_imports.values())
        if len(self.new_body.keys()) > 0:
            new_body += [k[1]["node"] for k in sorted(self.new_body.items(), key=lambda x: x[1]["insert_idx"])]
        else:
            new_body = []
        return node.with_changes(body=[*new_body])


def convert_file(diff_file, cst_transformers=None):
    model_name = re.search(r"diff_(.*)(?=\.py$)", diff_file).groups()[0]
    # Parse the Python file
    with open(diff_file, "r") as file:
        code = file.read()
    module = cst.parse_module(code)
    wrapper = MetadataWrapper(module)
    if cst_transformers is None:
        cst_transformers = DiffConverterTransformer(module, model_name)
    new_mod = wrapper.visit(cst_transformers)
    ruffed_code = run_ruff(new_mod.code, True)
    formatted_code = run_ruff(ruffed_code, False)
    if len(formatted_code.strip()) > 0:
        with open(diff_file.replace("diff_", "modeling_"), "w") as f:
            f.write(AUTO_GENERATED_MESSAGE + formatted_code)

    if hasattr(cst_transformers, "config_body"):
        config_module = cst.Module(body=[*cst_transformers.config_body], header=new_mod.header)
        with open(diff_file.replace("diff_", "configuration_"), "w") as f:
            ruffed_code = run_ruff(config_module.code, True)
            formatted_code = run_ruff(ruffed_code, False)
            f.write(AUTO_GENERATED_MESSAGE + formatted_code)

    # TODO optimize by re-using the class_finder
    return cst_transformers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files_to_parse",
        default=["/Users/arthurzucker/Work/transformers/examples/diff-conversion/diff_my_new_model.py"],
        nargs="+",
        help="A list of `diff_xxxx` files that should be converted to single model file",
    )
    args = parser.parse_args()
    if args.files_to_parse == ["all"]:
        args.files_to_parse = glob.glob("src/transformers/models/**/diff_*.py", recursive=True)
    for file_name in args.files_to_parse:
        print(f"Converting {file_name} to a single model single file format")
        module_path = file_name.replace("/", ".").replace(".py", "").replace("src.", "")
        converter = convert_file(file_name)
