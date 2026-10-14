"""Remove a pinned, unused ordinal conversion without altering CUDA visibility.

The upstream affinity implementation already uses the visible device's PCI
topology, then self._local_rank as fallback. Its initial int(UUID) conversion
feeds neither path. Keep the rest of that method byte-for-AST equivalent.
"""
import ast
import hashlib
import inspect
import textwrap

METHOD_SHA256 = 'f99c05c7593bd6b356af2ba4373ce69177f385bcb1af9cd9f0933aa70f8aa562'


def compile_compatible_affinity(source, globals_dict):
    source = textwrap.dedent(source)
    if hashlib.sha256(source.encode()).hexdigest() != METHOD_SHA256:
        raise RuntimeError('Unrecognized upstream NUMA method; inspect before patching')
    module = ast.parse(source)
    method = module.body[0]
    assert isinstance(method, ast.FunctionDef) and method.name == '_set_numa_affinity'
    helper, conversion = method.body[:2]
    assert isinstance(helper, ast.FunctionDef) and helper.name == 'local_rank_to_real_gpu_id'
    assert isinstance(conversion, ast.Assign) and ast.unparse(conversion) == 'rank = local_rank_to_real_gpu_id(rank)'
    rest = method.body[2:]
    # Reject future versions in which removing the conversion would affect binding.
    assert not any(isinstance(n, ast.Name) and n.id in ('rank', 'local_rank_to_real_gpu_id')
                   for statement in rest for n in ast.walk(statement))
    method.body = rest
    namespace = {}
    exec(compile(module, '<sarra-uuid-affinity>', 'exec'), globals_dict, namespace)
    return namespace['_set_numa_affinity']


def compatible_affinity(original):
    return compile_compatible_affinity(inspect.getsource(original), original.__globals__)
