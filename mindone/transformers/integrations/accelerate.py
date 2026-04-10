from collections import defaultdict

from mindspore import nn


def find_tied_parameters(model: "nn.Cell", **kwargs):
    """
    Find the tied parameters in a given model.

    <Tip warning={true}>

    The signature accepts keyword arguments, but they are for the recursive part of this function and you should ignore
    them.

    </Tip>

    Args:
        model (`mindspore.nn.Cell`): The model to inspect.

    Returns:
        list[list[str]]: A list of lists of parameter names being all tied together.
    """

    del kwargs

    all_named_parameters = {}

    def collect_local_parameters(prefix: str, cell: "nn.Cell"):
        for param_name, param in cell.parameters_and_names(expand=False):
            full_name = f"{prefix}.{param_name}" if prefix else param_name
            all_named_parameters.setdefault(full_name, param)

    collect_local_parameters("", model)
    for cell_name, cell in model.name_cells().items():
        collect_local_parameters(cell_name, cell)

    tied_param_groups = defaultdict(list)
    for param_name, param in all_named_parameters.items():
        tied_param_groups[id(param)].append(param_name)

    return [sorted(names) for names in tied_param_groups.values() if len(names) > 1]
