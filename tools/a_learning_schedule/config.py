"""Restore the historical str(CfgNode) format without changing experiment values."""
import ast


def parse_archived_config(text):
    import yaml

    def restore(value):
        if isinstance(value, dict):
            return {key: restore(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(restore(item) for item in value)
        if isinstance(value, str) and value.strip().startswith('(') and value.strip().endswith(')'):
            try:
                decoded = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
            if isinstance(decoded, tuple):
                return restore(decoded)
        return value

    try:
        config = restore(yaml.safe_load(text))
    except yaml.YAMLError as error:
        raise ValueError('Invalid archived resolved_config: ' + str(error)) from error
    if not isinstance(config, dict):
        raise ValueError('Archived resolved_config must be a configuration mapping')
    return config


def validate_input_config(config):
    """Reject broken input types before launching a GPU model or data transform."""
    inputs = config.get('INPUT', {})
    if not isinstance(inputs, dict):
        raise ValueError('Archived INPUT must be a configuration mapping')
    size = inputs.get('SIZE')
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or any(type(n) is not int for n in size) or tuple(size) != (224, 224)):
        raise ValueError('Expected archived INPUT.SIZE=(224, 224) for this ViT-B/16 protocol; got %r' % (size,))
    choices = inputs.get('TRANSFORMS')
    if not isinstance(choices, (list, tuple)) or any(not isinstance(x, str) for x in choices):
        raise ValueError('Archived INPUT.TRANSFORMS must be a sequence of transform names')
    scale = inputs.get('RRCROP_SCALE')
    if 'random_resized_crop' in choices:
        if (not isinstance(scale, (list, tuple)) or len(scale) != 2
                or any(type(n) not in (int, float) for n in scale)
                or not 0 < scale[0] <= scale[1]):
            raise ValueError('Archived INPUT.RRCROP_SCALE must contain two positive ordered numbers')
    return dict(size=list(size), transforms=list(choices), random_resized_crop_scale=scale)
