class Connection:
    def __init__(self, prev_layer=None, is_skip_to_layer=False, skip_from_layer=None):
        self.prev_layer = prev_layer
        self.is_skip_to_layer = is_skip_to_layer
        self.skip_from_layer = skip_from_layer
