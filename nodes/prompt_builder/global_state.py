class GlobalState:
    _instance = None
    _seed = None
    _width = None
    _height = None
    _replacements = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalState, cls).__new__(cls)
        return cls._instance

    @classmethod
    def set_seed(cls, seed):
        cls._seed = seed

    @classmethod
    def get_seed(cls):
        return cls._seed

    @classmethod
    def set_resolution(cls, width, height):
        cls._width = width
        cls._height = height

    @classmethod
    def get_resolution(cls):
        return cls._width, cls._height

    @classmethod
    def set_replacements(cls, replacements):
        cls._replacements = replacements

    @classmethod
    def get_replacements(cls):
        return cls._replacements

    @classmethod
    def reset(cls):
        cls._seed = None
        cls._width = None
        cls._height = None
        cls._replacements = None 