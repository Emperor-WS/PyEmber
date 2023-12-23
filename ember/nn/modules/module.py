from collections import OrderedDict
from abc import ABC, abstractmethod
import inspect
import warnings
import json
import pickle
from ember import Parameter


class Module(ABC):
    """
    Base class for all neural network modules.

    Attributes:
        training: Indicates whether the module is in training mode.
        _modules: Dictionary to store sub-modules.
        _params: Dictionary to store parameters.
        _grads: Dictionary to store gradients.
        _cache: Dictionary to store cached values.

    Methods:
        forward(self, *inputs, **kwargs): Abstract method to define forward pass.
        backward(self, *outputs, **kwargs): Abstract method to define backward pass.
        train(self): Set the module to training mode.
        eval(self): Set the module to evaluation mode.
        add(self, *modules): Add sub-modules to the module.
        parameters(self): Generator for accessing parameters.
        modules(self): Generator for accessing sub-modules.
        cache(self): Generator for accessing cached values.
        gradients(self): Generator for accessing gradients.
        zero_grad(self): Zeroes the gradients of all parameters.
        state_dict(self): Returns the state dictionary of parameters.
        load_state(self, state_dict): Loads state from a state dictionary.
        save(self, filename='model.pickle'): Saves the entire model to a file using pickle.
        save_dict(self, filename='state_dict.json'): Saves the state dictionary to a JSON file.
        cpu(self): Moves all parameters to CPU.
        cuda(self): Moves all parameters to GPU.
        get_name(self): Returns the name of the module.
        inner_repr(self): Returns a string representation for inner modules.
        __repr__(self): Returns a string representation of the module.
        __call__(self, *inputs, **kwargs): Calls the forward method.
        __setattr__(self, key, value): Custom method to handle attribute setting.
    """

    def __init__(self):
        """
        Constructor for the Module class.

        Initializes training mode and dictionaries for sub-modules, parameters, gradients, and cache.

        """
        self.training = True
        self._modules = OrderedDict()
        self._params = OrderedDict()
        self._grads = OrderedDict()
        self._cache = OrderedDict()

    @abstractmethod
    def forward(self, *inputs, **kwargs):
        """
        Abstract method for the forward pass.

        Args:
            *inputs: Variable-length argument list for inputs.
            **kwargs: Keyword arguments for inputs.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.

        """
        raise NotImplementedError

    def backward(self, *outputs, **kwargs):
        """
        Abstract method for the backward pass.

        Args:
            *outputs: Variable-length argument list for outputs.
            **kwargs: Keyword arguments for outputs.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.

        """
        raise NotImplementedError

    def train(self):
        """
        Set the module to training mode.

        Enables gradient tracking for all parameters.

        """
        self.training = True
        for param in self.parameters():
            param.requires_grad = True

    def eval(self):
        """
        Set the module to evaluation mode.

        Disables gradient tracking for all parameters.

        """
        self.training = False
        for param in self.parameters():
            param.requires_grad = False

    def add(self, *modules):
        """
        Add sub-modules to the module.

        Args:
            *modules: Variable-length argument list for sub-modules.

        """
        for module in modules:
            idx = len(self._modules)
            name = f"{idx}"
            setattr(self, name, module)
            self._modules[name] = module

    def parameters(self):
        """
        Generator for accessing parameters.

        Yields:
            Parameter: Parameters of the module.

        """
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

    def modules(self):
        """
        Generator for accessing sub-modules.

        Yields:
            Module: Sub-modules of the module.

        """
        yield from self._modules.values()

    def cache(self):
        """
        Generator for accessing cached values.

        Yields:
            dict: Cached values of the module.

        """
        for module in self.modules():
            yield module._cache

    def gradients(self):
        """
        Generator for accessing gradients.

        Yields:
            dict: Gradients of the module.

        """
        for module in self.modules():
            yield module._grads

    def zero_grad(self):
        """
        Zeroes the gradients of all parameters.

        """
        for parameter in self.parameters():
            parameter.zero_grad()

    def state_dict(self):
        """
        Returns the state dictionary of parameters.

        Returns:
            OrderedDict: State dictionary of parameters.

        """
        state = OrderedDict()
        for i, param in enumerate(self.parameters()):
            state[f'param{i}'] = param.tolist()
        return state

    def load_state(self, state_dict):
        """
        Loads state from a state dictionary.

        Args:
            state_dict (OrderedDict): State dictionary to load.

        Raises:
            UserWarning: If the shape from the state_dict does not match the model's parameter shape.

        """
        for i, param in enumerate(self.parameters()):
            data = state_dict[f'param{i}']
            if param.shape != data.shape:
                warnings.warn(f"Shape from the `state_dict` does not match model's parameter shape. "
                              f"Got {data.shape}, expected {param.shape}.", UserWarning, stacklevel=2)
            param.data = Parameter(data=data)

    def save(self, filename='model.pickle'):
        """
        Saves the entire model to a file using pickle.

        Args:
            filename (str): File name for saving the model.

        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def save_dict(self, filename='state_dict.json'):
        """
        Saves the state dictionary to a JSON file.

        Args:
            filename (str): File name for saving the state dictionary.

        """
        state = self.state_dict()
        with open(filename, 'w') as f:
            json.dump(state, f)

    def cpu(self):
        """
        Moves all parameters to CPU.

        """
        for parameter in self.parameters():
            parameter.cpu()

    def cuda(self):
        """
        Moves all parameters to GPU.

        """
        for parameter in self.parameters():
            parameter.cuda()

    def get_name(self):
        """
        Returns the name of the module.

        Returns:
            str: Name of the module.

        """
        return self.__class__.__name__

    def inner_repr(self):
        """
        Returns a string representation for inner modules.

        Returns:
            str: String representation for inner modules.

        """
        return ""

    def __repr__(self):
        """
        Returns a string representation of the module.

        Returns:
            str: String representation of the module.

        """
        # Representation similar to PyTorch
        string = f"{self.get_name()}("
        tab = "   "
        modules = self._modules
        if modules == {}:
            string += f'\n{tab}(parameters): {self.inner_repr()}'
        else:
            for key, module in modules.items():
                string += f"\n{tab}({key}): {module.get_name()}({module.inner_repr()})"
        return f'{string}\n)'

    def __call__(self, *inputs, **kwargs):
        """
        Calls the forward method.

        Args:
            *inputs: Variable-length argument list for inputs.
            **kwargs: Keyword arguments for inputs.

        Returns:
            Output: Result of the forward pass.

        """
        return self.forward(*inputs, **kwargs)

    def __setattr__(self, key, value):
        """
        Custom method to handle attribute setting.

        Args:
            key (str): Attribute key.
            value: Attribute value.

        """
        # First initialize the attribute we want to add
        self.__dict__[key] = value
        # Then update the inner dictionaries '_modules', '_params'
        if isinstance(value, Module):
            self._modules[key] = value
        elif isinstance(value, Parameter):
            self._params[key] = value
