from typing import Union, Iterable

import networkx as nx
from matplotlib.animation import FuncAnimation


class Refresher:
    
    _registry: dict[int, list[Union["Neuron", "UserInput"]]]
    _registered_ids: set[int]

    def __init__(self, max_tacts: int = 1000):
        self.level_limit = 1
        self._registry = {}
        self._animation: FuncAnimation | None = None
        self._registered_ids = set()
        self.__stop_condition = False
        self.tact: int = 0
        self.max_tacts = max_tacts
        
    def register(self, neuron: Union["Neuron", "UserInput"]):
        if not isinstance(neuron, (Neuron, UserInput)):
            raise TypeError("register() requires a Neuron or UserInput instance")

        if getattr(neuron, "id", None) is None:
            raise ValueError("Neuron/UserInput must have an integer 'id'")

        if not isinstance(neuron.id, int):
            raise TypeError("'id' must be int")

        if neuron.id in self._registered_ids:
            raise ValueError(f"Duplicate registration: id={neuron.id}")

        self._registered_ids.add(neuron.id)
        self._registry.setdefault(neuron.level, []).append(neuron)
        
    def _iter_all(self) -> Iterable[Union["Neuron", "UserInput"]]:
        for level in sorted(self._registry):
            for n in self._registry[level]:
                yield n

    def _build_graph(self) -> nx.DiGraph:
        _graph = nx.DiGraph()
        for n in self._iter_all():
            _graph.add_node(n.id, ref=n, level=n.level)
            for inp in n.inputs:
                _graph.add_node(inp.id, ref=inp, level=inp.level)
                _graph.add_edge(inp.id, n.id)
        return _graph

    def start(self):
        self.__stop_condition = False
        while not self.__stop_condition:
            self._one_tact()

    def stop(self) -> None:
        self.__stop_condition = True
        
    
    def _one_tact(self) -> None:
        for neuron in self._iter_all():
            neuron.calculate_value_staged()

        for neuron in self._iter_all():
            neuron.commit()

        print(f"\n--- Tact: {self.tact} ---")
        for level in sorted(self._registry):
            for neuron in self._registry[level]:
                print(f"level:{neuron.level}; id:{neuron.id}, value:{neuron.value}, output:{neuron.get_output()}")

        
class NeuronBase:
    id: int
    level: int
    value: int
    _next_value: int
    inputs: list[Union["Neuron", "UserInput"]]
    positive_output: int 
    negative_output: int
    is_false_positive: bool
    _critical_value: int
    
    def __init__(
        self, 
        id: int,
        positive_output: int = 1,
        negative_output: int = 1,
        level: int = 0, 
        critical_value: int = 1,
        is_false_positive: bool = False,
    ):
        self.id = id
        self.value = 0
        self._next_value = 0
        self.level = level
        self.inputs = []
        self.positive_output = positive_output
        self.negative_output = negative_output
        self._critical_value = critical_value
        self.is_false_positive = is_false_positive
    
    def add_input(self, input_neuron: Union["Neuron", "UserInput"]):
        self.inputs.append(input_neuron)
        
    def get_output(self) -> int:
        if self.is_false_positive and self.value >= self._critical_value:
            return self.positive_output
        return self.positive_output if self.value >= self._critical_value else self.negative_output

    def commit(self):
        self.value = self._next_value


class UserInput(NeuronBase):
    def _prompt_int(self) -> int:
        while True:
            try:
                raw = input(f"Enter integer for input (id={self.id}): ")
                return int(raw)
            except ValueError:
                print("Please enter a valid integer!")

    def calculate_value_staged(self) -> None:
        self._next_value = self._prompt_int()


class Neuron(NeuronBase):
    def calculate_value_staged(self) -> None:
        total = 0
        for inp in self.inputs:
            inp_output = inp.get_output()
            if inp.is_false_positive and inp_output == inp.positive_output:
                self._next_value = 0
                return
            total += inp_output
        self._next_value = total
