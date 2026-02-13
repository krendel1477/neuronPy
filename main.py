import sys
from typing import Dict, Union

from core import Refresher, UserInput, Neuron

def main():
    refresher = Refresher()
    neurons: Dict[int, Union[Neuron, UserInput]] = {}

    print("Введите определения нейронов (по одному в строке). Пустая строка для завершения:")
    lines = []
    while True:
        try:
            line = input().strip()
            if not line:
                break
            lines.append(line)
        except EOFError:
            break

    for line in lines:
        parts = line.split()
        if not parts:
            continue

        typ = parts[0]
        id_ = int(parts[1])
        level = int(parts[2])
        is_fp = bool(int(parts[3]))
        cr_val = int(parts[4]) if typ == "cls" else 0

        if typ == "inp":
            neuron = UserInput(
                id=id_,
                level=level,
                is_false_positive=is_fp,
                positive_output=1,
                negative_output=0
            )
        elif typ == "cls":
            neuron = Neuron(
                id=id_,
                level=level,
                critical_value=cr_val,
                is_false_positive=is_fp,
                positive_output=1,
                negative_output=0
            )
        else:
            print(f"Неизвестный тип нейрона: {typ}", file=sys.stderr)
            continue

        neurons[id_] = neuron

    for line in lines:
        parts = line.split()
        if not parts:
            continue

        id_ = int(parts[1])
        conn_ids_str = parts[5] if len(parts) > 5 else ""
        
        if conn_ids_str.strip().strip('"') == "":
            continue

        if not conn_ids_str:
            continue
        
        try:
            input_ids = list(map(int, conn_ids_str.split(',')))
            for inp_id in input_ids:
                if inp_id not in neurons:
                    print(f"Внимание: нейрон {id_} ссылается на отсутствующий вход {inp_id}", file=sys.stderr)
                    continue
                neurons[inp_id].add_input(neurons[id_])
        except ValueError:
            print(f"Неверный список соединений для нейрона {id_}: {conn_ids_str}", file=sys.stderr)

    for neuron in neurons.values():
        refresher.register(neuron)

    print("\nЗапуск нейронной сети с визуализацией. Для остановки нажмите Ctrl+C.\n")
    try:
        refresher.start()
    except KeyboardInterrupt:
        refresher.stop()
        print("\nСеть остановлена.")

if __name__ == "__main__":
    main()
