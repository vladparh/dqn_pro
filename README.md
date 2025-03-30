# DQN PRO

В этом репозитории реализована модель DQN PRO из статьи [Faster Deep Reinforcement Learning with Slower Online Network](https://arxiv.org/abs/2112.05848).

## Описание 
В папке `src` находится весь код, необходимый для обучения. `dqn_pro_results.ipynb` - в данном блокноте реализовано построение графиков.

## Обучение
Для запуска обучения надо выполнить команды:
```
 conda create -n myenv python=3.9
 pip install -r requirements.txt
 python -m src.train
```

## Отчёт
В ходе данной работы были проведены эксперименты для 2-х игр Atari: SpaceInvaders и NameThisGame, для DQN Pro и DQN (всего 4 экперимента). Эти игры бяли выбраны, посколько в них разница между DQN Pro и DQN значительна. По результатам экпериментов построены графики: средней награды, TD loss, нормы градиента и значение V-функции в начальном состоянии.

## Выводы и наблюдения
DQN Pro позволяет добится больших наград по сравнению со стандартным DQN. Также он менее склонен переоценивать значения V-функции. Также стоит отметить, что в процессе обучения у DQN Pro меньше норма градиета.
