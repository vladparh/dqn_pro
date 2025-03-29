# DQN PRO

В этом репозитории реализована модель DQN PRO из статьи [Faster Deep Reinforcement Learning with Slower Online Network](https://arxiv.org/abs/2112.05848).

## Обучение
Для запуска обучения надо воспользоваться коммандой:
```
 python -m src.train
```
## Выводы и наблюдения
DQN Pro позволяет добится больших наград по сравнению со стандартным DQN. Также он менее склонен переоценивать значения V-функции. Также стоит отметить, что в процессе обучения у DQN Pro меньше норма градиета.
