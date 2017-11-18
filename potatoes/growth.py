import math

import pandas as pd
import numpy as np


from vivarium.framework.state_machine import State, Transition, Machine
from vivarium.framework.util import rate_to_probability
from vivarium.framework.event import listens_for
from vivarium.framework.values import modifies_value, produces_value
from vivarium.framework.population import uses_columns

class GrowthPhaseState(State):
    def __init__(self, name, vegetative_growth_multiplier):
        super().__init__(name)
        self.vegetative_growth_multiplier = vegetative_growth_multiplier

    def setup(self, builder):
        self.population_view = builder.population_view([self._model])

        return [self.transition_set]

    @modifies_value('potato.daily_vegetative_growth')
    def adjust_growth_rate(self, index, rate):
        population = self.population_view.get(index)
        rate *= np.where(population[self._model] == self.state_id,
                         self.vegetative_growth_multiplier, 1.0)
        return rate

class TuberFormationPhase(GrowthPhaseState):
    def __init__(self, vegetative_growth_multiplier):
        super().__init__('tuber_formation', vegetative_growth_multiplier)

    @produces_value('potato.daily_tuber_growth')
    def tuber_growth_base_rate(self, index):
        population = self.population_view.get(index)

        daily_growth_in_ounces = 1.8
        return pd.Series((population[self._model] == self.state_id) * daily_growth_in_ounces, index=index)


class RateTransition(Transition):
    def __init__(self, input_state, output_state, rate):
        super().__init__(input_state, output_state, probability_func=self.rate_probability)
        self.rate = rate

    def setup(self, builder):
        self.effective_rate = builder.rate('{}.rate'.format(self.output_state.state_id))
        self.effective_rate.source = lambda index: pd.Series(self.rate, index=index)


    def rate_probability(self, index):
        return rate_to_probability(self.effective_rate(index))

class GrowthPhaseComponent(Machine):
    def __init__(self):
        super().__init__('growth_phase')

        emergence = GrowthPhaseState('emergence', 0.0)
        emergence.allow_self_transitions()
        main_growth = GrowthPhaseState('main_growth', 1.0)
        main_growth.allow_self_transitions()
        tuber_formation = TuberFormationPhase(0.1)
        tuber_formation.allow_self_transitions()
        senescence = GrowthPhaseState('senescence', 0.0)

        # Figure out transition rate out of emergence based on duration
        emergence_duration = 14 # about 2 weeks to emerge
        main_growth_rate = 365 / emergence_duration

        emergence.transition_set.append(RateTransition(emergence, main_growth, main_growth_rate))


        # I'm picking 4 months here but in a more complete model this would 
        # depend heavily on climate and potato variety
        main_growth_duration = 4 * 30
        tuber_formation_rate = 365 / main_growth_duration

        main_growth.transition_set.append(RateTransition(main_growth, tuber_formation, tuber_formation_rate))

        tuber_formation_duration = 1 * 30
        senescence_rate = 365 / tuber_formation_duration

        tuber_formation.transition_set.append(RateTransition(tuber_formation, senescence, senescence_rate))

        self.add_states([emergence, main_growth, tuber_formation, senescence])


    @listens_for('initialize_simulants')
    @uses_columns(['growth_phase'])
    def create_initial_state(self, event):
        event.population_view.update(pd.Series('emergence', index=event.index))


    @listens_for('time_step')
    def time_step_handler(self, event):
        self.transition(event.index, event.time)


    @listens_for('time_step__cleanup')
    def time_step__cleanup_handler(self, event):
        self.cleanup(event.index, event.time)


    @listens_for('simulation_end')
    @uses_columns(['growth_phase'])
    def metrics(self, event):
        print(f'Growth Phase counts: {pd.value_counts(event.population.growth_phase)}')


class VegetativeGrowthComponent:
    def setup(self, builder):
        self.daily_growth = builder.value('potato.daily_vegetative_growth')
        self.daily_growth.source = lambda index: pd.Series(0.2, index=index)

    @listens_for('initialize_simulants')
    @uses_columns(['plant_height'])
    def create_initial_height(self, event):
        event.population_view.update(pd.Series(0.0, index=event.index))

    @listens_for('time_step')
    @uses_columns(['plant_height'])
    def growth(self, event):
        effective_daily_growth = self.daily_growth(event.index)

        population = event.population

        population['plant_height'] += effective_daily_growth

        event.population_view.update(population)

    @modifies_value('potato.daily_tuber_growth')
    @uses_columns(['plant_height'])
    def modify_tuber_growth(self, index, rate, population_view):
        pop = population_view.get(index)
        multiplier = pop.plant_height / 24

        return rate * multiplier

    @listens_for('simulation_end')
    @uses_columns(['plant_height'])
    def metrics(self, event):
        print(f'Average Plant Height: {event.population.plant_height.mean()} in')


class TuberGrowthComponent:
    def setup(self, builder):
        self.daily_growth = builder.value('potato.daily_tuber_growth')

    @listens_for('initialize_simulants')
    @uses_columns(['tuber_weight'])
    def create_initial_weight(self, event):
        event.population_view.update(pd.Series(0.0, index=event.index))

    @listens_for('time_step')
    @uses_columns(['tuber_weight'])
    def growth(self, event):
        effective_daily_growth = self.daily_growth(event.index)

        population = event.population

        population['tuber_weight'] += effective_daily_growth

        event.population_view.update(population)

    @listens_for('simulation_end')
    @uses_columns(['tuber_weight'])
    def metrics(self, event):
        print(f'Average Total Tuber Weight: {event.population.tuber_weight.mean()} oz')
