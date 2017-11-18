import numpy as np
import pandas as pd

from scipy.spatial.distance import squareform, pdist

from vivarium.framework.event import listens_for
from vivarium.framework.values import modifies_value
from vivarium.framework.population import uses_columns

class FleaBeetleComponent:
    configuration_defaults = {
        'flea_beetle': {
            'migration_distance_limit': 10,
            'reproduction_rate': 0.0002,
        }
    }


    def setup(self, builder):
        self.migration_distance_limit = builder.configuration.flea_beetle.migration_distance_limit
        self.reproduction_rate = builder.configuration.flea_beetle.reproduction_rate
        self.initial_infestation_randomness = builder.randomness('initial_flea_beetle_infestation')


    @listens_for('initialize_simulants')
    @uses_columns(['beetle_population'])
    def create_initial_population(self, event):
        draw = self.initial_infestation_randomness.get_draw(event.index)
        event.population_view.update(np.clip(draw * 20 - 10, 0, 10))


    @listens_for('time_step')
    @uses_columns(['beetle_population', 'location_x', 'location_y'])
    def beetle_growth(self, event):
        population = event.population
        dist = pdist(population[['location_x', 'location_y']].iloc[:, 1:])
        dist = squareform(dist)
        index = population.index
        dist = pd.DataFrame(dist, columns=index, index=index)
        nearby_beetles = np.where(
            dist < self.migration_distance_limit,
            population.beetle_population, 0
            ).sum(axis=0)

        beetle_growth = nearby_beetles * self.reproduction_rate

        population['beetle_population'] = (population.beetle_population + beetle_growth)

        event.population_view.update(population[['beetle_population']])


    @uses_columns(['beetle_population'])
    def severities(self, index, population_view):
        beetle_population = population_view.get(index).beetle_population

        moderate_threshold = 10
        severe_threshold = 100

        mild = (beetle_population > 0) & (beetle_population < moderate_threshold)
        moderate = (beetle_population >= moderate_threshold) & \
                   (beetle_population < severe_threshold)
        severe = (beetle_population >= severe_threshold)

        return mild, moderate, severe


    @modifies_value('potato.daily_vegetative_growth')
    def modify_vegitative_growth(self, index, rate):
        mild, moderate, severe = self.severities(index)

        rate[mild] *= 0.9
        rate[moderate] *= 0.5
        rate[severe] *= 0.25

        return rate


    @listens_for('simulation_end')
    @uses_columns(['beetle_population'])
    def metrics(self, event):
        print(f'Average beetle population: {event.population.beetle_population.mean()}')

        mild, moderate, severe = self.severities(event.index)
        print(f'Proportion of mild infestations: {mild.mean()}')
        print(f'Proportion of moderate infestations: {moderate.mean()}')
        print(f'Proportion of severe infestations: {severe.mean()}')
