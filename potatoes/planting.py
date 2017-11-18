import numpy as np
import pandas as pd

from vivarium.framework.event import listens_for
from vivarium.framework.population import uses_columns

class PlantingComponent:
    configuration_defaults = {
        'planting': {
            'density': 0.05 # Plants per square foot
        }
    }

    def setup(self, builder):
        self.density = builder.configuration.planting.density
        self.randomness = builder.randomness('placement')

    @listens_for('initialize_simulants')
    @uses_columns(['location_x', 'location_y'])
    def create_initial_population(self, event):
        garden_edge_length = np.sqrt(len(event.index) / self.density)
        x = self.randomness.get_draw(event.index, 'x') * garden_edge_length
        y = self.randomness.get_draw(event.index, 'y') * garden_edge_length
        placements = pd.DataFrame({'location_x': x, 'location_y': y})
        event.population_view.update(placements)
