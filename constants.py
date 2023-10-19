
INCREASE_LC_TIME_HEADWAY = False
LANE_WIDTH = 4  # [m]
# Lane Keeping [s]
LK_TIME_HEADWAY = 1.0
# Lane Changing [s]
LC_TIME_HEADWAY = LK_TIME_HEADWAY + (0.2 if INCREASE_LC_TIME_HEADWAY else 0)
STANDSTILL_DISTANCE = 1.0  # [m]

UNIT_MAP = {'t': 's', 'x': 'm', 'y': 'm', 'theta': 'rad', 'v': 'm/s',
            'a': 'm/s^2', 'phi': 'rad', 'gap': 'm'}

COLORS = {'gray': (0.5, 0.5, 0.5), 'red': (1.0, 0.0, 0.0),
          'green': (0.0, 1.0, 0.0), 'blue': (0.0, 0.0, 1.0),
          'purple': (0.5, 0.0, 0.5), 'orange': (1.0, 0.5, 0.0),
          'dark_blue': (0, 0.0, 0.5)}
