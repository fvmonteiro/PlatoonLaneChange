import unittest

from platoon_functionalities.q_learning import QLearningAgent


class TestQLearningAgent(unittest.TestCase):

    # Definitely too simple, but it's a start

    def setUp(self):
        self.grid_size = (5, 5)
        self.goal_nodes = [(4, 4)]
        self.obstacles = {(1, 1), (2, 2), (3, 3)}
        self.agent = QLearningAgent(
            grid_size=self.grid_size, goal_nodes=self.goal_nodes,
            obstacles=self.obstacles)

    def test_initialization(self):
        # Test if the agent initializes correctly
        self.assertEqual(self.agent._grid_size, self.grid_size)
        self.assertEqual(self.agent._goal_nodes, self.goal_nodes)
        self.assertEqual(self.agent._obstacles, self.obstacles)
        self.assertEqual(self.agent._Q[self.goal_nodes[0]], {(0, 0): 0.})

    def test_invalid_goal_node(self):
        with self.assertRaises(ValueError):
            QLearningAgent(grid_size=self.grid_size, goal_nodes=[(1, 1)],
                           obstacles=self.obstacles)

    def test_is_free_node(self):
        self.assertTrue(self.agent._is_free_node((0, 0)))
        self.assertFalse(self.agent._is_free_node((1, 1)))
        self.assertFalse(self.agent._is_free_node((5, 5)))

    def test_choose_random_action(self):
        action = self.agent._choose_random_action((0, 0))
        self.assertIn(action, self.agent._actions)

    def test_traverse_edge(self):
        new_node, cost = self.agent._traverse_edge((0, 0), (1, 0))
        self.assertEqual(new_node, (1, 0))
        self.assertEqual(cost, 1.)

        new_node, cost = self.agent._traverse_edge((0, 0), (1, 1))
        self.assertEqual(new_node, (0, 0))
        self.assertEqual(cost, 1.e6)

    def test_train(self):
        self.agent.train(start_node=(0, 0), max_episodes=10)
        self.assertIn((0, 0), self.agent._Q)

    def test_explore(self):
        self.agent._Q[(0, 0)] = {(1, 0): 1.0, (0, 1): 2.0}
        action = self.agent.explore((0, 0))
        self.assertEqual(action, (0, 1))


if __name__ == '__main__':
    unittest.main()
