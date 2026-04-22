from ai.fitness import FitnessEvaluator
from ai.genome import Genome
from core.grid import Grid, Net
from router.multi_router import MultiNetRouter


def test_multi_router_metrics_for_simple_straight_nets():
    grid = Grid(width=5, height=3, layers=2)
    nets = [
        Net("N1", [(0, 0, 0), (2, 0, 0)]),
        Net("N2", [(0, 2, 0), (2, 2, 0)]),
    ]

    result = MultiNetRouter(grid).route_all(nets)

    assert result.failed_nets == []
    assert result.completion_rate == 1.0
    assert result.total_trace_length == 4
    assert result.total_vias == 0
    assert result.total_bends == 0


def test_fitness_evaluator_sets_genome_fitness_value():
    base_grid = Grid(width=4, height=4, layers=2)
    nets = [Net("SIG", [(0, 0, 0), (3, 0, 0)])]
    genome = Genome.default(num_nets=1)

    evaluator = FitnessEvaluator(base_grid, nets)
    fitness = evaluator.evaluate(genome)

    assert fitness >= 0
    assert genome.fitness == fitness
