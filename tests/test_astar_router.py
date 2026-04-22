from core.grid import Grid
from router.astar import AStarRouter


def test_route_returns_single_node_when_start_equals_goal():
    grid = Grid(width=5, height=5, layers=2)
    router = AStarRouter(grid)

    start = (2, 2, 0)
    path = router.route(start, start)

    assert path == [start]


def test_route_uses_vias_when_top_layer_is_blocked():
    grid = Grid(width=3, height=3, layers=2)
    router = AStarRouter(grid)

    # Block direct path and all top-layer detours, forcing a layer transition.
    grid.occupied[0, 1, 1] = 1
    grid.occupied[0, 0, :] = 1
    grid.occupied[0, 2, :] = 1

    path = router.route((0, 1, 0), (2, 1, 0))

    assert path is not None
    assert path[0] == (0, 1, 0)
    assert path[-1] == (2, 1, 0)

    via_count = sum(1 for i in range(1, len(path)) if path[i][2] != path[i - 1][2])
    assert via_count >= 2


def test_route_returns_none_when_all_possible_paths_are_blocked():
    grid = Grid(width=3, height=1, layers=2)
    router = AStarRouter(grid)

    # Block the middle column on both layers so there is no possible bridge.
    grid.occupied[:, 0, 1] = 1

    path = router.route((0, 0, 0), (2, 0, 0), max_iter=200)

    assert path is None
