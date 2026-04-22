from core.grid import Grid, Net
from executor.executor import place_component, route_all_nets


def test_place_component_rejects_invalid_dimensions():
    grid = Grid(width=10, height=10, layers=2)

    result = place_component(grid, ref="U1", x=1, y=1, width=0, height=2)

    assert result["status"] == "error"
    assert "width and height" in result["msg"]


def test_place_component_rejects_invalid_layer():
    grid = Grid(width=10, height=10, layers=2)

    result = place_component(grid, ref="U1", x=1, y=1, width=2, height=2, layer=2)

    assert result["status"] == "error"
    assert "Layer" in result["msg"]


def test_place_component_rejects_pins_outside_footprint():
    grid = Grid(width=10, height=10, layers=2)

    result = place_component(
        grid,
        ref="U1",
        x=1,
        y=1,
        width=2,
        height=2,
        pins=[(2, 0)],
    )

    assert result["status"] == "error"
    assert "Pin coordinates" in result["msg"]


def test_route_all_nets_rejects_invalid_net_order_length():
    grid = Grid(width=10, height=10, layers=2)
    nets = [
        Net("N1", [(0, 0, 0), (1, 0, 0)]),
        Net("N2", [(0, 1, 0), (1, 1, 0)]),
    ]

    result = route_all_nets(grid, nets, net_order=[0])

    assert result["status"] == "error"
    assert "length" in result["msg"]


def test_route_all_nets_rejects_non_permutation_order():
    grid = Grid(width=10, height=10, layers=2)
    nets = [
        Net("N1", [(0, 0, 0), (1, 0, 0)]),
        Net("N2", [(0, 1, 0), (1, 1, 0)]),
    ]

    result = route_all_nets(grid, nets, net_order=[0, 0])

    assert result["status"] == "error"
    assert "permutation" in result["msg"]
