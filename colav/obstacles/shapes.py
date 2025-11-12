from shapely import Point, affinity, simplify

SHIP = lambda loa, beam: [
    (0, loa/2),
    (beam/2, loa/4),
    (beam/2, -loa/2),
    (-beam/2, -loa/2),
    (-beam/2, loa/4),
    (0, loa/2)
]
CIRCLE = lambda r, tol=None: [*zip(*simplify(Point(0, 0).buffer(r), tol or r/15).exterior.coords.xy)]
ELLIPSE = lambda ax_surge, ax_sway, tol=None: [*zip(*simplify(affinity.scale(Point(0, 0).buffer(1), ax_sway, ax_surge), tol or (ax_surge+ax_sway)/30).exterior.coords.xy)]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print(len(CIRCLE(2)))
    plt.figure()
    plt.plot(*zip(*CIRCLE(2)))
    plt.plot(*zip(*ELLIPSE(2, 1)))
    plt.plot(*zip(*SHIP(2, 1)))
    plt.show()