from lazuli_exo.orbit import transit_duration_s


def test_hd209458b_circular_matches_archive():
    # PSCompPars: P=3.52474859 d, a/Rs=8.81, Rp/Rs=0.1209, i=86.71 deg; t14=3.072 h
    t14 = transit_duration_s(3.52474859, 8.81, 0.1209, inc_deg=86.71)
    assert abs(t14 / 3600.0 - 3.072) < 0.05


def test_eccentric_factor_shrinks_duration_at_periastron_transit():
    circ = transit_duration_s(3.0, 10.0, 0.1, inc_deg=89.0)
    # omega=90 deg: transit at periastron -> faster -> shorter
    ecc = transit_duration_s(3.0, 10.0, 0.1, inc_deg=89.0, ecc=0.3, w_deg=90.0)
    assert ecc < circ


def test_grazing_returns_zero_not_nan():
    assert transit_duration_s(3.0, 10.0, 0.1, inc_deg=80.0) == 0.0


def test_sky_position_matches_projected_separation():
    import numpy as np
    from lazuli_exo import orbit
    t = np.linspace(-2e5, 2e5, 101)
    args = (4.0, 11.4, 87.3, 0.1, 60.0)
    X, Y, Z = orbit.sky_position(t, *args)
    np.testing.assert_allclose(np.hypot(X, Y), orbit.projected_separation(t, *args))
    assert Z[50] > 0                                       # mid-transit: in front of the star
    b = 11.4 * np.cos(np.radians(87.3)) * (1 - 0.1 ** 2) / (1 + 0.1 * np.sin(np.radians(60.0)))
    assert abs(np.hypot(X[50], Y[50]) - b) < 1e-9
