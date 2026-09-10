def test_spread_is_the_difference(df):
    assert (df["spread"] - (df["imbalance_price"] - df["da_price"])).abs().max() < 1e-9


def test_system_sign_convention(df):
    assert (df.loc[df.imbalance_mwh > 0, "system_long"] == 1).all()
    assert (df.loc[df.imbalance_mwh < 0, "system_long"] == 0).all()


def test_sign_agreement_is_near_deterministic(df):
    """The finding the project is built on, pinned as a test."""
    from omie_imbalance.data import sign_agreement

    a = sign_agreement(df)
    assert a["P(spread>0 | system short)"] > 0.99
    assert a["P(spread>0 | system long)"] < 0.01
