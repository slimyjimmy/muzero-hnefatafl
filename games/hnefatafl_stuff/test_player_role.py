from games.hnefatafl_stuff.player_role import PlayerRole


def test_toggle():
    attacker = PlayerRole.ATTACKER
    assert attacker.toggle() == PlayerRole.DEFENDER
    defender = PlayerRole.DEFENDER
    assert defender.toggle() == PlayerRole.ATTACKER


def test_to_string():
    attacker = PlayerRole.ATTACKER
    assert attacker.to_string() == "Attacker"
    defender = PlayerRole.DEFENDER
    assert defender.to_string() == "Defender"
