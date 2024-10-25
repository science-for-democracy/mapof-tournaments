import pytest

from mapof.tournaments.objects.TournamentCultures import registered_cultures


singular_cultures_to_test = {
    'ordered',
    'rock-paper-scissors',
}




class TestCultures:

    @pytest.mark.parametrize("culture_id", singular_cultures_to_test)
    def test_tournaments_cultures(self, culture_id):
        num_participants = 6

        culture_func = registered_cultures[culture_id]
        graph = culture_func(num_participants, None, None)
        print(registered_cultures)

        assert len(graph[0]) == num_participants
