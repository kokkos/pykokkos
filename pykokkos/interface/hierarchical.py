class AUTO:
    pass


class TeamMember:
    def __init__(self, league_rank: int, team_rank: int):
        self._league_rank = league_rank
        self._team_rank = team_rank

    def league_rank(self) -> int:
        return self._league_rank

    def team_rank(self) -> int:
        return self._team_rank

    def team_size(self) -> int:
        pass

    def team_scratch(self, level: int) -> int:
        pass

    def team_barrier(self) -> None:
        pass

    def __index__(self) -> int:
        return self._league_rank

class PerTeam:
    def __init__(self, value):
        # Can be either a TeamMember or a scratch size value
        self.value = value
        # For backward compatibility
        if isinstance(value, TeamMember):
            self.team_member = value
        else:
            self.team_member = None


class PerThread:
    def __init__(self, value):
        # Can be either a TeamMember or a scratch size value
        self.value = value
        # For backward compatibility
        if isinstance(value, TeamMember):
            self.team_member = value
        else:
            self.team_member = None


def single(policy, functor):
    functor()
