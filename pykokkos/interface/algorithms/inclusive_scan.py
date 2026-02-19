from pykokkos.interface.hierarchical import TeamMember
from pykokkos.interface.views import ViewType


def inclusive_scan(team_member: TeamMember, view: ViewType, size: int = -1):
    """
    Perform an inclusive scan on a view using a team member.

    **`team_barrier()` should always be called before accessing scanned data.**

    :param team_member: the team member
    :param view: the view to scan
    :param size: (optional) the number of elements to scan
    """
    pass
