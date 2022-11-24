# see gh-125

# this is a bit tricky to probe in pytest
# because of the logging interception
# machinery, so running this outside of
# pytest for now

import logging
from numpy.testing import assert_equal

# imposing a default below WARNING level on other packages
# is probably bad
import pykokkos
assert_equal(logging.root.level, 30)
